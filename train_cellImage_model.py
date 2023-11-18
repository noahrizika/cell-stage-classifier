#TODO: Import your dependencies.
import argparse
import json
import logging
import os
import sys

#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# For Debug and Profile
import time
from smdebug import modes
import smdebug.pytorch as smd
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook


# force PIL to load truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, hook):
    print("START VALIDATING")
    if hook:
        hook.set_mode(modes.EVAL)
            
    # model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, criterion, optimizer, epoch, hook):
    print("START TRAINING")
    
    if hook:
        hook.set_mode(modes.TRAIN)
    print("after if hook")    
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        print("in for loop")
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

                
def net():

    num_classes = 17

    # load the pretrained model
    model = models.resnet50(pretrained=True)

    # freeze the different parameters of the model to use for feature extraction
    for param in model.parameters():
        param.requires_grad = False
        
    # find the number of inputs to the final layer of the network
    num_inputs = model.fc.in_features
    
    # replace the fc layer trained on imageNet with the fc for our dataset
    model.fc = nn.Linear(num_inputs, num_classes)
    
    return model


def create_data_loaders(data, batch_size):
    return torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True
    )

def main(args):

    # Initialize the model with the net() fx
    model=net()
    
    # get hook for SMDebugger
    # hook = get_hook(create_if_not_exists=True)
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)

    loss_criterion = nn.CrossEntropyLoss(ignore_index=17)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    DIMENSION_PROPORTIONS = 420*1110 # height * width
    resize_height = 420 / DIMENSION_PROPORTIONS
    resize_width = 1110/420
    transform = transforms.Compose([
        transforms.Resize((420, 159)), # (width, height) ... divide original dimensions of 1110x420 by 420 for resize dimensions ... cuts image size in a third but visually the same
        transforms.ToTensor(), 
    ])
    
    train_data = datasets.ImageFolder(root=os.path.join(args.data_dir, "train_every_nine"), transform=transform)
    test_data = datasets.ImageFolder(root=os.path.join(args.data_dir, "valid"), transform=transform)
    
    # how to include validation data for train_loader?
#     do I need to have args.batch-size and args.test-batch-size cause thats how its written in parser below? i dont think so, cause data_dir is underscored above but hyphened below
    train_loader = create_data_loaders(train_data, args.batch_size)
    test_loader = create_data_loaders(test_data, args.test_batch_size)
    
    # track loss
    if hook:
        hook.register_loss(loss_criterion)
        
    for epoch in range(1, args.epochs + 1):
        model.train() # why do I need this line here?
        train(model, train_loader, loss_criterion, optimizer, epoch, hook)
        test(model, test_loader, loss_criterion, hook)
    
    # where the model will be saved in S3
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

    
# if this file is being run from cmnd line, call this code below
if __name__=='__main__':
    parser=argparse.ArgumentParser()

    # Specify all the hyperparameters you need to use to train your model.
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    main(parser.parse_args())