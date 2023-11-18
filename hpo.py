#TODO: Import your dependencies.
import argparse
import json
import logging
import os
import sys
import csv

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader


# Add this line of you don't want your job to fail unexpected after multiple hours of training
# https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def test(model, test_loader, criterion, device):
    '''
    This function takes a model and a testing data loader and will get the test accuray/loss of the model
    '''    
    print("Testing Model on Whole Testing Dataset")    
    model.eval()
    running_loss=0.0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)        
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += float(loss.item() * inputs.size(0))
        running_corrects += float(torch.sum(preds == labels.data))

    total_loss = float(running_loss) // float(len(test_loader.dataset))
    total_acc = float(running_corrects) // float(len(test_loader.dataset))
    
    # here works regexp for "Testing Loss"
    print(f"Testing Loss: {total_loss}")
    print(f"Testing Accuracy: {total_acc}")

def train(model, train_loader, validation_loader, criterion, optimizer, device):
    '''
    This function takes a model and data loaders for training and will get train the model
    '''    
    # Number of epochs don't matter here - hpo tuning will be stopped after first epoch (see end of loop)
    epochs=5
    best_loss=float(1e6)
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch: {epoch}, Phase: {phase}")
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0.0
            running_samples=0
            
            total_samples_in_phase = len(image_dataset[phase].dataset)

            for inputs, labels in image_dataset[phase]:
                inputs=inputs.to(device)
                labels=labels.to(device)                  
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += float(loss.item() * inputs.size(0))
                running_corrects += float(torch.sum(preds == labels.data))
                running_samples+=len(inputs)

                accuracy = float(running_corrects)/float(running_samples)
                print("Epoch {}, Phase {}, Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                        epoch,
                        phase,
                        running_samples,
                        total_samples_in_phase,
                        100.0 * (float(running_samples) / float(total_samples_in_phase)),
                        loss.item(),
                        running_corrects,
                        running_samples,
                        100.0*accuracy,
                    ))
                 
                #NOTE: Comment lines below to train and test on whole dataset
                if (running_samples>(0.1*total_samples_in_phase)):
                    break
                
                
            epoch_loss = float(running_loss) // float(running_samples)
            epoch_acc = float(running_corrects) // float(running_samples)
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1


            print('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase,
                                                                           epoch_loss,
                                                                           epoch_acc,
                                                                           best_loss)) 
            
        # Break training when loss starts to increase. I am not sure if this is required since hpo tuning should handle these issues.
        if loss_counter==1:
            print("Finish training because epoch loss increased")            
            break
        
        # Comment out these lines if you actually would like to train the model
        if epoch==0:
            print("Finish training on Epoch 0")
            break
    return model
    
def create_pretrained_model():
    '''
    Create pretrained resnet50 model
    When creating our model we need to freeze all the convolutional layers which we do by their requires_grad() attribute to False. 
    We also need to add a fully connected layer on top of it which we do use the Sequential API.
    '''
    model = models.resnet50(pretrained=True, progress=True)

    for param in model.parameters():
        param.requires_grad = False   
        
    # find the number of inputs to the final layer of the network
    num_inputs = model.fc.in_features
    print('NUM_INPUTS: ...is below')
    print(num_inputs)

    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 17))
    return model


def create_data_loaders(data, batch_size):
    # Modernized data loaders to skip downloading dataset every time
    train_data_path = os.path.join(data, 'train_every_nine')
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')

    
    # https://discuss.pytorch.org/t/runtimeerror-stack-expects-each-tensor-to-be-equal-size-but-got-3-224-224-at-entry-0-and-3-224-336-at-entry-3/87211/9
    DIMENSION_PROPORTIONS = 420*1110 # height * width
    resize_height = 420 / DIMENSION_PROPORTIONS
    resize_width = 1110/420
    transform = transforms.Compose([
        transforms.Resize((420, 159)), # (width, height) ... divide original dimensions of 1110x420 by 420 for resize dimensions ... cuts image size in a third but visually the same
        transforms.ToTensor(), 
    ])
    
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=transform)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True) 
    
    return train_data_loader, test_data_loader, validation_data_loader

def main(args):
    print(f'Hyperparameters: Learning Rate: {args.learning_rate}, Batch Size: {args.batch_size}, Epochs: {args.epochs}')
    print(f'Database Path: {args.data_dir}')
    
    '''
    Create data loaders
    '''    
    train_loader, test_loader, validation_loader=create_data_loaders(args.data_dir, args.batch_size)
    
    '''
    Initialize pretrained model
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    model=create_pretrained_model()
    model.to(device)
    
    '''
    Create loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss(ignore_index=17)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
    '''
    Call the train function to start training model
    '''
    print("Starting Model Training")
    model=train(model, train_loader, validation_loader, criterion, optimizer, device)
    
    '''
    Test the model to see its accuracy
    '''    
    print("Testing Model")
    test(model, test_loader, criterion, device)
    
    '''
    Save the trained model
    '''    
    print("Saving Model")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))
    
    
if __name__=='__main__':
    '''
    All the hyperparameters needed to use to train your model.
    '''
    # Training settings
    parser = argparse.ArgumentParser(description="Udacity AWS ML project 3 - HPO tuning")
    # parser.add_argument(
    #     "--batch-size",
    #     type=int,
    #     default=64,
    #     metavar="N",
    #     help="input batch size for training (default: 64)",
    # )
    # parser.add_argument(
    #     "--learning_rate", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    # )
    # parser.add_argument('--data_path', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    # parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    # from hpo_for_tuner.py:
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
    
    args = parser.parse_args()
    print(args)
    
    main(args)