# Based on tutorial from: https://sagemaker-examples.readthedocs.io/en/latest/frameworks/pytorch/get_started_mnist_deploy.html
import json
import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

# import to load "corrupted" (converted) images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

JSON_CONTENT_TYPE = 'application/json'
PNG_CONTENT_TYPE = 'image/png'

def create_pretrained_model():
    '''
    Create pretrained resnet15 model
    When creating our model we need to freeze all the convolutional layers which we do by their requires_grad() attribute to False. 
    We also need to add a fully connected layer on top of it which we do use the Sequential API.
    '''
    num_classes = 17

    # load the pretrained model
    model = models.resnet18(pretrained=True, progress=True)

    # freeze the different parameters of the model to use for feature extraction
    for param in model.parameters():
        param.requires_grad = False
        
    # find the number of inputs to the final layer of the network
    num_inputs = model.fc.in_features
    
    # replace the fc layer trained on imageNet with the fc for our dataset
    model.fc = nn.Linear(num_inputs, num_classes)
    
    return model

'''
Load model
'''  
def model_fn(model_dir):
    print("In model_fn. Model directory is -")
    print(model_dir)
    '''
    Initialize pretrained model
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    model=create_pretrained_model()
    model.to(device)
    
    '''
    Load model state from directory
    '''    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        print("Loading the dog-classifier model")
        checkpoint = torch.load(f , map_location =device)
        model.load_state_dict(checkpoint)
        print('MODEL-LOADED')
        print('Model loaded successfully')
    model.eval()
    return model


'''
This function handles data decoding. 
'''  
def input_fn(request_body, content_type=PNG_CONTENT_TYPE):
    print('Deserializing the input data.')
    print(f'Request body CONTENT-TYPE is: {content_type}')
    print(f'Request body TYPE is: {type(request_body)}')
    
    if content_type == PNG_CONTENT_TYPE: 
        print('Loaded PNG content')
        return Image.open(io.BytesIO(request_body))
    
    # process a URL submitted to the endpoint
    if content_type == JSON_CONTENT_TYPE:
        print('Loaded JSON content')
        print(f'Request body is: {request_body}')
        request = json.loads(request_body)
        print(f'Loaded JSON object: {request}')
        url = request['url']
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content))
    
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

'''
Run inference
''' 
def predict_fn(input_object, model):
    print('In predict fn')
    DIMENSION_PROPORTIONS = 420*1110 # height * width
    resize_height = 420 / DIMENSION_PROPORTIONS
    resize_width = 1110/420
    test_transform = transforms.Compose([
        transforms.Resize((420, 159)), # (width, height) ... divide original dimensions of 1110x420 by 420 for resize dimensions ... cuts image size in a third but visually the same
        transforms.ToTensor(), 
    ])
    
    print("Transforming input")
    input_object=test_transform(input_object)
    
    with torch.no_grad():
        print("Calling model")
        prediction = model(input_object.unsqueeze(0))
    return prediction

'''
Data post-process
''' 
def output_fn(predictions, content_type):
    print(f'Postprocess CONTENT-TYPE is: {content_type}')
    assert content_type == JSON_CONTENT_TYPE
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)