import numpy as np
import time
import os
import argparse
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
import PIL
from PIL import Image
from collections import OrderedDict
from workspace_utils import active_session
from train import Classifier

def get_input_args():
    
    parser = argparse.ArgumentParser(description='Image Classifier - predict.py')
    
    parser.add_argument('--input', type = str, default = 'flowers/test/1/image_06743.jpg', help = '/path/to/image as an input for prediction', required = True)
    parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', help = 'checkpoint for model', required = True)
    parser.add_argument('--top_k', type = int, default = 5, help = 'number of top most likely classes')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'Category names as json file')
    parser.add_argument('--gpu', action = "store_true", help = 'GPU enabled instead of CPU?')
    

    return parser.parse_args()

#----------------------------------------------------------------------------------

# DONE: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(path = 'checkpoint.pth'):
    
    # Load the all parameters into checkpoint variable
    checkpoint = torch.load(path)
    
    # Then set the loadeds into model
    model = getattr(models, checkpoint['arch'])(pretrained = True)
    
    model.classifier = checkpoint['classifier']
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

#----------------------------------------------------------------------------------

def process_image(img_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # DONE: Process a PIL image for use in a PyTorch model
    
    image = Image.open(img_path)
    
    if image.width > image.height:
        image.thumbnail((10000000, 256))
    else:
        image.thumbnail((256, 10000000))
    
    
    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = (image.width + 224) / 2
    bottom = (image.height + 224) / 2
    
    image = image.crop((left, top, right, bottom))
    
    image = np.array(image)
    image = image / 255

    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image = (image - means) / stds
    
    image = image.transpose(2, 0, 1)
    
    return image

#----------------------------------------------------------------------------------

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # DONE: Implement the code to predict the class from an image file
    
    with torch.no_grad():
        model.to('cpu')
        img = process_image(image_path)
        img = torch.from_numpy(np.asarray(img).astype('float'))
        img = img.unsqueeze_(0)
        img = img.float()
    
        outputs = model(img)
        
        probs, classes = torch.exp(outputs).topk(topk)
    
        return probs[0].tolist(), classes[0].add(1).tolist()

#----------------------------------------------------------------------------------

def get_device(gpu_enabled):
    
    return torch.device("cuda:0" if gpu_enabled else "cpu")

#----------------------------------------------------------------------------------

def main():
    
    args = get_input_args()
    print("Arguments have been parsed..")
    
    category_names_file = args.category_names
    with open(category_names_file, 'r') as f:
        cat_to_name = json.load(f)
    print("Category names have been loaded..")
    
    model = load_checkpoint(args.checkpoint)
    print("Model has been built from checkpoint..")
    
    print("GPU enabled? {}".format(args.gpu))
    device = get_device(args.gpu)
    print("Device: {}".format(device))
    model.to(device)
    
    topk = args.top_k
    print("Top K: {}".format(topk))
    
    print("Predicting has been started..")
    img_path = args.input
    print("Image path: {}".format(img_path))
    probs, classes = predict(img_path, model, topk)
    
    flowers = [cat_to_name[str(x)] for x in classes]
    
    print("\n*** Result ***\n")
    for flower, prob in zip(flowers, probs):
        print("Flower: {:26}  Probability: {:.1f}%".format(flower, prob * 100))
    print("\n***\n")
#----------------------------------------------------------------------------------
    
if __name__ == '__main__':
    main()