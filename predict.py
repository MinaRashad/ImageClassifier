import matplotlib.pyplot as plt
import json
import os,glob,sys
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
import copy
import time
from PIL import Image
import argparse

'''
    Basic usage: python predict.py /path/to/image checkpoint
    Options:
        Return top K most likely classes: python predict.py input checkpoint --top_k 3
        Use a mapping of categories to real names: python predict.py input checkpoint --category_names=path/to/json
        Use GPU for inference: python predict.py input checkpoint --gpu
'''

parser = argparse.ArgumentParser()
parser.add_argument("img",help="path to Image")
parser.add_argument("checkpoint",help="path to checkpoint")
parser.add_argument("--top_k",help="Specify the top_k, default 5")
parser.add_argument("--category_names",help="path to category names, default *print the class name [the folder names]* ")
parser.add_argument("--gpu",help="If you want to use a GPU", action="store_true")

arguments = parser.parse_args()

img_path = arguments.img
checkpoint_path = arguments.checkpoint

device =  torch.device("cuda:0" if (arguments.gpu and torch.cuda.is_available()) else "cpu")
global 	maxK
global class_to_idx
def load_check_point(path):
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    checkpoint = torch.load(path,map_location=map_location)


    available_models = {'vgg16':models.vgg16,'vgg13':models.vgg13,'vgg11':models.vgg11}
    last_model = available_models[checkpoint['modelName']](pretrained=True)

    for parm in last_model.parameters():
        parm.requireds_grad = False
# changing out features
    classifier = nn.Sequential( nn.Linear(checkpoint['in_features'], checkpoint['hidden_features']) ,
                       nn.ReLU(),
                       nn.Linear(checkpoint['hidden_features'],checkpoint['out_features']),
                       nn.LogSoftmax(dim=1))
    global maxK
    global class_to_idx
    maxK = checkpoint['out_features']
    class_to_idx = checkpoint['class_to_idx']
    last_model.classifier = classifier
    last_model.load_state_dict(checkpoint['state_dict'])
    
    return last_model

model = load_check_point(checkpoint_path)
model.to(device)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    size = (256,256)
    mean = [0.485,0.486,0.406]
    std = [0.229,0.224,0.225]
    im = Image.open(image)
    
    trans = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
                                    ])
    tensor_img= trans(im)
    #tensor_img=tensor_img.unsqueeze(0)
    
    return tensor_img

def predict(image_path, model, topk,categories):
    model.eval()
    model.to(device)
    img = process_image(image_path)
    img.unsqueeze_(0)
    img=img.to(device)
    result = model.forward(img)
    predicted_flowers = []
    for flower in result.topk(topk)[1][0]:
        if arguments.category_names is None:
            predicted_flowers.append(categories[flower.item()])
        else:
            predicted_flowers.append(categories[ str( default_cat[ flower.item() ] ) ] )

    
    return np.exp(result.topk(topk)[0][0].tolist())*100,predicted_flowers


if arguments.category_names is None:
    keys =  class_to_idx.keys()
    values = class_to_idx.values()
    cat_dic = {v:k for k,v in zip(keys,values)}
else:
    with open(arguments.category_names, 'r') as f:
        cat_dic = json.load(f)
topk = (5 if maxK >5 else maxK) if (arguments.top_k is None) or not int(arguments.top_k) else int(arguments.top_k)
probs,classes = predict(img_path,model,topk,cat_dic)

print("Results:")
for prob,flower in zip(probs,classes):
    print(f"{flower}:  {round(prob,2)}%")
