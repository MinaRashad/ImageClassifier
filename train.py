# Some of these I did not use like plt and time

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
    Basic usage: python train.py data_directory
    Prints out training loss, validation loss, and validation accuracy as the network trains
    Options:
        Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
        Choose architecture: python train.py data_dir --arch "vgg16"
        Set hyperparameters: python train.py data_dir --lr=0.001 --moment=0.9 --epochs 10
        Use GPU for training: python train.py data_dir --gpu
'''


# getting Arguments
parser = argparse.ArgumentParser()
parser.add_argument("dir",help="Data directory, with test, validation and training sets")
parser.add_argument("out_features",help="Number of possible outputs e.g. if possible outputs are [cat, dog] then it should be 2")
parser.add_argument("--save_dir",help="where you want to save the checkpoint")
parser.add_argument("--lr",help="Specify the learning rate, default 0.001")
parser.add_argument("--momentum",help="Specify the momentum, default 0.9 ")
parser.add_argument("--epochs",help="Specify the number of epochs, default 10")
parser.add_argument("--gpu",help="If you want to use a GPU", action="store_true")
parser.add_argument("--arch",help="the Model you would like to use [vgg16, vgg13,vgg11], default vgg16")
parser.add_argument("--hidden",help="Specify the number of hidden units, default 4096")

arguments = parser.parse_args()


device =  torch.device("cuda:0" if (arguments.gpu and torch.cuda.is_available()) else "cpu")

plt.ion()

# Data transforms and normalization
train_transform = transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                    ])

test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                    ])

valid_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                     ])

# Importing Data
train_dir = arguments.dir + "/train"
test_dir = arguments.dir + "/test"
valid_dir = arguments.dir + "/valid"

train_data = datasets.ImageFolder(train_dir, transform=train_transform)
valid_data = datasets.ImageFolder(valid_dir, transform=test_transform )
test_data = datasets.ImageFolder(test_dir, transform=test_transform )


train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    
available_models = {'vgg16':models.vgg16,'vgg13':models.vgg13,'vgg11':models.vgg11}

if arguments.arch not in ['vgg16','vgg13','vgg11'] and arguments.arch is not None:
    print(f"This arch is not supported : {arguments.arch}")
    print(f"Choose only from {available_models.keys()}")
    print("Exiting")
    sys.exit()

model = models.vgg16(pretrained = True) if arguments.arch not in available_models.keys() else available_models[arguments.arch](pretrained = True)
model.to(device)
print(f"Initializing on {device}")

# Freezing the parameters

for parm in model.parameters():
    parm.requireds_grad = False
    
    
# changing out features
hidden = 4096 if arguments.hidden is None or  not int(arguments.hidden) else int(arguments.hidden)

classifier = nn.Sequential( nn.Linear(model.classifier[0].in_features, hidden) ,
                       nn.ReLU(),
                       nn.Linear(hidden,int(arguments.out_features)),
                       nn.LogSoftmax(dim=1))

model.classifier = classifier

## Validation Function
def validation(model, valid_dataloader, criterion):
    validation_loss = 0
    accuracy = 0
    for images,labels in valid_dataloader:
        images, labels = images.to(device), labels.to(device)
        #images.resize_(images.shape[0],25088)

        output = model.forward(images)
        validation_loss += criterion(output,labels)

        p = torch.exp(output).max(dim=1)[1]
        eq = (labels.data == p)
        accuracy += eq.type(torch.FloatTensor).mean()
    return accuracy*100,validation_loss


def train_vgg(model,epochs,print_every, criterion , optimizer):
    # to save the best model encountered while training
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0

    print("starting!")
    steps =0
    for e in range(epochs):
        # Clearing the cache so that the GPU is not fulled
        if device == "cuda:0":
            torch.cuda.empty_cache()
        running_loss = 0
        
        model.train()
        model.to(device)
        for images,labels in train_dataloader:
            #os.system('clear')
            images, labels = images.to(device), labels.to(device)
            steps +=1
            #images.resize_(images.shape[0],25088)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                output = model.forward(images)
            
                loss = criterion(output,labels)
            loss.backward()
            
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every ==0:
                model.eval()
            
                with torch.no_grad():
                     accuracy, validation_loss = validation(model, valid_dataloader, criterion)
                
                print(f"Epoch: {e+1} of {epochs}")
                print(f"Training Loss:{running_loss/print_every}")
                print(f"Test Loss:{validation_loss/len(valid_dataloader)}")
                print(f"accuracy:{accuracy/len(valid_dataloader)}")
                #looking for best model
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_model = copy.deepcopy(model.state_dict())

                running_loss = 0
                model.train()
    print("Training finished")
    model.eval()
    model.load_state_dict(best_model)
    return model

epochs = 10 if not ( (arguments.epochs is not None) and int(arguments.epochs)) else int(arguments.epochs)
print_every = 50
criterion = nn.CrossEntropyLoss()#NLLLoss()
lr = 0.001 if not ( (arguments.lr is not None) and float(arguments.lr)) else float(arguments.lr)
momentum = 0.9 if not ( (arguments.momentum is not None) and float(arguments.momentum)) else float(arguments.momentum)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
print(" Data about the current state of training will be outputed every 50 steps\n")
input("Press Enter to start the training,\n")
model = train_vgg(model, epochs, print_every, criterion, optimizer)

if device == "cuda:0":
    torch.cuda.empty_cache()

def testing(model, test_dataloader, criterion):
    model.to(device)
    model.eval()
    torch.cuda.empty_cache()
    validation_loss = 0
    accuracy = 0
    
    
    for images,labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
            
        #images.resize_(images.shape[0],25088)
        output = model.forward(images)
        validation_loss += criterion(output,labels)
        p = torch.exp(output).max(dim=1)[1]
        eq = (labels.data == p)
        accuracy += eq.type(torch.FloatTensor).mean()
        print("Finished a batch")
    print("Accuracy is:")
    print(f"{accuracy*100/len(test_dataloader)}%")

print("Testing Began\n ")
with torch.no_grad():
    testing(model,test_dataloader, nn.CrossEntropyLoss())
chk_point = os.getcwd()+'/' if arguments.save_dir is None else arguments.save_dir
print(f"Now saving the file into {chk_point}checkpoint.pth")

# Now there are more types of models possible so we need to 
# just add the model type into the dict

checkpoint = {
            'lr':lr,
            'momentum':momentum,
            'epochs':epochs,
            'in_features':25088,
            'hidden_features':hidden,
            'out_features':int(arguments.out_features),
            'modelName':"vgg16" if arguments.arch not in available_models.keys() else arguments.arch,
            'state_dict': model.state_dict(),
	    'class_to_idx':train_data.class_to_idx}
torch.save(checkpoint,f"{chk_point}checkpoint.pth")
