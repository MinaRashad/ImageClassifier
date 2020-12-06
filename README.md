# Image Classifier

a feed forward deeb neural network with VGG architure

This classfier can be used to classify a set of images
To use it you need to follow these steps

**as an example , I will train it to identify dogs vs cats**

## Step Zero: Initialization the environment 

Install python3 on your computer
download the files into an empty folder or 

clone it with git with this command : 
```bash
git clone https://github.com/MinaRashad/ImageClassifier
```
then install the requirements with this command

    pip3 install -r requirements.txt

Finally , download your data, you can download from from [here](http://www.image-net.org/) 

## Step One: Organising the data

create a new folder inside for your dataset. 

the AI expects the folder to contain a train , test, and valid directory

inside each there must be a folder for each category

after finished have a project tree that looks like this

```
.
├── dataset
│   ├── test
│   │   ├── cat
│   │   └── dog
│   ├── train
│   │   ├── cat
│   │   └── dog
│   └── valid
│       ├── cat
│       └── dog
├── predict.py
├── README.md
├── requirements.txt
└── train.py

10 directories, 4 files
```
Now put distribute data as follows:
  
  1. Put 70%-80% of the data in the training set
  
  2. Put 10%-15% of data in the validation set
  
  3. put 10%-15% of data in the testing set
  
  4.Never put images in the training or the validation that are in training

## Step Two: Training

Now we should be ready for training! all the training will be done in `train.py`.

if you need help type `python3 train.py -h` and you should get this message:
```
usage: train.py [-h] [--save_dir SAVE_DIR] [--lr LR] [--momentum MOMENTUM]
                [--epochs EPOCHS] [--gpu] [--arch ARCH] [--hidden HIDDEN]
                dir out_features

positional arguments:
  dir                  Data directory, with test, validation and training sets
  out_features         Number of possible outputs e.g. if possible outputs are
                       [cat, dog] then it should be 2

optional arguments:
  -h, --help           show this help message and exit
  --save_dir SAVE_DIR  where you want to save the checkpoint
  --lr LR              Specify the learning rate, default 0.001
  --momentum MOMENTUM  Specify the momentum, default 0.9
  --epochs EPOCHS      Specify the number of epochs, default 10
  --gpu                If you want to use a GPU
  --arch ARCH          the Model you would like to use [vgg16, vgg13,vgg11],
                       default vgg16
  --hidden HIDDEN      Specify the number of hidden units, default 4096
  ```
  Now lets take about important arguments:
  
  **dir** : where is your dataset, in this example is is ./dataset
  
  **out_features** : Number of categories in your dataset, in this example is is either a dog or a cat so its 2
  
  **save_dir** : Where you will save the checkpoint, the checkpoint is the result of training, you can called it "The AI itself"
  
  **epochs** : number of times the script should loop through the dataset, it depends on the dataset. Making it to big it will result in overfitting and making it too small will result in underfitting. These will lead into a drop in accuracy. Luckly, the script is designed to prevent overfitting as it will save the state with highest accuracy but making a very big number (e.g. 1000) of epochs could make the training take a long amount of time. So choose it wisely or you can just not choose it and the script will choose 10 as default

  **gpu** : If I never recommend training on a CPU! unfortunaly, the script only supports Nvidia GPUs. using a GPU will speed training exponentially from days to hours 
 
 You should not change the rest of parameters unless you know that you are doing
 
 Finally, your command should look like this:
 
 ```bash
 python3 train.py dataset 2 --save_dir=./ epochs=12 --gpu
 ```
 After that you should wait
 

