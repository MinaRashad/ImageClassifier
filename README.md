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
  
  4. Never put images in the testing set or the validation set that are in training set

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
  
  **save_dir** : Where you will save the checkpoint, the checkpoint is the result of training, you can called it "The AI itself". You can leave it and the script will save it in the current folder
  
  **epochs** : number of times the script should loop through the dataset, it depends on the dataset. Making it to big it will result in overfitting and making it too small will result in underfitting. These will lead into a drop in accuracy. Luckly, the script is designed to prevent overfitting as it will save the state with highest accuracy but making a very big number (e.g. 1000) of epochs could make the training take a long amount of time. So choose it wisely or you can just not choose it and the script will choose 10 as default

  **gpu** : If I never recommend training on a CPU! unfortunaly, the script only supports Nvidia GPUs. using a GPU will speed training exponentially from days to hours 
 
 You should not change the rest of parameters unless you know that you are doing
 
 Finally, your command should look like this:
 
 ```bash
 python3 train.py ./dataset 2 --save_dir=./ epochs=12 --gpu
 ```
 After that you should wait, firstly the script will download a VGG16 (depending on the arch argument) pretrained model that will forward its output features
 
 When if finishes training, It should create a file in the current directory by default or in the path  "chechpoint.pth" which contain all the network. 
 when typing the path, Make sure to put a forward slash "./" if you use linux or a backslash "\" on Windows.
 
## Step Three: Pridection
same as the `train.py`, you can type `python3 predict.py -h` and it will output this message:

```
usage: predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES]
                  [--gpu]
                  img checkpoint

positional arguments:
  img                   path to Image
  checkpoint            path to checkpoint

optional arguments:
  -h, --help            show this help message and exit
  --top_k TOP_K         Specify the top_k, default 5
  --category_names CATEGORY_NAMES
                        path to category names, default *print the class name
                        [the folder names]*
  --gpu                 If you want to use a GPU
```
Important Arguments:

   **img** : The path of the image you want to check
   
   **checkpoint** : the path of the checkpoint.pth produced by the `train.py`
   
   **gpu**: You dont really need a GPU here but if you are going to do a massive amount of predictions its recommended
   
   **topk**: the top **K** elements predicted. for example, top 10, top 5, top 1 ..etc

   example:
 ```bash
 python3 train.py /dataset/testing/cat/cute_cat122.png ./checkpoint.pth
 ```

## Troubleshooting

**pip3 error when installing libraries!**

Try downloading the source instead

**It is taking lots of time!**

if you are using a CPU .thats normal. Try running it on an NVIDIA GPU

if you are using a GPU, Try reducing the epochs ,hidden units or change the arch to a less accurate version e.g. VGG11
