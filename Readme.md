# ECEN758 project

## Overview

Code submission for ECEN758 project. Includes the code structure and instructions on training and evaluating the model.

### Project structure

    root
    ├── Readme.md
    ├── report.pdf
    ├── EDA_ML_demo.ipynb
    ├── ResNet
        ├── Resnet18_train.ipynb
        ├── demo.ipynb
        ├── cifar10-resnet18.ckpt
    ├── CCT
        ├── code            
        │   ├── Configure.py          
        │   ├── DataLoader.py         
        │   ├── ImageUtils.py 
        |   ├── loss.py
        |   ├── main.py 
        |   ├── Model.py
        |   ├── Network.py
        |   ├── Readme.md
        |   ├── requirements.txt           
        └── saved_models
        |   ├── checkpoint.pth
    ├── multi_modal
        ├── demo.ipynb


## Running Exploratory data analysis and Machine Learning models
cd root/
Open the python notebook EDA_ML_demo.ipynb to run the file

## Running the ResNet model
cd root/ResNet/

--- Training the model --- 

Open Resnet18_train.ipynb

--- Testing the model ---

Open demo.ipynb

### Notes 
1. Ensure that the cifar10-resnet18.ckpt is present while evaluating the model. 

## Running the Compact Convolutional Transformer model
cd root/CCT/code/

--- Training the model --- 

python main.py "train" ./data

--- Testing the model ---

python main.py "test" ./data

### Notes 
1. Please note that CIFAR-10 data is downloaded into ./data folder for the first time. 
2. Ensure the GPU mode is enabled as it takes lot of time in CPU mode.


## Running the Multi-modal model
cd root/multi_modal

Open demo.ipynb
