# Torch_CV_Utils
Code to help with exploration of CV Deep Learning Training using CIFAR10 Dataset &amp; Various Models.

## **Folder Structure**

|── config \
|── ── config.yaml \
|──  model  \
|── ── resnet.py  \
|── utils  \
|── ── __init__.py   
|── ──  train.py  \
|── ──  test.py  \
|── ──  augmentation.py   
|── ── helpers.py   
|── ──  gradcam.py  
|── ── data_handling.py   
|── main.py       
|── README.md   

## Configuration File
Stores configuration options for augmentations (using albumentations library) and for learning rate scheduling.

## Utils

### Train & Test
Training and testing codes with options to input multiple optimizers, schedulers and loss criteria.

### Augmentation
augmentation.py contains individual augmentation functions for different tasks/assignments. Requires config dictionary input for augmentation config options.

### Gradcam
Generalized gramcam implementation to visualize what the network "sees" or is focused upon at different layers.
Implementation influenced by:
[Implementing Grad-CAM in PyTorch](https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82)
[Grad-CAM with PyTorch](https://github.com/kazuto1011/grad-cam-pytorch/tree/fd10ff7fc85ae064938531235a5dd3889ca46fed)

## Data Handling
Functions to download data, calculate dataset statistics and create test and train loaders using defined augmentations and downloaded data.
Functions to display sample data are also present.

## Helpers
Miscellaneous functions to:
1. Check for presence of GPU in runtime and create a device accordingly
2. Loading configuration variables from yaml file as a dictionary
3. Plotting loss and accuracy metrics post training
4. Finding accuracy of trained model per class
5. Creating a list of misclassified images and corresponding labels
6. Plotting misclassified images with labels alongside images
