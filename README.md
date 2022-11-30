# Pleural-Effusion-Detection

Detect pleural effusion using neural network 

![Screenshot from 2022-11-30 11-08-52](https://user-images.githubusercontent.com/35038779/204744843-8a9d4867-5f3a-4078-a6e7-9d36d3c586e3.png)


# Project overview

Pleural effusion is an unusual amount of fluid around the lung. 

![image](https://user-images.githubusercontent.com/35038779/204744882-ae481140-068b-4892-9579-4bb26178ed99.png)

Pleural effusions can be investigating with magnetic resonance methods (MRI). On the image above you can see the slice image of lungs using MRI. On this slice pleural effusion looks like grey-solid area (E) inside MRI-transparent healthy part of lungs.  

Based on the MRI images of lungs and masks of pleural effusions we will try to train segmentational neural network to detect this condition 

## Contents
  - [Installation](#installation)
  - [Data description](#data-description)
  - [Preprocessing](#preprocessing)
  - [Neural network architecture](#neural-network-architecture)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Results](#results)
  - [Modules and tools](#modules-and-tools)

### Installation

To install project follow these step (instruction for Linux):
  1. Install pip: ```sudo apt install python3-pip```
    
  3. Install git: ```pip install git```
  
  4. Clone project: ``` git clone https://github.com/LtvnSergey/Pleural-Effusion-Detection.git ```
 
  5. Install Tensorboard: ``` pip install tesnorboard```


### Data description
- 1087 one-channel images of MRI scans
- 1087 binary masks for every image indicating pleural effusion  

![image](https://user-images.githubusercontent.com/35038779/204751501-a64f5898-fba1-4dbd-9d6e-ce6c1fb0a188.png)

- [Raw images](https://github.com/LtvnSergey/Pleural-Effusion-Detection/tree/main/data/raw/images) were provided in special ".dcm" format which is used in medcine.
- [Raw masks](https://github.com/LtvnSergey/Pleural-Effusion-Detection/tree/main/data/raw/images)  were stored as compressed NIFTI files (.nii.gz). 


- Raw data were read and stored as numpy arrays in compressed ".npz" format in [processed data](https://github.com/LtvnSergey/Pleural-Effusion-Detection/tree/main/data/processed) folder. That way not only data occupies less space, but these types of files supports 'lazy IO' which means we can get access to specific image/mask by index and we dont need to read the whole dataset and store it to operative memory.


- Also special [meta-file](https://github.com/LtvnSergey/Pleural-Effusion-Detection/blob/main/data/processed/meta_file.csv) was created which inlcudes indexes of all images and masks stored in '.npz' files, corresponding patient id folder and flag that indicates if mask is empty or not.


- Turns out 43% masks are non zero, which means 57% of samples are without plueral effusion. During split on train and validation sets this has been taken into account.


- In the notebook [Input_Analysis.ipynb](https://github.com/LtvnSergey/Pleural-Effusion-Detection/blob/main/notebook/Pleural-Effusion-Detection%20-%20Input%20Analysis.ipynb)  you can find additional information about input data and statistics


### Preprocessing

- Input image data was normilized before training

- 80% data were used for training and 20% for validation. Both sets had equal amount of zero and non-zero masks 


### Neural network architecture

UNet architecture was used to solve this segmentation problem as it achives very good performance on very different biomedical segmentation applications according to various papers and articles.

- In this article you can find usefull and detailed information about U-Net architecture: ['U-Net: Convolutional Networks for Biomedical
Image Segmentation'](https://arxiv.org/pdf/1505.04597.pdf)


- U-net architecture was used for training with following parameters:

  - Resnet 18 as an encoder. 
    More information about ResNet18 you can find in this article: ['Deep Residual Learning for Image Recognition'](https://arxiv.org/pdf/1512.03385.pdf)

  - 9 million total parameters

  - Pre-trained Imagenet weights. Due to low amount (~1000) of high-resolution data (512x512) its better to try to use transfer learning.

  - 1 channel input data

  - 1 channel output corresponding to predicted binary mask

- Model defenition is in file [model.py](https://github.com/LtvnSergey/Pleural-Effusion-Detection/blob/main/model.py)

- It is possible to investigate the  whole graph of the net in the tab 'Graph' of Tesnorboard.
To do so, run this command in the project directory:  

  ```tensorboard --logdir='runs' ```

![Example of model graph in Tensorboard](https://user-images.githubusercontent.com/35038779/204790041-33e0c8ec-3cae-42ce-8113-404a47a4e002.png)




### Training

Following parameters were used during training:

- Loss function: Dice loss for binary mask
- Optimizer: Adam (momentum1 = 0.9, momentum2 = 0.999)
- Learning rate: 0.0001
- Batch size: 8
- Number of epoches: 26

Hardware used:
- CPU: AMD Ryzen 4000
- GPU: Nvidia RTX2060 16Gb

Estimated time:
- Total: 1860 sec
- Per epoch: ~61 sec



### Evaluation
- Dice coefficient was used to estimate quality of predicted masks.

- Dice coefficient is 2 times The area of Overlap divided by the total number of pixels in both the images

<img src="https://user-images.githubusercontent.com/35038779/204766842-4fe0044e-a1f8-4f56-83df-859836d86ef3.png" width="400">
<em>source: [Biomedical Image Segmentation - U-Net](https://jinglescode.github.io/2019/11/07/biomedical-image-segmentation-u-net/)</em>



### Modules and tools

#### Python-CNN:
Python | Pandas | Numpy | Pillow | Torch | Torchvision
