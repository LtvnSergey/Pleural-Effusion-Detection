# Pleural-Effusion-Detection
Detect pleural effusion using neural network 

# Project overview

Pleural effusion is an unusual amount of fluid around the lung. 

Pleural effusions can be investigating with magnetic resonance methods (MRI). On the image above you can see the slice image of lungs using MRI. On this slice pleural effusion looks like grey-solid area (E) inside MRI-transparent healthy part of lungs.  

Based on the MRI images of lungs and masks of pleural effusions we will try to train segmentational neural network to detect this condition 

## Contents
  - [Data description](#data-description)
  - [Evaluation](#evaluation)
  - [Neural network architecture](#neural-network-architecture)
  - [Training](#training)
  - [Results](#results)
  - [Demo](#demo)
  - [Modules and tools](#modules-and-tools)

### Data description
- 1087 one-channel images of MRI scans
- 1087 binary masks for every image indicating pleural effusion  


### Evaluation
- Dice coefficient was used to estimate quality of predicted masks.

- Dice coefficient is 2 times The area of Overlap divided by the total number of pixels in both the images
(image)

(formula)

### Neural network architecture

UNet architecture was used for segmentation problem as it achives very good performance on very different biomedical segmentation applications according to various papers and articles.

(image of arch)

U-net architecture (example for 32x32 pixels in the lowest resolution). Each blue
box corresponds to a multi-channel feature map. The number of channels is denoted
on top of the box. The x-y-size is provided at the lower left edge of the box. White
boxes represent copied feature maps. The arrows denote the different operations.

Image Courtesy: UNet [Ronneberger et al.]


### Training

- Initial data was split into training set (80%) and validation set (20%)
- Input images were normalized

- Loss function: Dice loss for binary mask
- Optimizer: Adam (momentum1 = 0.9, momentum2 = 0.999)
- Learning rate: 
- Number of epoches:

- Hardware CPU: AMD Ryzen 4000
- Hardware GPU: Nvidia RTX2060 16Gb
- Estimated training time: 

### Results


### Demo


### Modules and tools

#### Python-CNN:
Python | Pandas | Numpy | Pillow | Torch | Torchvision
