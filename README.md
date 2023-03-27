# Image Classification using LeNet, VGGNet and ResNet on Fashion MNIST, CIFAR-10 and EuroSAT datasets

This project aims to show how to train LeNet, VGGNet and ResNet Convolutional Network models to classify images from Fashion MNIST, CIFAR-10 and EuroSAT datasets from scratch using PyTorch. Each model were trained on each dataset so there are 9 models in total. No pre-trained models were used.

## Description

### Datasets

#### Fashion MNIST Dataset

[Fashion MNIST](https://pytorch.org/vision/main/generated/torchvision.datasets.FashionMNIST.html) is a dataset of images of clothing items, designed to serve as a more complex replacement for the original MNIST dataset of handwritten digits. It consists of 70,000 grayscale images, each of size 28x28 pixels, and is divided into 60,000 training examples and 10,000 test examples.

The dataset contains 10 different classes of clothing items, each with 6,000 images in the training set and 1,000 images in the test set. The classes are as follows: (1) T-shirt/top, (2) Trouser, (3) Pullover, (4) Dress, (5) Coat, (6) Sandal, (7) Shirt, (8) Sneaker, (9) Bag and (10) Ankle boot.

#### CIFAR-10 Dataset

The [CIFAR-10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html) dataset is a widely-used benchmark dataset for image classification tasks in computer vision. It consists of 60,000 color images in 10 different classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images, and were collected from a variety of sources.

The images are of size 32x32 pixels and are labeled with one of the following classes: (1) Airplane, (2) Automobile, (3) Bird, (4) Cat, (5) Deer, (6) Dog, (7) Frog, (8) Horse, (9) Ship and (10) Truck.

#### EuroSAT Dataset

The [EuroSAT](https://pytorch.org/vision/main/generated/torchvision.datasets.EuroSAT.html) dataset is a benchmark dataset for land use and land cover classification tasks in remote sensing and computer vision. It consists of 27,000 satellite images of size 64x64 pixels, covering 10 different land use and land cover classes commonly found in Europe. The images were acquired by the Sentinel-2 satellite and preprocessed to ensure consistency in terms of resolution, band selection, and geometric correction.

The classes are as follows: (1) Annual Crop, (2) Forest, (3) Herbaceous Vegetation, (4) Highway, (5) Industrial, (6) Pasture, (7) Permanent Crop, (8) Residential, (9), River and (10) Sea/Lake.

### CNN Architecture

The CNN architectures used are as follows:
- **LeNet**: Based on the default LeNet5 architecture.
- **VGGNet**: A smaller version of VGGNet with 3 blocks of 3 convolutional layers each. Each convolutional layer has batch normalization and ReLU non-linearity. Each convolutional block is followed by a max pooling layer (in the 1st 2 blocks) and a global average pooling layer (in the 3rd block). Finally, there is a fully-connected layer for the label classification.
- **ResNet**: A smaller version of ResNet with four blocks of convolutional layers. The convolutional blocks have batch normalization, ReLU non-linearity and skip connections. After the convolutional blocks, there is a global average pooling layer followed by a fully-connected layer for the label classification.

### Technologies

- Python
- PyTorch
- Matplotlib

## Results

### Fashion MNIST

Train and test accuracy curves of LeNet, VGGNet and ResNet on Fashion MNIST dataset:

LeNet: **88.63%**
VGGNet: **92.68**
ResNet: **92.18%**

<img src="logs/fashionmnist/lenet_acc_hist.png" width="32%" alt="LeNet" />
<img src="logs/fashionmnist/vggnet_acc_hist.png" width="32%" alt="VGGNet" />
<img src="logs/fashionmnist/resnet_acc_hist.png" width="32%" alt="ResNet" />

Train and test loss curves of LeNet, VGGNet and ResNet on Fashion MNIST dataset:
<img src="logs/fashionmnist/lenet_loss_hist.png" width="32%" alt="LeNet" />
<img src="logs/fashionmnist/vggnet_loss_hist.png" width="32%" alt="VGGNet" />
<img src="logs/fashionmnist/resnet_loss_hist.png" width="32%" alt="ResNet" />


Random images grouped by the predicted class. Incorrectly predicted images have red borders and the correct label is displayed in red font.

LeNet predictions:
<img src="logs/fashionmnist/lenet_images.png" alt="LeNet" />

VGGNet predictions:
<img src="logs/fashionmnist/vggnet_images.png" alt="VGGNet" />

ResNet predictions:
<img src="logs/fashionmnist/resnet_images.png" alt="ResNet" />

Classification report per classes:

LeNet:
| Label | Precision | Recall | F1-score |
| ----- | --------- | ------ | -------- |
| T-shirt/top | 0.84 | 0.82 | 0.83 | 
| Trouser | 0.98 | 0.97 | 0.98 | 
| Pullover | 0.86 | 0.82 | 0.84 | 
| Dress | 0.81 | 0.95 | 0.87 | 
| Coat | 0.81 | 0.81 | 0.81 | 
| Sandal | 0.97 | 0.96 | 0.96 | 
| Shirt | 0.72 | 0.65 | 0.68 | 
| Sneaker | 0.93 | 0.96 | 0.95 | 
| Bag | 0.97 | 0.97 | 0.97 | 
| Ankle boot | 0.97 | 0.95 | 0.96 | 

VGGNet:
| Label | Precision | Recall | F1-score |
| ----- | --------- | ------ | -------- |
| T-shirt/top | 0.85 | 0.89 | 0.87 | 
| Trouser | 0.99 | 0.99 | 0.99 | 
| Pullover | 0.9 | 0.9 | 0.9 | 
| Dress | 0.94 | 0.93 | 0.94 | 
| Coat | 0.85 | 0.93 | 0.88 | 
| Sandal | 0.99 | 0.99 | 0.99 | 
| Shirt | 0.82 | 0.73 | 0.77 | 
| Sneaker | 0.96 | 0.98 | 0.97 | 
| Bag | 0.99 | 0.98 | 0.99 | 
| Ankle boot | 0.98 | 0.95 | 0.97 | 

ResNet:
| Label | Precision | Recall | F1-score |
| ----- | --------- | ------ | -------- |
| T-shirt/top | 0.88 | 0.86 | 0.87 | 
| Trouser | 0.98 | 0.99 | 0.99 | 
| Pullover | 0.92 | 0.83 | 0.87 | 
| Dress | 0.93 | 0.93 | 0.93 | 
| Coat | 0.84 | 0.91 | 0.87 | 
| Sandal | 0.99 | 0.99 | 0.99 | 
| Shirt | 0.77 | 0.79 | 0.78 | 
| Sneaker | 0.96 | 0.98 | 0.97 | 
| Bag | 0.99 | 0.99 | 0.99 | 
| Ankle boot | 0.98 | 0.96 | 0.97 | 

## Getting Started

