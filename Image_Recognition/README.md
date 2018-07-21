# Image Recognition 

I built an image recognition system to tackle [Kaggle's invasive species competition](https://www.kaggle.com/c/invasive-species-monitoring), in which I tried to idenfity whether 
or not an invasive species of plant are present in an image. 

## Finetuning the VGG network 
I begin by finetuning the VGG network ([link](https://medium.com/@gabrieltseng/learning-about-data-science-building-an-image-classifier-3f8252952329) to a blog post where I describe the process), a neural network which has been pre-trained on the ImageNet corpus. 

## Building other networks 
I then finetune 2 other pre-trained neural networks, ResNet (which I train over two notebooks, ResNet and ResNet2) and V3. 

## Ensembling
I combine the outputs of the 3 neural networks in Ensemble1, and then use scikit-learn to predict whether or not an image contains an invasive species. ([link](https://medium.com/towards-data-science/learning-about-data-science-building-an-image-classifier-part-2-a7bcc6d5e825) to a blog post where I describe training the other networks, and the ensembling). 

Specifically, I use the [optunity](http://optunity.readthedocs.io/en/latest/) module to find the best algorithm, which is a SVM with a polynomial kernel. 
