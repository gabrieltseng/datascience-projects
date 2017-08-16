# Style Neural Networks

[Link to blogpost](https://medium.com/@gabrieltseng/montreal-painted-by-huang-gongwang-neural-style-networks-ec1697b2ac54), where I explain in more detail what I am doing. 

In this folder, I explore style neural networks, and implement papers describing descriptive style neural networks, in which the image is iterated to minimize a loss function which describes the style and content of two input images. 

My approach exploring this was to implement papers on style neural networks. There are three papers I implemented: 

1. [A Neural Network of Artistic Style](https://arxiv.org/abs/1508.06576)
2. [Incorporating Long Range Consistency in CNN based Texture Generation](https://arxiv.org/pdf/1606.01286.pdf)
3. [Stable and Controllable Neural Texture Synthesis and Style Transfer Using Histogram Losses](https://arxiv.org/abs/1701.08893)

These notebooks require Keras, with a Tensorflow backend (as I use some tensorflow specific methods to manipulate the tensors). 

I slightly rewrite the VGG model, so that average pooling instead of max pooling occurs between layers. This modified model is in VGG16.py

## A Neural Network of Artistic Style
As this is the basis of style neural networks, this paper has already been implemented in Keras and Tensorflow. I used [this](http://blog.romanofoti.com/style_transfer/) blog's implementation. 

## Long Range Consistency 

I implement this paper over two ipython notebooks: 
1. In [Spatial_Co_Occurences.ipynb](https://github.com/GabrielTseng/LearningDataScience/blob/master/Style_Neural_Network/Spatial_Co_Occurences.ipynb), I write the loss functions describing the spatially transformed outputs, and their Gramian matrices. 
2. In [Style_Network_w_CoOccurence.ipynb](https://github.com/GabrielTseng/LearningDataScience/blob/master/Style_Neural_Network/Style_Network_w_CoOccurence.ipynb), I add the loss function to the already existing style and content loss functions. 

This yields the following style output (compared to a normal style loss function):
![](https://cdn-images-1.medium.com/max/1600/1*IZqCbKcmfF9QRXJvmwfY7Q.png)

## Histogram Losses 

I implement this paper over two ipython notebooks: 
1. In [Histogram Loss.ipynb](https://github.com/GabrielTseng/LearningDataScience/blob/master/Style_Neural_Network/Histogram%20Loss.ipynb), I implement histogram mapping for the outputs of the VGG model, and generate a loss function from them. 
2. In [Style Transfer with Histogram Loss.ipynb](https://github.com/GabrielTseng/LearningDataScience/blob/master/Style_Neural_Network/Style%20Transfer%20with%20Histogram%20Loss.ipynb), I add this to the original style and content loss functions. 

## Combining it all
I add it all together in [Final_Style_Network.ipynb](https://github.com/GabrielTseng/LearningDataScience/blob/master/Style_Neural_Network/Final_Style_Network.ipynb)
