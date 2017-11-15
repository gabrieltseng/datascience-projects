# Learning Data Science 

In this repository, I'll keep the code I write as I learn about Data Science. 

I write about what I am learning here: 
https://medium.com/@gabrieltseng/

For all notebooks which require a GPU (anything which includes Keras or Tensorflow), I use an [AWS P2 instance](https://aws.amazon.com/ec2/instance-types/p2/). 

## 
I approached the projects in the following order (latest to earliest)

### Natural Language Processing/TwitterDisasters

I build a tweet summarizer ([COWTS](http://dl.acm.org/citation.cfm?id=2914600)), with the goal of providing a useful summary of tweets to a rescue team in a disaster scenario. This involves experimenting with Integer Linear Programming, term frequency - inverse document frequency scores and word graphs. 

  * [Post 1 on medium](https://medium.com/@gabrieltseng/summarizing-tweets-in-a-disaster-e6b355a41732) 
  * [Post 2 on medium](https://medium.com/@gabrieltseng/summarizing-tweets-in-a-disaster-part-ii-67db021d378d

### Natural Language Processing/Detecting Bullies

I train machine learning algorithms on a smaller dataset (~3000 datapoints) to recognize bullying in online discussions, as part of Kaggle's [Detecting Insults in Social Commentary](https://www.kaggle.com/c/detecting-insults-in-social-commentary) competition. By implementing word embeddings, I significantly improve the competition's best result. 

  * [Post on medium](https://medium.com/towards-data-science/using-scikit-learn-to-find-bullies-c47a1045d92f)

### Style Neural Network

I experiment with generative neural networks by building a style neural network, which takes as input two images, and outputs an image with the content of the first image and the style of the second image. I improve the original neural style network ([A Neural Network of Artistic Style](https://arxiv.org/abs/1508.06576)) by implementing two additional papers ([Incorporating Long Range Consistency in CNN based Texture Generation](https://arxiv.org/pdf/1606.01286.pdf) and [Stable and Controllable Neural Texture Synthesis and Style Transfer Using Histogram Losses](https://arxiv.org/abs/1701.08893)). 

  * [Post on medium](https://medium.com/towards-data-science/montreal-painted-by-huang-gongwang-neural-style-networks-ec1697b2ac54) 
  
### Natural Language Processing/Quora

I build a recurrant neural network based on the [GloVe](https://nlp.stanford.edu/projects/glove/) word embeddings to recognize the intent of questions posted on [Quora](https://www.quora.com) as part of Kaggle's [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) competition. 

  * [Post on medium](https://medium.com/towards-data-science/natural-language-processing-with-quora-9737b40700c8) 

### Recommender System

In this project, I use the [Movie Lens](https://grouplens.org/datasets/movielens/) dataset to explore a variety of data science tools, including dimensionality reduction and word embeddings. I build a recommender system using a recurrant neural network, and implement Google's [Wide and Deep](https://arxiv.org/abs/1606.07792) recommender neural network. 

  * [Post 1 on medium](https://medium.com/@gabrieltseng/clustering-and-collaborative-filtering-visualizing-clusters-using-t-sne-f9718e7491e6)
  * [Post 2 on medium](https://medium.com/@gabrieltseng/clustering-and-collaborative-filtering-implementing-neural-networks-bccf2f9ff988) 
  * [Post 3 on medium](https://medium.com/towards-data-science/clustering-and-collaborative-filtering-using-word-embeddings-56ee60f0575d)

### Image Recognition

In this project, I finetune and ensemble a variety of pretrained convolutional neural networks in Keras to identify invasive plant species in images, as part of Kaggle's [Invasive Species Monitoring](https://www.kaggle.com/c/invasive-species-monitoring) competition. 

  * [Post 1 on medium](https://medium.com/@gabrieltseng/learning-about-data-science-building-an-image-classifier-3f8252952329)
  * [Post 2 on medium](https://medium.com/towards-data-science/learning-about-data-science-building-an-image-classifier-part-2-a7bcc6d5e825)
  


