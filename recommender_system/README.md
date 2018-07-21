# MovieLens

This is my exploration into the MovieLens data set. I use it to experiment with dimensionality reduction, embeddings, and using neural nets for collaborative filtering. 

## Visualizing the dataset
My approach to visualizing the data is detailed in this [blog post](https://medium.com/@gabrieltseng/clustering-and-collaborative-filtering-visualizing-clusters-using-t-sne-f9718e7491e6)

I begin by experimenting with different dimensionality reduction techniques, so that I can see how the movies are clustered, using PCA and then t-SNE. 

## Neural Networks
My approach to the Neural Networks is detailed in this [blog post](https://medium.com/@gabrieltseng/clustering-and-collaborative-filtering-implementing-neural-networks-bccf2f9ff988)

I begin by creating embeddings for the movies and for the users, and the use this as input to train an [RNN](https://github.com/GabrielTseng/LearningDataScience/blob/master/Recommender_System/RNNs.ipynb). I also visualize the activatins of the RNN layers, to see which movies are most activating different nodes. 

## Word embeddings
My use of word embeddings is detailed in this [blog post](https://medium.com/@gabrieltseng/clustering-and-collaborative-filtering-using-word-embeddings-56ee60f0575d)

Finally, I use word embeddings to add an additional input to the neural network, in the form of word tags which users can add when they rate a movie. 
