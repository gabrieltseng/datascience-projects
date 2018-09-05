# Language Model

This folder contains code to do the following:

* Preprocess the [wikitext datasets](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset)
* Train a universal language model on the datasets. This model can be recurrent or feedforward.
* Finetune the language model on a different task. Specifically, the 
[Toxic comment classification challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

Universal models are motivated by [1](#1). The two different "universal" models are explored: a recurrent model (RecLM), 
based off [2](#2), and a convolutional model (ConvLM), motivated by [3](#3).

### 1. Preprocessing

[SpaCy](https://spacy.io/) is used to tokenize the wikitext dataset. Parallel processing is used for efficiency.

### 2a. ConvLM

The [convolutional language model](https://github.com/GabrielTseng/LearningDataScience/blob/master/natural_language_processing/language_model/lm/models/tcn.py)
consists of temporal convolutional blocks, which themselves are composed of variationally weight-dropped convolutional layers. In addition,
each block is residual.

The convolutional layers are variationally weight-dropped to mimic the variational weight drop employed in the recurrent language model. The motivation
to do this is to allow the same weights to be dropped across multiple timesteps, so that all timesteps in a convolution's output sequence will have
been processed in the same way.

Variational dropout is also used for the embedding layer.

### 2b. RecLM

The [recurrent language model](https://github.com/GabrielTseng/LearningDataScience/blob/master/natural_language_processing/language_model/lm/models/awd_lstm.py)
consists of weight dropped RNNs stacked on top of each other, as happens in [2](#2).

Variational dropout is used for the embedding layer.

## References

1. [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146)<a name="3"></a>

2. [Regularizing and Optimizing LSTM language models](https://arxiv.org/abs/1708.02182)<a name="2"></a>

3. [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modelling](https://arxiv.org/abs/1803.01271)<a name="3"></a>
