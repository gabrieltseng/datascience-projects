
# coding: utf-8

# # Kaggle Competition: Quora Question Pairs
# 
# Link to competition : 
# https://www.kaggle.com/c/quora-question-pairs
# 
# Contents: 
# 0. Setup 
# 1. My First NLP Neural Network
# 2. Cleaning the text, and using this for my second NLP neural network
# 3. Adding leaky features

# In[1]:

get_ipython().magic(u'matplotlib inline')

import re

import pandas as pd
import numpy as np 

from itertools import cycle
import string
import os
from collections import Counter
from tqdm import *

import cPickle as pickle

import bcolz

import seaborn as sns
import matplotlib.pyplot as plt

from keras.layers.embeddings import Embedding
from keras.layers import Input, merge, TimeDistributed
from keras.models import Model
from keras.layers.core import Flatten, Dropout, Dense
from keras.optimizers import Adam, Nadam
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution1D, Conv1D

from keras.regularizers import l1, l2

from keras.layers.recurrent import GRU, LSTM

from keras import backend as K

from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical
import gc


# ## Importing the data

# In[2]:

train = pd.read_csv('data/Quora/train.csv')


# In[3]:

train.head()


# ## Preprocessing

# So I want to take as inputs questions 1 and 2, and as outputs 1 if it is a duplicate, and 0 if it is not. 
# 
# The first thing I want to be able to do is take each question, and turn it into embeddings. First, I'm going to turn them into an array of words, using Keras' tokenizer. A challenge when I do this is that I need to set a max number of words. There are empty cells in question 2 (for some reason??). I'm just going to ignore them. 

# In[4]:

num = 0 
for i in train.question2.isnull():
    if i == True:
        num += 1
print str(num) + " questions in question 2 are null"


# In[5]:

num = 0 
for i in train.question1.isnull():
    if i == True:
        num += 1
print str(num) + " questions in question 1 are null"


# In[6]:

len(train)


# In[7]:

train = train[train.question2.notnull()]
train = train[train.question1.notnull()]


# In[8]:

len(train)


# In[9]:

MAX_Q1_LENGTH = np.amax(train.question1.apply(lambda x: len(x.split())))
MAX_Q2_LENGTH = np.amax(train.question2.apply(lambda y: len(y.split())))

Mean_q1_length = train.question1.apply(lambda x: len(x.split())).mean()
Mean_q2_length = train.question2.apply(lambda x: len(x.split())).mean()

print "The longest question length is " + str(np.amax(np.asarray([MAX_Q1_LENGTH, MAX_Q2_LENGTH]))) + " words"
print "The mean question lengths are " + str(Mean_q1_length) + " and " +  str(Mean_q2_length)


# In[10]:

q1_lengths = np.asarray([train.question1.apply(lambda x: len(x.split()))])[0]
q2_lengths = np.asarray([train.question2.apply(lambda x: len(x.split()))])[0]


# In[11]:

sns.distplot(q1_lengths)
sns.distplot(q2_lengths)
plt.xlabel('Question Lengths')
plt.xlim(0,50)


# 237 words takes a LONG time to train, and most questions are far shorter. 35 words is a reasonable limit, as very few questions are longer than this. 

# In[9]:

MAX_LENGTH = 35


# So I want to turn all of the words in all of my Quora questions into an array, with each word associated with an index. 

# In[15]:

tokenizer = Tokenizer(nb_words = 20000)
tokenizer.fit_on_texts(train.question1 + train.question2)


# In[16]:

q1sequence = tokenizer.texts_to_sequences(train.question1)
q2sequence = tokenizer.texts_to_sequences(train.question2)


# In[17]:

word_index = tokenizer.word_index


# In[18]:

word_index


# So we have now turned our words into indices based on how commonly they occur in the text (I **haven't** implemented word embeddings yet!). I now want all of my sequences to be the same length, since neural nets require uniformly sized inputs. 

# In[19]:

q1_input = pad_sequences(q1sequence, maxlen = MAX_LENGTH)
q2_input = pad_sequences(q2sequence, maxlen = MAX_LENGTH)


# In[24]:

labels = np.asarray(train.is_duplicate)
labels


# Awesome. So now, I can turn my arrays into my train and validation subsets. 

# In[21]:

msk = np.random.rand(len(train)) < 0.8

q1_train = q1_input[msk]
q1_valid = q1_input[~msk]

q2_train = q2_input[msk]
q2_valid = q2_input[~msk]

labels_train = labels[msk]
labels_valid = labels[~msk]


# Awesome! Now, I want to turn this array of word indices into an array of embeddings, as calculated in GloVe. 

# In[10]:

embeddings_index = {}
f = open('data/glove.6B.300d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[11]:

EMBEDDING_DIM = 300


# Now that I have the embeddings, I want to turn these into weights, where each word index in my sample of questions can be turned into its appropriate 50-dimensional embedding. 

# In[27]:

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# In[28]:

embedding_matrix[i]


# Wow that was INSANELY easier than I had made it be. 
# 
# Now I can train a neural network!

# In[29]:

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_LENGTH,
                            trainable=False)


# ## My first NLP Neural Network

# In[30]:

q1_input = Input(shape=(MAX_LENGTH,), dtype = 'int32')
q1_embedded = embedding_layer(q1_input)
q1 = BatchNormalization(axis =1)(q1_embedded)

q1 = LSTM(225, dropout_U = 0.2, dropout_W = 0.2, consume_less='gpu' )(q1)
q1 = Dropout(0.5)(q1)

#########################

q2_input = Input(shape=(MAX_LENGTH,), dtype = 'int32')
q2_embedded = embedding_layer(q2_input)
q2 = BatchNormalization(axis = 1)(q2_embedded)

q2 = LSTM(225, dropout_U = 0.2, dropout_W = 0.2,  consume_less='gpu')(q2)
q2 = Dropout(0.5)(q2)

#########################

x = merge([q1, q2], mode = 'concat')
x = BatchNormalization()(x)

x = Dense(125)(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)

preds = Dense(1, activation = 'sigmoid')(x)


# In[31]:

nlp_nn = Model([q1_input, q2_input], preds)
nlp_nn.compile(Adam(0.001), loss = 'binary_crossentropy')


# In[32]:

nlp_hist = nlp_nn.fit([q1_train, q2_train], labels_train, batch_size = 2048, nb_epoch = 5, 
                     validation_data=([q1_valid, q2_valid],labels_valid))


# In[44]:

plt.plot(range(1,6), nlp_hist.history['val_loss'], label = 'val_loss')
plt.plot(range(1,6), nlp_hist.history['loss'], label = 'loss')
plt.title("My first NLP neural network")
plt.xlabel("Epochs")
plt.ylabel("Log Loss")
plt.legend()
plt.show()


# Okay, not a great start. Considering I just picked the words from the dataset, this makes sense; there's plenty more to do.

# The first thing which could be affecting the strength of my neural network is the zeroed values in the embedding matrix (remember if a word is not in the GloVe dictionary, I just set it to zeroes). Lets take a look at some of the words which are zeroed out. 

# In[34]:

not_found = []
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is None:
        # words not found in embedding index will be all-zeros.
        not_found.append(word)


# In[37]:

print "Out of " + str(len(word_index.items())) + " words, " + str(len(not_found)) + " don't have Glove equivalents."


# So there are 36,000 words which aren't assigned values, and are all zeroed. This may be contributing to the low performance of my neural network. Let's take a look at some of the words which aren't being embedded. 

# Okay, so part of the problem is the text itself, which is full of eccentricities. Luckily, someone on Kaggle (a 'Kaggler?') has already written a method to clean up this text, so lets jump on that train. 

# ## Cleaning up the text 
# https://www.kaggle.com/currie32/the-importance-of-cleaning-text

# In[12]:

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from string import punctuation


# In[13]:

stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
              'Is','If','While','This']


# In[14]:

def text_to_wordlist(text, remove_stop_words=True, stem_words=False):
    # Clean the text, with the option to remove stop_words and to stem words.

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text) 
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text) 
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)


# In[15]:

def process_questions(question_list, questions, question_list_name, dataframe):
    '''transform questions and display progress'''
    for question in tqdm(questions):
        question_list.append(text_to_wordlist(question))


# In[16]:

train_question1 = []
process_questions(train_question1, train.question1, 'train_question1', train)


# In[17]:

train_question2 = []
process_questions(train_question2, train.question2, 'train_question2', train)


# Great. Let's try preparing the data for my neural network again. 

# In[18]:

cleaned_tokenizer = Tokenizer(nb_words = 20000)
cleaned_tokenizer.fit_on_texts(train_question1 + train_question2)


# In[19]:

clean_q1sequence = cleaned_tokenizer.texts_to_sequences(train_question1)
clean_q2sequence = cleaned_tokenizer.texts_to_sequences(train_question2)


# In[20]:

clean_word_index = cleaned_tokenizer.word_index


# In[21]:

clean_not_found = []
for word, i in clean_word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is None:
        # words not found in embedding index will be all-zeros.
        clean_not_found.append(word)


# In[22]:

print "Out of " + str(len(clean_word_index.items())) + " words, " + str(len(clean_not_found)) + " don't have Glove equivalents."


# So by reducing the number of unique words, and rewriting some of the words, we've reduced the % of words not being recognized from 37% to 29%. Success!

# In[23]:

clean_q1_input = pad_sequences(clean_q1sequence, maxlen = MAX_LENGTH)
clean_q2_input = pad_sequences(clean_q2sequence, maxlen = MAX_LENGTH)


# In[25]:

msk = np.random.rand(len(train)) < 0.8

clean_q1_train = clean_q1_input[msk]
clean_q1_valid = clean_q1_input[~msk]

clean_q2_train = clean_q2_input[msk]
clean_q2_valid = clean_q2_input[~msk]

clean_labels_train = labels[msk]
clean_labels_valid = labels[~msk]


# ## My 2nd NLP Neural Network
# Spoiler: It's identical to my first one!

# In[27]:

clean_embedding_matrix = np.zeros((len(clean_word_index) + 1, EMBEDDING_DIM))
for word, i in clean_word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        clean_embedding_matrix[i] = embedding_vector


# In[28]:

clean_embedding_layer = Embedding(len(clean_word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[clean_embedding_matrix],
                            input_length=MAX_LENGTH,
                            trainable=False)


# In[37]:

clean_q1_input = Input(shape=(MAX_LENGTH,), dtype = 'int32')
q1_embedded = clean_embedding_layer(clean_q1_input)
q1 = BatchNormalization(axis =1)(q1_embedded)

q1 = LSTM(225, dropout_U = 0.2, dropout_W = 0.2, consume_less='gpu' )(q1)
q1 = Dropout(0.5)(q1)

#########################

clean_q2_input = Input(shape=(MAX_LENGTH,), dtype = 'int32')
q2_embedded = clean_embedding_layer(clean_q2_input)
q2 = BatchNormalization(axis = 1)(q2_embedded)

q2 = LSTM(225, dropout_U = 0.2, dropout_W = 0.2,  consume_less='gpu')(q2)
q2 = Dropout(0.5)(q2)

#########################

x = merge([q1, q2], mode = 'concat')
x = BatchNormalization()(x)

x = Dense(125)(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)

clean_preds = Dense(1, activation = 'sigmoid')(x)


# In[55]:

clean_nlp_nn = Model([clean_q1_input, clean_q2_input], clean_preds)
clean_nlp_nn.compile(Adam(0.001), loss = 'binary_crossentropy')


# In[57]:

clean_nlp_hist = clean_nlp_nn.fit([clean_q1_train, clean_q2_train], clean_labels_train, batch_size = 2048, nb_epoch = 5, 
                     validation_data=([clean_q1_valid, clean_q2_valid],clean_labels_valid))


# In[61]:

plt.plot(range(1,6), clean_nlp_hist.history['val_loss'], label = 'val_loss')
plt.plot(range(1,6), clean_nlp_hist.history['loss'], label = 'loss')
plt.plot(range(1,6), nlp_hist.history['val_loss'], label = 'old val_loss')
plt.title("My second NLP neural network \n (with text cleaned!)")
plt.xlabel("Epochs")
plt.ylabel("Log Loss")
plt.legend()
plt.show()


# As Jeremy from fast.ai says, 'overfitting is only bad if its affecting your validation results'; here, the validation loss seem to be dropping quite quickly! So we're doing better, but there's still a lot of improving which can happen. 
# 
# It's worth noting that although these scores are pretty awful, I've only trained my network for 5 epochs; many of the better scores on Kaggle were trained for ~200 epochs, and I'm confident that training this network more would reduce the loss significantly, as the validation loss hasn't converged at all. 

# ## Adding leaky features

# So far, my neural network has only been built on the input words. However, the questions contain far more information, which I'm not taking advantage of at all. In this case, there are two additional features which are very predictive of whether or not a question is a duplicate: 
# 
# 1 [Frequency](https://www.kaggle.com/jturkewitz/magic-features-0-03-gain), which is the frequency of a question in the dataset
# 
# 2 [Interesection](https://www.kaggle.com/tour1st/magic-feature-v2-0-045-gain), which defines how many questions have indices common to others. 
# 
# To incorporate these into my network, I'm going to make an additional input to my neural network, 'leaky features'. 

# First, I want to isolate the questions:

# In[29]:

ques = train[['question1', 'question2']]
print(ques.shape)


# Then, I want to add them all to a dictionary: 

# In[30]:

from collections import defaultdict
q_dict = defaultdict(set)
for i in tqdm(range(ques.shape[0])):
        q_dict[ques.iloc[i].question1].add(ques.iloc[i].question2)
        q_dict[ques.iloc[i].question2].add(ques.iloc[i].question1)


# Excellent. I can now use this dictionary to define these leaky features. 

# In[31]:

def q1_freq(row):
    return(len(q_dict[row['question1']]))
    
def q2_freq(row):
    return(len(q_dict[row['question2']]))
    
def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

train['q1_q2_intersect'] = train.apply(q1_q2_intersect, axis=1, raw=True)
train['q1_freq'] = train.apply(q1_freq, axis=1, raw=True)
train['q2_freq'] = train.apply(q2_freq, axis=1, raw=True)


# In[32]:

train.head()


# Awesome. Now, I want to turn my leaks into a bunch of unique inputs. 

# In[33]:

leaks = train[['q1_q2_intersect', 'q1_freq', 'q2_freq']]


# I also want to normalize my leaks, so that there isn't a single variable which can overwhelm the neural networks (its worth noting that the embedding inputs for the words are also already normalized). I'll use sklearn's standard scaler to do this. 

# In[34]:

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(leaks)
leaks = ss.transform(leaks)


# In[35]:

leaks


# Okay! Now, I just need to split this into train and validation sets (using the same mask as above), and add a 'leaks' input to my neural network.

# In[36]:

train_leaks = leaks[msk]
valid_leaks = leaks[~msk]


# ## My 3rd NLP Neural Network 
# This one with leaky features! 

# In[50]:

leaks_input = Input(shape=(leaks.shape[1],))
leaks_dense = Dense(50, activation='relu')(leaks_input)

#########################

clean_q1_input = Input(shape=(MAX_LENGTH,), dtype = 'int32')
q1_embedded = clean_embedding_layer(clean_q1_input)

q1 = LSTM(225, dropout_U = 0.2, dropout_W = 0.2, consume_less='gpu' )(q1_embedded)
q1 = Dropout(0.5)(q1)

#########################

clean_q2_input = Input(shape=(MAX_LENGTH,), dtype = 'int32')
q2_embedded = clean_embedding_layer(clean_q2_input)


q2 = LSTM(225, dropout_U = 0.2, dropout_W = 0.2,  consume_less='gpu')(q2_embedded)
q2 = Dropout(0.5)(q2)

#########################

x = merge([q1, q2, leaks_dense], mode = 'concat')
x = BatchNormalization()(x)

x = Dense(125)(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)

clean_preds = Dense(1, activation = 'sigmoid')(x)


# In[51]:

leaky_nlp_nn = Model([clean_q1_input, clean_q2_input, leaks_input], clean_preds)


# In[47]:

leaky_nlp_nn.compile(Adam(0.001), loss = 'binary_crossentropy')


# In[48]:

leaky_nlp_hist2 = leaky_nlp_nn.fit([clean_q1_train, clean_q2_train, train_leaks], clean_labels_train, batch_size = 2048, 
                                   nb_epoch = 10, 
                     validation_data=([clean_q1_valid, clean_q2_valid, valid_leaks],clean_labels_valid))


# In[64]:

plt.plot(range(1,11), leaky_nlp_hist2.history['val_loss'], label = 'leaky val_loss')
plt.plot(range(1,11), leaky_nlp_hist2.history['loss'], label = 'leaky loss')
plt.title("My second NLP neural network \n (with text cleaned!)")
plt.xlabel("Epochs")
plt.ylabel("Log Loss")
plt.legend()
plt.show()

