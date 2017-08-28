# Twitter Disasters

This project involved summarizing tweets which are spatially and geographically linked to a disaster, and creating a summary of tweets which can then be useful to rescue teams. 

In essence, this involved implementing the following two papers: 
1. [Extracting Situational Information from Microblogs during Disaster Events: a Classification-Summarization Approach](http://dl.acm.org/citation.cfm?id=2806485) 
2. [Summarizing Situational Tweets in Crisis Scenario](http://dl.acm.org/citation.cfm?id=2914600) 

The tweets used are from [Twitter as a Lifeline: Human-annotated Twitter Corpora for NLP of Crisis-related Message](https://arxiv.org/abs/1605.05894). 

Link to blog posts where I describe my approach:

COWTS:
https://medium.com/@gabrieltseng/summarizing-tweets-in-a-disaster-e6b355a41732

COWABS: https://medium.com/@gabrieltseng/summarizing-tweets-in-a-disaster-part-ii-67db021d378d

I repeat the exercise using both NLTK and spaCy, to compare the results of using different NLP tools. 

## [Tweets from IDs](https://github.com/GabrielTseng/LearningDataScience/blob/master/Natural_Language_Processing/TwitterDisasters/1%20-%20Tweets%20from%20IDs.ipynb) 
This notebook involved using [Twython](https://twython.readthedocs.io/en/latest/) to get the tweets from the tweet IDs (since the above corpus only stores tweet ids and user ids). 

The following two files are both in the NLTK and spaCy folders: 

## Content Word Based Tweet Summarization (COWTS)

In this notebook, I identify content words in the tweets, and assign them tf-idf scores. I then use this information (and Integer Linear Programming) to generate a summary of the best tweets. 

## COntent Words Based ABstractive Summarization (COWABS)

In this notebook, I use the tweet summarization created in COWTS to generate a word graph, and word paths through this word graph. I then use Integer Linear Programming to pick the best word paths, to create a summary which goes beyond the tweets to generate a paragraph. 


