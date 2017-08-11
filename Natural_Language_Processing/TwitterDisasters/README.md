# Twitter Disasters

This project involved summarizing tweets which are spatially and geographically linked to a disaster, and creating a summary of tweets which can then be useful to rescue teams. 

In essence, this involved implementing the following two papers: 
1. [Extracting Situational Information from Microblogs during Disaster Events: a Classification-Summarization Approach](http://dl.acm.org/citation.cfm?id=2806485) 
2. [Summarizing Situational Tweets in Crisis Scenario](http://dl.acm.org/citation.cfm?id=2914600) 

The tweets used are from [Twitter as a Lifeline: Human-annotated Twitter Corpora for NLP of Crisis-related Message](https://arxiv.org/abs/1605.05894). 

Link to blog post where I describe my approach: https://medium.com/@gabrieltseng/summarizing-tweets-in-a-disaster-e6b355a41732

## [Tweets from IDs](https://github.com/GabrielTseng/LearningDataScience/blob/master/Natural_Language_Processing/TwitterDisasters/1%20-%20Tweets%20from%20IDs.ipynb) 
This notebook involved using [Twython](https://twython.readthedocs.io/en/latest/) to get the tweets from the tweet IDs (since the above corpus only stores tweet ids and user ids) 

## [Content Word Based Tweet Summarization (COWTS)](https://github.com/GabrielTseng/LearningDataScience/blob/master/Natural_Language_Processing/TwitterDisasters/2%20-%20COWTS.ipynb)

