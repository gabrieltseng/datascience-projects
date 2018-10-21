import numpy as np
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor

from spacy.attrs import IS_UPPER
from spacy.lang.en import English

from .base import Tokenizer, BOS_TOKEN
from ..utils import chunk


class IMDBTokenizer(Tokenizer):
    """
    Loads and tokenizes the IMDB dataset
    """
    dimension = 2
    sentiment2int = {'pos': 1, 'neg': 0}

    def __init__(self, filepaths, processes=6, parallelism=4,
                 chunks=1000):
        super().__init__(filepaths, processes, parallelism, chunks)

    def read_articles(self):
        """
        Returns a list of articles
        """

        # Expect the pos to be in filepath/pos, and neg to be in filepath/neg
        reviews = []
        labels = []
        for filepath in self.filepaths:
            for sentiment in ['pos', 'neg']:
                sentiment_filepath = filepath/sentiment
                for file in sentiment_filepath.glob('*.txt'):
                    with file.open() as f:
                        data = f.read()
                        labels.append(self.sentiment2int[sentiment])
                        reviews.append(data)
        print('Loaded {} articles'.format(len(reviews)))
        self.labels = np.asarray(labels)
        return reviews

    def get_labels(self):
        return self.labels

    def tokenize(self):
        """
        Articles will be processed in parallel
        """
        articles_iter = chunk(self.articles, size=self.chunks)
        length = int(len(self.articles) / self.chunks)
        nlp_iter = repeat(English())

        tokenized_comments = []
        with ProcessPoolExecutor() as executor:
            chunksize = int(max(length / (self.processes * self.parallelism), 1))
            i = 0
            for result in executor.map(_tokenize_article, articles_iter,
                                        nlp_iter, chunksize=chunksize):
                for article in result:
                    tokenized_comments.append(article)
                    i += 1
                    if i % 10000 == 0:
                        print('Processed {} comments'.format(i))
        return tokenized_comments


def _tokenize_article(articles, nlp):
    # https://stackoverflow.com/questions/36123586/python-multiprocessing-cant-pickle-type-function

    output = []
    for doc in nlp.pipe(article_iter(articles)):
        upper = np.where(doc.to_array([IS_UPPER]))[0]
        token_list = [str(token).lower() for token in doc]
        acc = 0
        for i in upper:
            token_list.insert(i + acc, 't_up')
            acc += 1
        output.append(token_list)

    return output


def article_iter(articles):
    for article in articles:
        yield BOS_TOKEN + ' ' + article