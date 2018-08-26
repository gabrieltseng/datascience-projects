import pandas as pd
import numpy as np
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor

from spacy.attrs import IS_UPPER
from spacy.lang.en import English

from .base import Tokenizer, BOS_TOKEN


class ToxicTokenizer(Tokenizer):
    """
    Loads and tokenizes the toxic tokenizer dataset
    """
    dimension = 2

    def __init__(self, filepaths, label_filepaths=None, processes=6, parallelism=4,
                 chunks=1000):
        self.label_filepaths = label_filepaths
        super().__init__(filepaths, processes, parallelism, chunks)

    def read_articles(self):
        df = pd.DataFrame()
        for file in self.filepaths:
            df = pd.concat([df, pd.read_csv(file)])

        if self.label_filepaths:
            labels_df = pd.DataFrame()
            for label_file in self.label_filepaths:
                labels_df = pd.concat([labels_df, pd.read_csv(label_file)])

            df = df.join(labels_df.set_index('id'), on='id')

        # make sure we have the labels
        assert 'toxic' in df.columns, 'Labels not loaded; try setting label_filepaths'

        print('Loaded {} articles'.format(len(df.comment_text.values)))

        label_cols = [col for col in df.columns if col not in ('id', 'comment_text')]
        self.labels = df[label_cols].values
        self.header2index = {idx: col for idx, col in enumerate(label_cols)}
        self.ids = df.id.values

        return df.comment_text.values

    def get_articles(self):
        return self.a

    def get_ids(self):
        return self.ids

    def get_labels(self):
        return self.labels, self.header2index

    def tokenize(self):
        """
        Articles will be processed in parallel
        """
        articles_iter = self.chunk(self.articles, size=self.chunks)
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