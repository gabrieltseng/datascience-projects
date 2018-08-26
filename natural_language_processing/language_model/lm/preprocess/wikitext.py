import numpy as np
import re

from itertools import repeat
from concurrent.futures import ProcessPoolExecutor

from spacy.attrs import IS_UPPER
from spacy.lang.en import English

from .base import Tokenizer, BOS_TOKEN, FIELD_TOKEN


class WikiTextTokenizer(Tokenizer):
    """
    Loads, and tokenizes the wikitext dataset
    """

    def read_articles(self):
        """
        Returns a list of articles
        """
        all_data = []
        for filepath in self.filepaths:
            with filepath.open() as f:
                # different articles are split by "\n = {title} = \n"
                data = re.split(r"(\n){1}(?=(\s{1}={1}\s{1})[^=]+(\s{1}={1}\s{1}\n))",
                                f.read())[1:][3::4]
                all_data.extend(data)
                print('Loaded {} articles'.format(len(data)))
        return all_data

    def tokenize(self):
        """
        Articles will be processed in parallel
        """
        articles_iter = self.chunk(self.articles, size=self.chunks)
        length = int(len(self.articles) / self.chunks)
        nlp_iter = repeat(English())

        tokenized_wikitext = []
        with ProcessPoolExecutor() as executor:
            chunksize = int(max(length / (self.processes * self.parallelism), 1))
            i = 0
            for result in executor.map(_tokenize_wiki_article, articles_iter,
                                        nlp_iter, chunksize=chunksize):
                for article in result:
                    tokenized_wikitext.extend(article)
                    i += 1
                    if i % 100 == 0:
                        print('Processed {} articles'.format(i))
        return tokenized_wikitext


def _tokenize_wiki_article(articles, nlp):
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
        article = re.sub(r"(\n){1}(?=((\s{1}={1})+)[^=]+((\s{1}={1})+))", FIELD_TOKEN, article)
        yield BOS_TOKEN + ' ' + article
