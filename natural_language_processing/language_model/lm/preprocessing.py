from pathlib import Path
import multiprocessing
import functools
import numpy as np

from spacy.attrs import IS_UPPER
from spacy.lang.en import English


BOS_TOKEN = 'xbos'
FIELD_TOKEN = 'xfld'


class WikiTextTokenizer(object):
    """
    Loads, tokenizes and then returns batches of the
    wikitext103 dataset
    """
    def __init__(self, filepaths,
                 processes=6, parallelism=4, chunksize=1000):

        self.filepaths = filepaths
        self.articles = self.read_articles()
        self.processes = processes
        self.parallelism = parallelism

        self.num_chunks = max(int(len(self.articles) / chunksize), 1)

    def get_one_article(self):
        return self.articles[0]

    def read_articles(self):
        """
        Returns a list of articles
        """
        all_data = []
        for filepath in self.filepaths:
            with filepath.open() as f:
                # different articles are split by 3 newlines
                data = f.read().split(' \n \n \n')
                all_data.extend(data)
                print('Loaded {} articles'.format(len(data)))
        return data

    def tokenize(self):
        """
        To reduce overhead costs, split the list into chunks. All articles
        within a chunk will be processed in parallel
        """
        tokenized_wikitext = []
        print('Splitting articles into {} chunks'.format(self.num_chunks))
        for list_num, sublist in enumerate(np.array_split(self.articles, self.num_chunks)):
            tokenized_chunk = self.tokenize_chunk(sublist)
            tokenized_wikitext.extend(tokenized_chunk)
            print('Done {}/{} chunks'.format(list_num + 1, self.num_chunks))

        return tokenized_wikitext

    def tokenize_chunk(self, articles_chunk):
        """
        Returns a list of lists of tokens
        """

        # turn the articles into a generator
        articles_iter = iter(articles_chunk)

        tokenized_wikitext = []
        with multiprocessing.Pool(processes=self.processes) as pool:
            f = functools.partial(_tokenize_wiki_article)
            chunksize = int(max(len(articles_chunk) / (self.processes * self.parallelism), 1))
            results = pool.imap(f, articles_iter, chunksize=chunksize)
            for article in results:
                tokenized_wikitext.extend(article)
        return tokenized_wikitext


def _tokenize_wiki_article(article):
    # https://stackoverflow.com/questions/36123586/python-multiprocessing-cant-pickle-type-function

    # make a tokenizer
    nlp = English()
    tokenizer = English().Defaults.create_tokenizer(nlp)

    # first, add beginning of string and field tokens
    article = article.replace(' \n \n ', ' ' + FIELD_TOKEN + ' ')
    article = BOS_TOKEN + ' ' + article

    doc = tokenizer(article.strip())

    # Next, add the t_up token
    upper = np.where(doc.to_array([IS_UPPER]))[0]
    token_list = [str(token).lower() for token in doc]

    acc = 0
    for i in upper:
        token_list.insert(i + acc, 't_up')
        acc += 1

    return token_list

