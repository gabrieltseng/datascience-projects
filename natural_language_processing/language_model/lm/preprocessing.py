import multiprocessing
import functools
import numpy as np
from random import randint
import re

from spacy.attrs import IS_UPPER
from spacy.lang.en import English


BOS_TOKEN = 'xbos'
FIELD_TOKEN = 'xfld'


class WikiTextTokenizer(object):
    """
    Loads, tokenizes and then returns batches of the
    wikitext dataset
    """
    def __init__(self, filepaths, processes=6, parallelism=4):

        self.filepaths = filepaths
        self.articles = self.read_articles()
        self.processes = processes
        self.parallelism = parallelism

    def get_one_article(self):
        idx = randint(0, len(self.articles) - 1)
        return self.articles[idx]

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

    def preprocess(self, word2int=None, vocab_size=30000, min_frequency=2):
        wikitext = self.tokenize()
        print('Tokenized articles!')
        return_word2int = False
        if word2int is None:
            return_word2int = True
            assert vocab_size is not None, "Vocab size must be defined"
            assert min_frequency is not None, "Minimum word frequency must be defined"

            # now, to find the frequency of words
            unique_tokens, count = np.unique(wikitext, return_counts=True)

            # get the ordered indices
            sort_indices = count.argsort()[::-1]
            sorted_tokens = unique_tokens[sort_indices]
            sorted_counts = count[sort_indices]

            # make sure all my selected vocab has at least min_frequency words
            assert sorted_counts[vocab_size] >= min_frequency

            # we will now add the unknown and padding tokens
            sorted_tokens = np.insert(sorted_tokens, 0, '_unk_')
            sorted_tokens = np.insert(sorted_tokens, 0, '_pad_')

            # word2int
            word2int = {tok: idx for idx, tok in enumerate(sorted_tokens[:(vocab_size + 2)])}

        unknown_int = word2int['_unk_']
        # now, we can turn tw_ar into an array of ints
        tokenized_ints = [word2int.get(tok, unknown_int) for tok in wikitext]

        if return_word2int: return tokenized_ints, word2int
        else: return tokenized_ints

    def tokenize(self):
        """
        All articles will be processed in parallel
        """
        articles_iter = iter(self.articles)

        tokenized_wikitext = []
        with multiprocessing.Pool(processes=self.processes) as pool:
            f = functools.partial(_tokenize_wiki_article)
            chunksize = int(max(len(self.articles) / (self.processes * self.parallelism), 1))
            results = pool.imap(f, articles_iter, chunksize=chunksize)
            i = 0
            for article in results:
                tokenized_wikitext.extend(article)
                i += 1
                if i % 100 == 0:
                    print('Processed {} articles'.format(i))

        return np.asarray(tokenized_wikitext)


def _tokenize_wiki_article(article):
    # https://stackoverflow.com/questions/36123586/python-multiprocessing-cant-pickle-type-function

    # make a tokenizer
    nlp = English()
    tokenizer = English().Defaults.create_tokenizer(nlp)

    # first, add beginning of string and field tokens
    article = re.sub(r"(\n){1}(?=((\s{1}={1})+)[^=]+((\s{1}={1})+))", FIELD_TOKEN, article)
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

