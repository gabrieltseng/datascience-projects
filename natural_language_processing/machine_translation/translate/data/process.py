from random import randint
import numpy as np

from itertools import repeat
from concurrent.futures import ProcessPoolExecutor

from spacy.attrs import IS_UPPER
from spacy.lang.en import English
from spacy.lang.fr import French

from translate.utils import chunk

from ..data import BOS_TOKEN, EOS_TOKEN


class QuestionTokenizer(object):
    """
    Base class for tokenizers. Mostly taken from
    natural_language_processing/language_model/lm/preprocess/base.py
    """

    def __init__(self, filepaths, processes=6, parallelism=4, chunks=1):

        english, french = filepaths
        self.read_questions(english, french)
        self.processes = processes
        self.parallelism = parallelism
        self.chunks = chunks

    def get_one_question(self):
        idx = randint(0, len(self.articles) - 1)
        return self.french[idx], self.english[idx]

    def read_questions(self, english, french):
        """
        Reads the files, and then filters out so that only who, what, when, where, why
        questions remain
        """
        french_qs, english_qs = [], []
        for en, fr in zip(english.open(), french.open()):
            # remove the newline character
            en, fr = en[:-1], fr[:-1]
            if en.startswith('Wh') and en.endswith('?') and fr.endswith('?'):
                # some questions seem to be from forms; this removes them
                # This is particularly important, because sentences with lots
                # of punctuation will have an outsized effect on the model,
                # since they will be comparatively longer
                form_substrings = ['_____', '.....', '. . . . .', '_ _ _ _ _']
                if all(form not in en for form in form_substrings):
                    english_qs.append(en)
                    french_qs.append(fr)

        print(f'Loaded {len(english_qs)} questions')

        self.english = english_qs
        self.french = french_qs

    def tokenize(self, dataset, language):
        """
        Articles will be processed in parallel
        """
        articles_iter = chunk(dataset, size=self.chunks)
        length = int(len(dataset) / self.chunks)
        if language == 'english':
            nlp_iter = repeat(English())
        else:
            nlp_iter = repeat(French())

        tokenized_questions = []
        with ProcessPoolExecutor() as executor:
            chunksize = int(max(length / (self.processes * self.parallelism), 1))
            i = 0
            for result in executor.map(_tokenize_questions, articles_iter,
                                        nlp_iter, chunksize=chunksize):
                for article in result:
                    tokenized_questions.append(article)
                    i += 1
                    if i % 10000 == 0:
                        print('Processed {} articles'.format(i))
        return tokenized_questions

    def _preprocess_single(self, tokenized_texts, vocab_size):

        # now, to find the frequency of words
        flat_texts = [tok for sublist in tokenized_texts for tok in sublist]
        unique_tokens, count = np.unique(flat_texts, return_counts=True)

        # get the ordered indices
        sort_indices = count.argsort()[::-1]
        sorted_tokens = unique_tokens[sort_indices]

        # we will now add the beginning of sentence, unknown and padding tokens
        sorted_tokens = np.insert(sorted_tokens, 0, '_unk_')
        sorted_tokens = np.insert(sorted_tokens, 0, '_pad_')
        sorted_tokens = np.insert(sorted_tokens, 0, BOS_TOKEN)

        # word2int
        word2int = {tok: idx for idx, tok in enumerate(sorted_tokens[:(vocab_size + 2)])}

        unknown_int = word2int['_unk_']
        # now, we can turn tw_ar into an array of ints
        tokenized_ints = [[word2int.get(tok, unknown_int) for tok in subtext] for subtext
                          in tokenized_texts]
        return tokenized_ints, word2int

    def preprocess(self, vocab_size=40000):
        print('Tokenizing english questions')
        tokenized_english = self.tokenize(self.english, 'english')
        print('Tokenizing french questions')
        tokenized_french = self.tokenize(self.french, 'french')
        print('Tokenized questions!')

        assert vocab_size is not None, "Vocab size must be defined"

        eng_ints, eng_word2int = self._preprocess_single(tokenized_english, vocab_size)
        fr_ints, fr_word2int = self._preprocess_single(tokenized_french, vocab_size)
        return (eng_ints, eng_word2int), (fr_ints, fr_word2int)


def _tokenize_questions(questions, nlp):
    # https://stackoverflow.com/questions/36123586/python-multiprocessing-cant-pickle-type-function

    output = []
    for doc in nlp.pipe(questions):
        upper = np.where(doc.to_array([IS_UPPER]))[0]
        token_list = [str(token).lower() for token in doc]
        acc = 0
        for i in upper:
            token_list.insert(i + acc, 't_up')
            acc += 1
        token_list += [EOS_TOKEN]
        output.append(token_list)

    return output
