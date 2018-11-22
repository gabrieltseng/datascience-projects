from sklearn.utils import shuffle
import numpy as np

import torch
from ..data import pad_sequence

from ..utils import chunk


class QuestionDataLoader(object):
    """
    DataLoader for the matched question dataset

    Args:
        reviews: numpy array or list of tokenized questions
        batch_size: how many samples per batch to load
        pad_idx: the index of padded values
        sortish: if True, uses the sort-ish iterator. This allows some sorting,
            as well as shuffling. If False, just returns the comments sorted by
            length.
        {en, fr}_max_seq_length: Truncate sequences to be this length
        batch_factor: the chunk size for the sort ish iterator. Not used if a normal
            sorter is used.
    """
    def __init__(self, english, french, batch_size, en_pad_idx, fr_pad_idx,
                 sortish=True, device=torch.device("cpu"),
                 fr_max_seq_length=None, en_max_seq_length=None, batch_factor=100):
        self.english = english
        self.french = french
        self.en_pad_idx = en_pad_idx
        self.fr_pad_idx = fr_pad_idx
        self.batch_size = batch_size
        self.sortish = sortish
        self.device = device
        self.en_max_seq_length = en_max_seq_length
        self.fr_max_seq_length = fr_max_seq_length
        self.batch_factor = batch_factor

    def __iter__(self):
        if self.sortish:
            return _SortishIter(self)
        else:
            return _SortIter(self)

    def __len__(self):
        return int(len(self.english) / self.batch_size)


class SorterBase(object):

    def __init__(self, loader):
        self.english = loader.english
        self.french = loader.french
        self.en_pad_idx = loader.en_pad_idx
        self.fr_pad_idx = loader.fr_pad_idx
        self.batch_size = loader.batch_size
        self.device = loader.device
        self.idx = 0
        self.max_idx = len(loader.english) - 1
        self.en_max_seq_length = loader.en_max_seq_length
        self.fr_max_seq_length = loader.fr_max_seq_length
        self.batch_factor = loader.batch_factor

        sorted_en, sorted_fr = self._order()
        self.english = sorted_en
        self.french = sorted_fr

    def _order(self):
        """
        Should return sorted comments and labels
        """
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx < self.max_idx:
            max_idx = self.idx + self.batch_size

            en = self.english[self.idx: max_idx]
            fr = self.french[self.idx: max_idx]
            if self.en_max_seq_length:
                en = [l[:self.en_max_seq_length] for l in en]
            if self.fr_max_seq_length:
                fr = [l[:self.fr_max_seq_length] for l in fr]

            # pad
            en = pad_sequence(en, batch_first=True, padding_value=self.en_pad_idx)

            # so that the final state will reflect the final token in the input sentence
            fr = pad_sequence(fr, batch_first=True, padding_value=self.fr_pad_idx,
                              padding_first=True)
            self.idx = max_idx
            return en, fr
        else:
            raise StopIteration()


class _SortishIter(SorterBase):
    """
    From torchtext:
    `Partitions data into chunks of size batch_factor*batch_size, sorts examples within
    each chunk using sort_key, then batch these examples and shuffle the
    batches.`

    Note that the sorting will happen with respect to the length of the french questions, since
    these will be the input to the model
    """
    def _order(self):
        # first, shuffle everything
        english, french = shuffle(self.english, self.french)
        data = zip(english, french)
        ishsorted_english = []
        ishsorted_french = []
        # then, chunk them
        for megabatch in chunk(data, self.batch_size * self.batch_factor):
            # sort within the batch
            en, fr = zip(*sorted(megabatch, key=lambda x: len(x[1]),
                                 reverse=True))
            ishsorted_english.extend([torch.tensor(x, device=self.device) for x in en])
            ishsorted_french.extend([torch.tensor(x, device=self.device) for x in fr])
        return ishsorted_english, ishsorted_french


class _SortIter(SorterBase):
    """
    Just sorts the comments and labels. Useful for the validation set.
    """
    def _order(self):
        data = zip(self.english, self.french)
        en, fr = map(list, zip(*sorted(data, key=lambda x: len(x[1]),
                     reverse=True)))
        en = [torch.tensor(x, device=self.device) for x in en]
        fr = [torch.tensor(x, device=self.device) for x in fr]

        return en, fr
