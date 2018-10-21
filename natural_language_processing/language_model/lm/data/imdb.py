from sklearn.utils import shuffle
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence

from ..utils import chunk


class IMDBDataLoader(object):
    """
    DataLoader for IMDB reviews

    Args:
        reviews: numpy array or list of tokenized review intergers
        labels: numpy array or list of labels
        batch_size: how many samples per batch to load
        pad_idx: the index of padded values
        sortish: if True, uses the sort-ish iterator. This allows some sorting,
            as well as shuffling. If False, just returns the comments sorted by
            length.
    """
    def __init__(self, reviews, labels, batch_size, pad_idx, sortish=True,
                 device=torch.device("cpu")):
        self.reviews = reviews
        self.labels = labels
        self.pad_idx = pad_idx
        self.batch_size = batch_size
        self.sortish = sortish
        self.device = device

    def __iter__(self):
        if self.sortish:
            return _SortishIter(self)
        else:
            return _SortIter(self)

    def __len__(self):
        return int(len(self.reviews) / self.batch_size)


class SorterBase(object):

    def __init__(self, loader):
        self.reviews = loader.reviews
        self.labels = loader.labels
        self.pad_idx = loader.pad_idx
        self.batch_size = loader.batch_size
        self.device = loader.device
        self.idx = 0
        self.max_idx = len(loader.reviews) - 1

        sorted_comments, sorted_labels = self._order()
        self.sorted_comments = sorted_comments
        self.sorted_labels = sorted_labels

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

            x = self.sorted_comments[self.idx: max_idx]
            y = self.sorted_labels[self.idx: max_idx]

            # pad
            x = pad_sequence(x, batch_first=True, padding_value=self.pad_idx)
            self.idx = max_idx
            return x, y
        else:
            raise StopIteration()


class _SortishIter(SorterBase):
    """
    From torchtext:
    `Partitions data into chunks of size 100*batch_size, sorts examples within
    each chunk using sort_key, then batch these examples and shuffle the
    batches.`
    """
    def _order(self):
        # first, shuffle everything
        reviews, labels = shuffle(self.reviews, self.labels)
        data = zip(reviews, labels)
        ishsorted_comments = []
        ishsorted_labels = []
        # then, chunk them
        for megabatch in chunk(data, self.batch_size * 100):
            # sort within the batch
            com, lab = zip(*sorted(megabatch, key=lambda x: len(x[0]),
                                   reverse=True))
            ishsorted_comments.extend([torch.tensor(x, device=self.device) for x in com])
            ishsorted_labels.extend(lab)
        return ishsorted_comments, torch.tensor(np.array(ishsorted_labels), device=self.device)


class _SortIter(SorterBase):
    """
    Just sorts the comments and labels. Useful for the validation set.
    """
    def _order(self):
        data = zip(self.reviews, self.labels)
        com, lab = map(list, zip(*sorted(data, key=lambda x: len(x[0]),
                       reverse=True)))
        com = [torch.tensor(x, device=self.device) for x in com]
        return com, torch.tensor(lab, device=self.device)
