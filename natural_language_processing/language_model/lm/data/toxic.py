from sklearn.utils import shuffle
from itertools import islice

import torch
from torch.nn.utils.rnn import pad_sequence


class ToxicDataLoader(object):

    def __init__(self, comments, labels, pad_idx, batch_size):
        self.comments = comments
        self.labels = labels
        self.pad_idx = pad_idx
        self.batch_size = batch_size

    def __iter__(self):
        return _sortish_iter(self)

    def __len__(self):
        return int(len(self.comments) / self.batch_size)


class _sortish_iter(object):
    """
    From torchtext:
    `Partitions data into chunks of size 100*batch_size, sorts examples within
    each chunk using sort_key, then batch these examples and shuffle the
    batches.`
    """

    def __init__(self, loader):
        self.comments = loader.comments
        self.labels = loader.labels
        self.pad_idx = loader.pad_idx
        self.batch_size = loader.batch_size
        self.idx = 0
        self.max_idx = len(loader.comments) - 1

        self._order_ish()

    def _order_ish(self):
        # first, shuffle everything
        comments, labels = shuffle(self.comments, self.labels)
        data = zip(comments, labels)
        ishsorted_comments = []
        ishsorted_labels = []
        # then, chunk them
        for megabatch in self.chunk(data, self.batch_size * 100):
            # sort within the batch
            com, lab = map(list, zip(*sorted(megabatch, key=lambda x: len(x[0]),
                                             reverse=True)))
            ishsorted_comments.extend([torch.Tensor(x) for x in com])
            ishsorted_labels.extend(lab)
        self.sorted_comments = ishsorted_comments
        self.sorted_labels = torch.Tensor(ishsorted_labels)

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

    @staticmethod
    def chunk(it, size):
        """
        An iterator which returns chunks of items (i.e. size items per call, instead of 1).
        Setting size=1 returns a normal iterator.
        """
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())
