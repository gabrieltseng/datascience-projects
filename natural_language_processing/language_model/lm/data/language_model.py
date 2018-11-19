import numpy as np


class LMDataLoader(object):
    """
    Variable length sequences, from https://arxiv.org/abs/1708.02182.

    Note that when using this, the learning rate should be rescaled depending
    on the length of the resulting sequence, to avoid favouring short sequences
    over larger ones.

    Arguments:
        words: a 1D int tensor representing the stream of documents to be trained on.
            Not called a dataset to explicitly differentiate it from a PyTorch Dataset
            object, which it is not.
        base_sequence_length: the base sequence length, before randomization (i.e. on
            average, how long should the sequences be?)
        batch_size: how many samples per batch to load
        half_proba: the probability with which sequence lengths are halved. Should be close
            to 0 (default: 0.05)
    """

    def __init__(self, words, base_sequence_length, batch_size, half_proba=0.05):

        # to allow us to fit the words neatly into columns
        mod = len(words) % batch_size
        self.dataset = words[:-mod].view(batch_size, -1)

        self.bsl = base_sequence_length
        self.batch_size = batch_size
        self.half_proba = half_proba

    def __iter__(self):
        return _LM_iter(self)

    def __len__(self):
        # this is approximate, but is still useful
        return int(self.dataset.shape[1] / self.batch_size)


class _LM_iter(object):

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.bsl = loader.bsl
        self.batch_size = loader.batch_size
        self.half_proba = loader.half_proba

        self.idx = 0
        self.max_idx = loader.dataset.shape[1] - 1

    def __iter__(self):
        return self

    def __next__(self):

        if self.idx < (self.max_idx - 2):
            # lets find the base sequence length
            factor = np.random.choice([0.5, 1], size=1, p=[0.05, 0.95])[0]
            iter_bsl = self.bsl * factor

            # next, the true sequence length from a normal distribution
            sequence_length = int(np.random.normal(loc=iter_bsl, scale=5))

            x_max_idx = min(self.max_idx - 1, self.idx + sequence_length)
            y_max_idx = min(self.max_idx, self.idx + sequence_length + 1)

            x = self.dataset[:, self.idx: x_max_idx]
            y = self.dataset[:, self.idx + 1: y_max_idx]

            self.idx += sequence_length + 2
            return x, y
        else:
            raise StopIteration()
