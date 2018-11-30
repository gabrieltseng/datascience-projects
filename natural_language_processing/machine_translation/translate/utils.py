from itertools import islice
import numpy as np
import torch


def read_sentence(word2int, ints):
    if type(ints) == torch.Tensor:
        if len(ints.shape) == 2:
            # turn a prediction into a sequence of ints
            ints = torch.argmax(ints, dim=-1)
        if ints.is_cuda:
            ints = ints.cpu()
        if ints.requires_grad:
            ints = ints.detach()
        ints = ints.numpy()

    int2word = {int(idx): word for word, idx in word2int.items()}

    return " ".join([int2word[i] for i in ints])


def chunk(it, size):
    """
    An iterator which returns chunks of items (i.e. size items per call, instead of 1).
    Setting size=1 returns a normal iterator.
    """
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def to_scalar(list):
    """
    PyTorch 0.4.0 has a hard time with numpy.int64
    """

    return [np.asscalar(x) for x in list]
