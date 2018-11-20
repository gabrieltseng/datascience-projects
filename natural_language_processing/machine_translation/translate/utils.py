from itertools import islice
import numpy as np


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
