from itertools import islice


def chunk(it, size):
    """
    An iterator which returns chunks of items (i.e. size items per call, instead of 1).
    Setting size=1 returns a normal iterator.
    """
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())
