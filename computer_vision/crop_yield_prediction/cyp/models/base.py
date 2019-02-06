class ModelBase:
    """
    Base class for all models
    """

    def train(self, path_to_histogram):
        raise NotImplementedError

    def save(self, filedir):
        raise NotImplementedError

    def load(self, filedir):
        raise NotImplementedError
