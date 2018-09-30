import torch
from torch import nn


class ConcreteDropout(nn.Module):
    """
    Concrete dropout. A variant of variational dropout.
    Arguments:
        input_shape: tuple. The shape of the tensor which will be passed to the
            layer (i.e. the shape the mask should take)
        init_values: float, or tensor of shape input_shape. The initial dropout
            parameters
        t: float, weight (to be applied before the sigmoid)
    """

    def __init__(self, input_shape, init_values=0.3, t=0.1):
        super().__init__()

        self.input_shape = input_shape

        if type(init_values) is float:
            init_values = torch.ones(input_shape) * init_values
        else:
            message = 'Mask tensor must have the same shape as the input tensor!'
            assert init_values.shape == input_shape, message

        self.parameter_mask = nn.Parameter(init_values)
        self.t = t

    def forward(self, input):
        # sigmoid prevents nan when we take a log of the mask
        p = torch.sigmoid(self.parameter_mask)
        uniform_distribution = torch.empty(self.input_shape).uniform_()

        # +1e-7 to prevent nans if p or uniform_distribution = 0
        mask = torch.sigmoid((1/self.t) * (torch.log(p + 1e-7) - torch.log(1 - p + 1e-7) +
                                           torch.log(uniform_distribution + 1e-7) -
                                           torch.log(1 - uniform_distribution + 1e-7)))
        if self.training:
            return mask * input, mask
        else:
            return mask * input


class ConcreteRegularizer(nn.Module):
    """A regularizer for concrete dropout
    """
    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def forward(self, mask):
        """Apply the regularizing weights to concrete dropout
        """
        return self.lam * torch.sum(mask, tuple(range(mask.ndimension())))
