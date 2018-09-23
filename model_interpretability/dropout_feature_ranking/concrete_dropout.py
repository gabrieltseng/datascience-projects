import torch
from torch import nn
import torch.nn.functional as F


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

    def __init__(self, input_shape, init_values=0.5, t=0.1):
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
        mask = F.sigmoid((1/self.t) * (torch.log(self.parameter_mask) - torch.log(1 - self.parameter_mask) +
                                       torch.log(torch.empty(self.input_shape).uniform_()) -
                                       torch.log(1 - torch.empty(self.input_shape).uniform_(to=1))))
        if self.training:
            return mask * input, mask
        else:
            return mask * input
