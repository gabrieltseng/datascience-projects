import torch
from torch import nn


class VariationalDropout(nn.Module):
    """
    A binary dropout mask is sampled only
    once upon the first call, and then
    repeatedly used for each item in the minibatch
    """
    def __init__(self, p):
        super().__init__()

        self.p = p

    def forward(self, x):
        if not self.training:
            return x

        # first, make the mask, remembering to scale
        mask = torch.bernoulli(torch.ones(x.shape[1:]) * (1 - self.p)) / (1 - self.p)
        # in case we're on a GPU
        if x.is_cuda: mask = mask.cuda()

        return mask.unsqueeze(0) * x
