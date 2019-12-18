"""
The model described in part 2 (MADE)
"""
import torch
from torch import nn
import torch.nn.functional as F

from typing import List, Tuple


class Masked2DLinear(nn.Module):
    """
    This only works in the 2d case, where the order of the dimensions
    is x_1, x_2 (i.e. this specific use case)
    """

    def __init__(self, in_features: int, out_features: int,
                 x1_dims: int, x2_dims: int, mask_in: bool) -> None:
        super().__init__()

        self.w = nn.Parameter(torch.zeros(out_features, in_features).float(), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_features).float(), requires_grad=True)
        self.mask = self.set_mask(in_features, out_features, x1_dims, x2_dims, mask_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.mask * self.w, self.bias)

    def set_mask(self, in_features: int, out_features: int, x1_dims: int, x2_dims: int, mask_in: bool) -> torch.Tensor:

        # if mask_in, then I want to mask the input (i.e. x_2_in)
        # otherwise, I want to mask the output (i.e. x_1_out)

        # x1_dims, x2_dims are expected to correspond to the input / output depending on mask_in

        mask = torch.ones(out_features, in_features).float()

        if mask_in:
            mask[:, x1_dims:] = 0
        else:
            mask[:x1_dims, :] = 0

        return mask


class Model2(nn.Module):
    name = "model_2"  # aka MADE

    def __init__(self, x1_dim: int = 200, x2_dim: int = 200,
                 hidden_sizes: Tuple[int, ...] = (200, 200)) -> None:
        super().__init__()

        layer_list: List[nn.Module] = [Masked2DLinear(in_features=x1_dim + x2_dim, out_features=hidden_sizes[0],
                                                      x1_dims=x1_dim, x2_dims=x2_dim, mask_in=True)]

        for idx, hidden_size in enumerate(hidden_sizes[1:]):
            layer_list.append(nn.Linear(in_features=hidden_sizes[idx],
                                        out_features=hidden_size))
            layer_list.append(nn.ReLU())

        layer_list.append(Masked2DLinear(in_features=hidden_sizes[-1], out_features=x1_dim + x2_dim,
                                         x1_dims=x1_dim, x2_dims=x2_dim, mask_in=False))

        self.net = nn.Sequential(*layer_list)
        self.x1_dim = x1_dim
        self.x2_dim = x2_dim

    def encoder(self, x: torch.Tensor, num_dims: int) -> torch.Tensor:
        return torch.eye(num_dims)[x.long()].squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x1, x2 = self.encoder(x[:, 0], self.x1_dim), self.encoder(x[:, 1], self.x2_dim)

        x = torch.cat((x1, x2), dim=-1)

        out = self.net(x.float())

        x1_dist = out[:, :self.x1_dim]
        x2_dist = out[:, self.x1_dim:]

        return torch.stack((x1_dist, x2_dist), dim=-1)
