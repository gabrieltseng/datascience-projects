"""
The model described in part 1
"""
import torch
from torch import nn


class Px1Model(nn.Module):
    """
    As in the warmup, a model for p(x1)
    """

    def __init__(self, num_dims: int = 200):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(1, num_dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.theta


class Px2Model(nn.Module):
    """
    Given p(x1), produce a distribution over x2
    """

    def __init__(self, x1_dims: int = 1, x2_dims: int = 200, num_layers: int = 1):
        super().__init__()

        self.net = nn.Sequential(
            *[
                nn.Linear(x1_dims if i == 0 else x2_dims, x2_dims, bias=False)
                for i in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Model1(nn.Module):
    def __init__(
        self,
        x1_dims: int = 200,
        x2_dims: int = 200,
        num_layers: int = 1,
        one_hot_encode: bool = True,
    ):
        super().__init__()

        self.x1_dims = x1_dims
        self.one_hot_encode = one_hot_encode
        self.probx1 = Px1Model(x1_dims)
        self.probx2 = Px2Model(x1_dims if one_hot_encode else 1, x2_dims, num_layers)

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        return torch.eye(self.x1_dims)[x.long()].squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        input_tensor = x[:, 0].unsqueeze(-1).float()
        x1_dist = self.probx1(input_tensor)

        if self.one_hot_encode:
            input_tensor = self.encoder(input_tensor)
        x2_dist = self.probx2(input_tensor)

        return torch.stack((x1_dist, x2_dist), dim=-1)
