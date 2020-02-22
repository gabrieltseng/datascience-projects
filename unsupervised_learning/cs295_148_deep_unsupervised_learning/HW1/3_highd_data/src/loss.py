import torch
from torch import nn


def quantized_mnist_loss(pred, true) -> torch.Tensor:

    assert (
        pred.size(1) == 12
    ), f"Pred channels must be 12 (3 channels each with 2 bits). Got {pred.size(1)}"

    loss_function = nn.CrossEntropyLoss()

    red_loss = loss_function(pred[:, :4, :, :], true[:, 0, :, :].long())
    green_loss = loss_function(pred[:, 4:8, :, :], true[:, 1, :, :].long())
    blue_loss = loss_function(pred[:, 8:12, :, :], true[:, 2, :, :].long())

    return red_loss + green_loss + blue_loss
