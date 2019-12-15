from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

from typing import List, Optional, Tuple


def plot_loss(
    train_loss: List[float],
    val_loss: List[float],
    epoch_length: int,
    savepath: Optional[Path] = None,
):
    plt.clf()
    fig, ax = plt.subplots()

    ax.scatter(range(len(train_loss)), train_loss, label="Train loss")
    ax.scatter(
        [idx * epoch_length for idx, _ in enumerate(val_loss)],
        val_loss,
        label="Val loss",
    )
    plt.legend()

    plt.xlabel("Training step")
    plt.ylabel("Loss (bits per dimension)")

    plt.savefig(
        "diagrams/losses.png" if savepath is None else savepath,
        dpi=300,
        bbox_inches="tight",
    )


def plot_2d_hist(x: np.ndarray, y: np.ndarray, savepath: Path):

    plt.clf()
    fix, ax = plt.subplots()

    ax.hist2d(x, y)

    plt.savefig(savepath, dpi=300, bbox_inches="tight")


def sample_model(model: nn.Module, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    x1_samples: List[float] = []
    x2_samples: List[float] = []
    for _ in range(num_samples):
        input = torch.tensor(np.random.randint(0, 100, size=(1, 2)))
        output = model(input)
        x1_samples.append(Categorical(F.softmax(output[:, :, 0])).sample())
        x2_samples.append(Categorical(F.softmax(output[:, :, 1])).sample())

    return np.asarray(x1_samples), np.asarray(x2_samples)
