from pathlib import Path
import matplotlib.pyplot as plt

from typing import List, Optional


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
