import numpy as np
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from src.data import load_distribution, generate_samples, train_test_split
from src.plots import plot_loss, plot_2d_hist, sample_model

from typing import List


def train_model(
    model: nn.Module,
    num_samples: int = 10000,
    test_size: float = 0.2,
    val_size: float = 0.2,
    num_epochs: int = 1000,
    early_stopping: int = 10,
):

    data = generate_samples(load_distribution(), num_samples)

    plot_2d_hist(
        data[:, 0], data[:, 1], savepath=Path("diagrams/original_data_distribution.png")
    )

    train, test = train_test_split(data, test_size)
    train, val = train_test_split(train, (1 - test_size) * val_size)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train)), batch_size=128, shuffle=True,
    )
    val_tensor, test_tensor = torch.from_numpy(val), torch.from_numpy(test)

    optimizer = torch.optim.Adam(model.parameters())
    loss_function = nn.CrossEntropyLoss()

    train_loss: List[float] = []
    epoch_val_loss: List[float] = []

    best_val_loss = np.inf
    patience_counter = 0
    best_model_dict = None

    for epoch in range(num_epochs):
        for (x_batch,) in train_loader:
            optimizer.zero_grad()

            pred_dist = model(x_batch)
            loss = loss_function(pred_dist, x_batch).mean()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item() / 2)

        with torch.no_grad():
            val_loss = loss_function(model(val_tensor), val_tensor).mean()

            epoch_val_loss.append(val_loss.item() / 2)

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                best_model_dict = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    print("Early stopping!")
                    if best_model_dict is not None:
                        model.load_state_dict(best_model_dict)
                    break
        print(
            f"Epoch {epoch + 1}: Train loss (bits / dim): {np.mean(train_loss)}, Val loss: {val_loss.item() / 2}"
        )

    plot_loss(
        train_loss,
        epoch_val_loss,
        len(train_loader),
        savepath=Path(f"diagrams/{model.name}_loss_curve.png"),
    )

    print(
        f"Test loss: {loss_function(model(test_tensor), test_tensor).mean() / 2} bits / dim"
    )

    x1_samples, x2_samples = sample_model(model, 1000)
    plot_2d_hist(
        x1_samples,
        x2_samples,
        savepath=Path(f"diagrams/{model.name}_sampled_data_distribution.png"),
    )
