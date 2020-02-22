import numpy as np
from pathlib import Path

from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader

from .utils import load_data
from .pixelcnn import PixelCNN
from .loss import quantized_mnist_loss
from .plots import plot_loss

from typing import List


def train_model(
    val_size: float = 0.1,
    num_epochs: int = 1000,
    early_stopping: int = 10,
):

    train, val, test = load_data(Path("data/mnist-hw1.pkl"), val_size)

    train_loader = DataLoader(
        TensorDataset(train), batch_size=128, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val), batch_size=128, shuffle=False
    )

    test_loader = DataLoader(
        TensorDataset(test), batch_size=128, shuffle=False
    )

    model = PixelCNN()

    optimizer = torch.optim.Adam(model.parameters())

    train_loss: List[float] = []
    epoch_val_loss: List[float] = []

    best_val_loss = np.inf
    patience_counter = 0
    best_model_dict = None

    for epoch in range(num_epochs):
        for (x_batch,) in tqdm(train_loader):
            optimizer.zero_grad()

            pred_dist = model(x_batch)
            loss = quantized_mnist_loss(pred_dist, x_batch).mean()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        with torch.no_grad():
            val_loss = []
            for (val_batch,) in tqdm(val_loader):
                val_loss.append(quantized_mnist_loss(model(val_batch), val_batch).mean().item())

            epoch_val_loss.append(np.mean(val_loss))

            if np.mean(val_loss) < best_val_loss:
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

    with torch.no_grad():
        test_loss = []

        for (test_batch,) in tqdm(test_loader):
            test_loss.append(quantized_mnist_loss(model(test_batch), test_batch).mean().item())
    print(
        f"Test loss: {np.mean(test_loss)} bits"
    )
