import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from typing import List


def sample_data() -> np.ndarray:

    count = 10000
    rand = np.random.RandomState(0)
    a = 0.3 + 0.1 * rand.randn(count)
    b = 0.8 + 0.05 * rand.randn(count)
    mask = rand.rand(count) < 0.5
    samples = np.clip(a * mask + b * (1 - mask), 0.0, 1.0)
    return np.digitize(samples, np.linspace(0.0, 1.0, 100))


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(1, 100))
        self.softmax = nn.Softmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax(x * self.theta)


def train_model(num_epochs: int = 100, early_stopping: int = 5):

    model = Model()
    data = np.expand_dims(sample_data(), -1)

    train_data = torch.from_numpy(data[:int(len(data) * 0.64)])
    val_data = torch.from_numpy(data[int(len(data) * 0.64):int(len(data) * 0.8)])

    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=64,
        shuffle=True,
    )
    test_data = torch.from_numpy(data[int(len(data) * 0.2):])

    optimizer = torch.optim.Adam(model.parameters())

    epoch_train_loss: List[float] = []
    epoch_val_loss: List[float] = []

    best_val_loss = np.inf
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss: List[np.ndarray] = []
        for x_batch, in train_loader:
            optimizer.zero_grad()

            loss = (- torch.log(model(x_batch))).mean()

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        epoch_train_loss.append(np.mean(train_loss))

        with torch.no_grad():
            val_prob = model(val_data)
            val_loss = (- torch.log(val_prob)).mean()

            epoch_val_loss.append(np.mean(val_prob.numpy()))
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    print("Early stopping!")
        print(f"Train loss: {np.mean(train_loss)}, Val loss: {val_loss.item()}")

    print(f"Test loss: {- torch.log(model(test_data)).mean()}")


if __name__ == '__main__':
    train_model()
