import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.categorical import Categorical

from typing import List


def sample_data() -> np.ndarray:

    count = 10000
    rand = np.random.RandomState(0)
    a = 0.3 + 0.1 * rand.randn(count)
    b = 0.8 + 0.05 * rand.randn(count)
    mask = rand.rand(count) < 0.5
    samples = np.clip(a * mask + b * (1 - mask), 0.0, 1.0)
    return np.digitize(samples, np.linspace(0.0, 1.0, 100))


def plot_loss(train_loss: List[float], val_loss: List[float], epoch_length: int):
    fig, ax = plt.subplots()

    ax.scatter(range(len(train_loss)), train_loss, label="Train loss")
    ax.scatter([idx * epoch_length for idx, _ in enumerate(val_loss)], val_loss, label="Val loss")
    plt.legend()

    plt.xlabel("Training step")
    plt.ylabel("Loss (bits)")

    plt.savefig("diagrams/losses.png", dpi=300, bbox_inches="tight")


def plot_probabilities(x: np.ndarray, y: np.ndarray):
    fig, ax = plt.subplots()

    ax.bar(x, y)
    plt.title("Model probabilities")

    plt.savefig("diagrams/model_probabilities.png", dpi=300, bbox_inches="tight")


def draw_samples(model: nn.Module, num_samples: int) -> List[int]:

    samples: List[int] = []
    for _ in range(num_samples):
        input = torch.tensor(np.random.randint(1, 100))
        samples.append(Categorical(F.softmax(model(input))).sample())

    return samples


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(1, 100))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.theta


def train_model(num_epochs: int = 100, early_stopping: int = 5):

    model = Model()
    data = np.expand_dims(sample_data(), -1)

    train_data = torch.from_numpy(data[:int(len(data) * 0.64)])
    val_data = torch.from_numpy(data[int(len(data) * 0.64):int(len(data) * 0.8)])

    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=128,
        shuffle=True,
    )
    test_data = torch.from_numpy(data[int(len(data) * 0.2):])

    optimizer = torch.optim.Adam(model.parameters())
    loss_function = nn.CrossEntropyLoss()

    train_loss: List[float] = []
    epoch_val_loss: List[float] = []

    best_val_loss = np.inf
    patience_counter = 0
    best_model_dict = None

    for epoch in range(num_epochs):
        for x_batch, in train_loader:
            optimizer.zero_grad()

            loss = loss_function(model(x_batch), x_batch.squeeze(1)).mean()

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        with torch.no_grad():
            val_loss = loss_function(model(val_data), val_data.squeeze(1)).mean()

            epoch_val_loss.append(val_loss.item())

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
        print(f"Epoch {epoch + 1}: Train loss: {np.mean(train_loss)}, Val loss: {val_loss.item()}")

    plot_loss(train_loss, epoch_val_loss, len(train_loader))

    print(f"Test loss: {loss_function(model(test_data), test_data.squeeze(1)).mean()}")

    with torch.no_grad():
        x = np.arange(1, 100)
        probabilities = (F.softmax(model(torch.from_numpy(np.expand_dims(x, 1))))).numpy()

        relevant_probabilities = [prob[x_i] for prob, x_i in zip(probabilities, x)]
        plot_probabilities(x, np.asarray(relevant_probabilities))

    plt.clf()
    plt.hist(data.squeeze(1))
    plt.title("Original Data Distribution")
    plt.savefig("diagrams/original_data_distribution.png", dpi=300, bbox_inches="tight")

    samples = draw_samples(model, num_samples=1000)
    plt.clf()
    plt.hist(samples)
    plt.title("Sampled Data Distribution")
    plt.savefig("diagrams/sampled_data_distribution.png", dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    train_model()
