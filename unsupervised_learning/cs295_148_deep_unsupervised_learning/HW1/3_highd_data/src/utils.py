import pickle
from pathlib import Path
import numpy as np

import torch

from typing import Tuple


def load_data(
    data_path: Path = Path("data/mnist-hw1.pkl"), val_size: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    with data_path.open("rb") as f:
        data = pickle.load(f)

    train_data = data["train"]
    test_data = data["test"]

    # swap the channels to be channels first
    train_data = train_data.swapaxes(-1, 1)
    test_data = test_data.swapaxes(-1, 1)

    mask = np.random.random(size=len(train_data))

    train, val = train_data[mask >= val_size], train_data[mask < val_size]

    return (
        torch.from_numpy(train).float(),
        torch.from_numpy(val).float(),
        torch.from_numpy(test_data).float(),
    )
