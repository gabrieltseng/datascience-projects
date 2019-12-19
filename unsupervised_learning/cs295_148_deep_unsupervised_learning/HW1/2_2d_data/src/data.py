import numpy as np

from typing import List, Tuple


def load_distribution() -> np.ndarray:

    return np.load("data/distribution.npy")


def generate_samples(probs: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Given a 2d input array probs, return a np.ndarray of shape [num_samples, 2]
    """
    flat_probs: List[float] = []
    indices: List[Tuple[int, int]] = []

    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            flat_probs.append(probs[i, j])
            indices.append((i, j))

    sampled_indices = np.random.choice(len(indices), size=num_samples, p=flat_probs)

    return np.asarray(indices)[sampled_indices]


def train_test_split(
    x: np.ndarray, val_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:

    return x[: int(len(x) * 1 - val_size)], x[-int(len(x) * val_size) :]
