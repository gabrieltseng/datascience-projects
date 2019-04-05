import numpy as np
import torch
from pathlib import Path

from .utils import normalize


class ClassifierDataset:
    def __init__(self, processed_folder=Path('data/processed'), normalize=True,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

        self.device = device
        self.normalize = normalize

        solar_files = list((processed_folder / 'solar/org').glob("*.npy"))
        empty_files = list((processed_folder / 'empty/org').glob("*.npy"))

        self.y = torch.as_tensor([1 for _ in solar_files] + [0 for _ in empty_files],
                                 device=self.device).float()
        self.x_files = solar_files + empty_files

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        y = self.y[index]
        x = np.load(self.x_files[index])
        if self.normalize: x = normalize(x)
        return torch.as_tensor(x, device=self.device).float(), y
