import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from pathlib import Path
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from .base import ModelBase


class ConvModel(ModelBase):

    def __init__(self, in_channels=9, dropout=0.25, out_channels_list=None, stride_list=None,
                 dense_features=None, savedir=Path('data/models')):
        self.model = ConvNet(in_channels=in_channels, dropout=dropout,
                             out_channels_list=out_channels_list, stride_list=stride_list,
                             dense_features=dense_features)
        self.savedir = savedir

    def train(self, path_to_histogram=Path('data/img_output/histogram_all_full.npz'),
              pred_years=None, num_runs=2, train_steps=25000, batch_size=32,
              starter_learning_rate=1e-3):

        with np.load(path_to_histogram) as hist:
            images = hist['output_image']
            locations = hist['output_locations']
            yields = hist['output_yield']
            years = hist['output_year']
            indices = hist['output_index']

        if pred_years is None:
            pred_years = range(2009, 2016)
        for pred_year in pred_years:
            for run_number in range(1, num_runs + 1):
                print(f'Training to predict on {pred_year}, Run number {run_number}')
                self._train_1_year(images, yields, years, pred_year, run_number, train_steps, batch_size,
                                   starter_learning_rate)
                print('-----------')

        # TODO: delete broken images (?)

    def _train_1_year(self, images, yields, years, predict_year, run_number, train_steps,
                      batch_size, starter_learning_rate):
        train_indices = np.nonzero(years < predict_year)[0]
        val_indices = np.nonzero(years == predict_year)[0]

        print(f'Train set size: {train_indices.shape[0]}, Val set size: {val_indices.shape[0]}')

        train_images = torch.tensor(images[train_indices]).float()
        train_yields = torch.tensor(yields[train_indices]).float().unsqueeze(1)

        val_images = torch.tensor(images[val_indices]).float()
        val_yields = torch.tensor(yields[val_indices]).float().unsqueeze(1)

        train_dataset = TensorDataset(train_images, train_yields)
        val_dataset = TensorDataset(val_images, val_yields)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        criterion = nn.MSELoss()  # TODO: L1 loss as well?
        optimizer = torch.optim.Adam([pam for pam in self.model.parameters()],
                                     lr=starter_learning_rate)

        num_epochs = int(train_steps / train_indices.shape[0])
        print(f'Training for {num_epochs} epochs')
        step_number = 0

        train_scores = defaultdict(list)
        val_scores = defaultdict(list)

        for epoch in range(num_epochs):
            self.model.train()
            running_train_scores = defaultdict(list)

            for train_x, train_y in tqdm(train_dataloader):
                optimizer.zero_grad()
                pred_y = self.model(train_x)
                loss = criterion(pred_y, train_y)
                loss.backward()
                optimizer.step()

                running_train_scores['loss'].append(loss.item())
                train_scores['loss'].append(loss.item())

                step_number += 1
                if (step_number == 2000) or (step_number == 4000):
                    for param_group in optimizer.param_groups:
                        param_group['lr'] /= 10

            train_output_strings = []
            for key, val in running_train_scores.items():
                train_output_strings.append('{}: {}'.format(key, round(np.array(val).mean(), 5)))

            running_val_scores = defaultdict(list)
            self.model.eval()
            with torch.no_grad():
                for val_x, val_y in tqdm(val_dataloader):
                    val_pred_y = self.model(val_x)
                    val_loss = criterion(val_pred_y, val_y)
                    running_val_scores['loss'].append(val_loss.item())
                    val_scores['loss'].append(val_loss.item())
            val_output_strings = []
            for key, val in running_val_scores.items():
                val_output_strings.append('{}: {}'.format(key, round(np.array(val).mean(), 5)))
            print('TRAINING: {}'.format(*train_output_strings))
            print('VALIDATION: {}'.format(*val_output_strings))

            model_information = {
                'state_dict': self.model.state_dict(),
                'val_loss': val_scores,
                'train_loss': train_scores
            }
            filename = f'{predict_year}_{run_number}.pth.tar'
            torch.save(model_information, self.savedir / filename)


class ConvNet(nn.Module):
    """
    A crop yield conv net.

    Parameters
    ----------
    in_channels: int, default=9
        Number of channels in the input data. Default taken from the number of bands in the
        MOD09A1 + the number of bands in the MYD11A2 datasets
    dropout: float, default=0.25
        Default taken from the original repository
    out_channels_list: list or None, default=None
        Out_channels of all the convolutional blocks. If None, default values will be taken
        from the paper. Note the length of the list defines the number of conv blocks used.
    stride_list: list or None, default=None
        Strides of all the convolutional blocks. If None, default values will be taken from the paper
        If not None, must be equal in length to the out_channels_list
    dense_features: list, or None, default=None.
        output feature size of the Linear layers. If None, default values will be taken from the paper.
        The length of the list defines how many linear layers are used.
    """
    def __init__(self, in_channels=9, dropout=0.25, out_channels_list=None, stride_list=None,
                 dense_features=None):
        super().__init__()

        # default values taken from the paper
        if out_channels_list is None:
            out_channels_list = [128, 256, 256, 512, 512, 1024]
        out_channels_list.insert(0, in_channels)

        if stride_list is None:
            stride_list = [1, 2, 1, 2, 1, 2]
        stride_list.insert(0, 0)

        if dense_features is None:
            dense_features = [1024, 1]
        dense_features.insert(0, out_channels_list[-1])

        assert len(stride_list) == len(out_channels_list), \
            "Stride list and out channels list must be the same length!"

        self.convblocks = nn.ModuleList([
            ConvBlock(in_channels=out_channels_list[i-1], out_channels=out_channels_list[i],
                      kernel_size=3, stride=stride_list[i],
                      dropout=dropout) for i in range(1, len(stride_list))
        ])

        self.dense_layers = nn.ModuleList([
            nn.Linear(in_features=dense_features[i-1],
                      out_features=dense_features[i]) for i in range(1, len(dense_features))
        ])
        self.dense = nn.Linear(in_features=1024, out_features=1024)

    def forward(self, x):
        for block in self.convblocks:
            x = block(x)
        x = x.squeeze(-1).squeeze(-1)

        for dense_layer in self.dense_layers:
            x = dense_layer(x)
        return x


class ConvBlock(nn.Module):
    """
    A 2D convolution, followed by batchnorm, a ReLU activation, and dropout
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.batchnorm(self.relu(self.conv(x)))
        return self.dropout(x)
