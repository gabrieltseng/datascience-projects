import torch
from torch import nn


class PhysioNet(nn.Module):
    """
    An RNN to predict in hospital mortality. From the Che et al. architecture:
    "For RNN models, we use a one layer RNN to model the sequence, and then apply a soft-max
    regressor on top of the last hidden state to do classification."

    Default values taken from the dropout feature ranking paper
    """
    def __init__(self, input_size=37, hidden_size=64):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.regressor = nn.Linear(in_features=hidden_size, out_features=1)
        self.dropout_1 = nn.Dropout(0.3)
        self.dropout_2 = nn.Dropout(0.5)
        self.batchnorm = nn.BatchNorm1d(num_features=hidden_size)

    def forward(self, x):
        output, hidden = self.gru(self.dropout_1(x))
        # we only want the final layer to be returned
        prediction = self.regressor(self.batchnorm(self.dropout_2(output[:, -1, :].squeeze(1))))
        return torch.sigmoid(prediction)
