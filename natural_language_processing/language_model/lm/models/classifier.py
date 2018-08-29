from torch import nn
import torch.nn.functional as F
import numpy as np


class LanguageClassifier(nn.Module):
    """
    Classifier which extends the base model
    """

    def __init__(self, language_base, base_output_dim=400, num_layers=3,
                 num_classes=6, dropout=0.1):
        super().__init__()

        self.pretrained = language_base

        layer_outputs = np.linspace(base_output_dim, num_classes, num=num_layers + 1)[1:]
        num_inputs = base_output_dim
        for layer_num, num_outputs in enumerate(layer_outputs):
            # linear layer, with a relu activation added in the forward pass
            layer = nn.Linear(num_inputs, int(num_outputs))
            setattr(self, 'finetune_linear_{}'.format(layer_num), layer)
            batchnorm = nn.BatchNorm1d(int(num_outputs))
            setattr(self, 'finetune_batchnorm_{}'.format(layer_num), batchnorm)
            num_inputs = int(num_outputs)
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = num_layers

    def forward(self, *x):
        return_hidden = False
        x = self.pretrained(*x)

        if type(x) is tuple:
            x, new_hidden, final_rnn_hidden = x
            return_hidden = True

        for i in range(self.num_layers):
            x = self.dropout(getattr(self, 'finetune_linear_{}'.format(i))(x))
            x = getattr(self, 'finetune_batchnorm_{}'.format(i))(x)
            x = F.relu(x)
        if return_hidden: return x, new_hidden
        return x
