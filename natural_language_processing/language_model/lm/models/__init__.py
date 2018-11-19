import torch
from torch import nn
import math
from copy import deepcopy

from lm.models.awd_lstm import RecLM, ARTAR
from lm.models.tcn import ConvLM
from lm.models.classifier import LanguageClassifier


class FlatCrossEntropyLoss(nn.Module):
    """
    Flattens a 3D prediction (and a 2D target), so that they can
    be passed to PyTorch's cross entropy loss module
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        return self.loss(
            pred.view(pred.shape[0] * pred.shape[1], pred.shape[2]),
            target.contiguous().view(target.shape[0] * target.shape[1]))


def accuracy(output_labels, true_labels, flatten=True):
    """
    For a more interpretable metric, calculate the accuracy of the predictions.
    Copied (and slightly modified) from
    https://github.com/GabrielTseng/LearningDataScience/blob/master/computer_vision/object_detection/voc/models.py#L44
    """
    if flatten:
        output_labels = output_labels.view(output_labels.shape[0] * output_labels.shape[1],
                                           output_labels.shape[2])
        true_labels = true_labels.contiguous().view(true_labels.shape[0] * true_labels.shape[1])
    if output_labels.shape[1] > 1:
        output_labels = torch.nn.functional.softmax(output_labels, dim=1).argmax(dim=1)
    else:
        output_labels = torch.round(torch.nn.functional.sigmoid(output_labels))
    correct = torch.eq(true_labels, output_labels).sum().item()
    accuracy = correct / output_labels.shape[0]
    return accuracy


def perplexity(cross_entropy_loss):
    """
    https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf

    return 2^J, where J is the cross entropy loss.
    """
    if type(cross_entropy_loss) is torch.Tensor:
        cross_entropy_loss = cross_entropy_loss.item()
    return math.pow(2, cross_entropy_loss)


def prune_state_dict(pt_info):
    """
    Remove the decoder layers, so they can be replaced
    with a classifier
    """
    state_dict = deepcopy(pt_info['state_dict'])
    # then, the decoding layers
    state_dict.pop('decoder.weight')
    state_dict.pop('decoder.bias')
    return state_dict
