import torch
import math
from copy import deepcopy

from lm.models.awd_lstm import RecLM, ARTAR
from lm.models.tcn import ConvLM
from lm.models.classifier import LanguageClassifier


def accuracy(output_labels, true_labels):
    """
    For a more interpretable metric, calculate the accuracy of the predictions.
    Copied from
    https://github.com/GabrielTseng/LearningDataScience/blob/master/computer_vision/object_detection/voc/models.py#L44
    """
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
