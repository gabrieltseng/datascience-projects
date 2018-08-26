import torch
import math

from lm.models.awd_lstm import RecLM, ARTAR
from lm.models.tcn import ConvLM


def accuracy(output_labels, true_labels):
    """
    For a more interpretable metric, calculate the accuracy of the predictions.
    Copied from
    https://github.com/GabrielTseng/LearningDataScience/blob/master/computer_vision/object_detection/voc/models.py#L44
    """
    output_labels = torch.nn.functional.softmax(output_labels, dim=1).argmax(dim=1)
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
