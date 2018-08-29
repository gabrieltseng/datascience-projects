import torch
import math
from copy import deepcopy

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


def prune_state_dict(pt_info, model_type='ConvLM'):
    """
    To implement dropconnect, a bunch of temporary layers are created
    which aren't initialized when the model is created (because they don't need
    to be). This removes them from the state dict when loading a pretrained model.
    """
    state_dict = deepcopy(pt_info['state_dict'])

    # first, the embedding layer
    state_dict.pop('embedding.embedding.weight')
    # then, the decoding layers
    state_dict.pop('decoder.weight')
    state_dict.pop('decoder.bias')

    if model_type == 'ConvLM':
        for block in range(pt_info['num_blocks']):
            block_name = 'TCNBlock_{}'.format(block)
            for layer in range(1, pt_info['num_layers'] + 1):
                layer_name = 'conv{}'.format(layer)
                extra_parameter = '{}.{}.conv.weight'.format(block_name, layer_name)
                state_dict.pop(extra_parameter)
    return state_dict
