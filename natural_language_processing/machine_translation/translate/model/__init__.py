import torch
from torch import nn

from .seq2seq import FrenchToEnglish
from .schedulers import OneCycle, TeacherForcing
from .train import find_learning_rate, train


class FlatCrossEntropyLoss(nn.Module):
    """
    Flattens a 3D prediction (and a 2D target), so that they can
    be passed to PyTorch's cross entropy loss module.
    From natural_language_processing/language_model/lm/models/__init__.py

    In addition, adds padding where appropriate to ensure the target and
    prediction sequences have the same length
    """

    def __init__(self, pad_idx=0):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.pad_idx = pad_idx

    def forward(self, pred, target):

        if pred.shape[1] < target.shape[1]:
            # somewhere, padding is needed
            padding_length = target.shape[1] - pred.shape[1]
            pred_padding = torch.zeros(pred.shape[0], padding_length,
                                       pred.shape[-1])
            if pred.is_cuda: pred_padding = pred_padding.cuda()
            pred_padding[:, :, self.pad_idx] = 1
            pred = torch.cat([pred, pred_padding], dim=1)
        elif pred.shape[1] > target.shape[1]:
            pred = pred[:, :target.shape[1], :].contiguous()

        return self.loss(
            pred.view(pred.shape[0] * pred.shape[1], pred.shape[2]),
            target.view(target.shape[0] * target.shape[1]))
