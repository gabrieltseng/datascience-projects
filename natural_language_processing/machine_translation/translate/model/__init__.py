import torch
from torch import nn

from .seq2seq import FrenchToEnglish
from .schedulers import OneCycle


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
            pred_right_size = torch.zeros(pred.shape[0], target.shape[1],
                                          pred.shape[-1])
            if pred.is_cuda: pred_right_size = pred_right_size.cuda()
            pred_right_size[:, :, self.pad_idx] = 1
            pred_right_size[:, :pred.shape[1], :] = pred
            pred = pred_right_size
        elif pred.shape[1] > target.shape[1]:
            target_right_size = torch.ones(target.shape[0], pred.shape[1]).long() * self.pad_idx
            if target.is_cuda: target_right_size = target_right_size.cuda()
            target_right_size[:, :target.shape[1]] = target
            target = target_right_size

        return self.loss(
            pred.view(pred.shape[0] * pred.shape[1], pred.shape[2]),
            target.contiguous().view(target.shape[0] * target.shape[1]))
