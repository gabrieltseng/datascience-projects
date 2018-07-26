import torch
from torch import nn
from torchvision.models import resnet34


class SODNet(nn.Module):
    """
    Single Object Detection network:
    Finetuning resnet
    """

    def __init__(self, num_classes):
        super().__init__()

        # first, take all the layers of resnet up to the last one
        resnet = resnet34(pretrained=True).float()
        self.pretrained = nn.Sequential(*list(resnet.children())[:-2])

        self.finetune_interim = nn.Linear(25088, 256)
        self.batchnorm = nn.BatchNorm1d(256)

        # we will have 4 output classes (xmin, ymin, xmax, ymax)
        self.finetune_sod = nn.Linear(256, 4)

        # in addition, we will have a multiclass classifier
        self.finetune_label = nn.Linear(256, num_classes)

    def forward(self, x):
        f = self.pretrained(x)
        f = f.view(f.size(0), -1)
        f = self.batchnorm(nn.functional.relu(self.finetune_interim(f)))

        # multiply by 224, to make sure the bounding box coordinates are
        # within the image. This points the neural net in the right direction
        bounding_boxes = nn.functional.sigmoid((self.finetune_sod(f))) * 224
        labels = self.finetune_label(f)
        return bounding_boxes, labels


def accuracy(output_labels, true_labels):
    """
    For a more interpretable metric, calculate the accuracy of the predictions
    """
    output_labels = torch.nn.functional.softmax(output_labels, dim=1).argmax(dim=1)
    correct = torch.eq(true_labels, output_labels).sum().item()
    accuracy = correct / output_labels.shape[0]
    return accuracy


def get_sod_weight(model, inputs):
    """
    Calculate the scalar factor which allows
    the weights to be combined in a comparable manner
    """

    # first, lets define our losses
    bb_criterion = torch.nn.modules.loss.L1Loss()
    label_criterion = torch.nn.modules.loss.CrossEntropyLoss()

    im, bb, lab = inputs

    output_bb, output_labels = model(im)

    bb_loss = bb_criterion(output_bb, bb.float())
    label_loss = label_criterion(output_labels, lab.long())

    return abs(float((label_loss / bb_loss).detach()))