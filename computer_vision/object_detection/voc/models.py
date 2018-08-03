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
        self.finetune_batchnorm = nn.BatchNorm1d(256)

        # we will have 4 output classes (xmin, ymin, xmax, ymax)
        self.finetune_bb = nn.Linear(256, 4)

        # in addition, we will have a multiclass classifier
        self.finetune_label = nn.Linear(256, num_classes)
        self.finetune_dropout = nn.Dropout()

    def forward(self, x):
        f = self.pretrained(x)
        f = self.finetune_dropout(nn.functional.relu(f.view(f.size(0), -1)))
        f = nn.functional.relu(self.finetune_interim(f))
        f = self.finetune_dropout(self.finetune_batchnorm(f))

        # multiply by 224, to make sure the bounding box coordinates are
        # within the image. This points the neural net in the right direction
        bounding_boxes = nn.functional.sigmoid((self.finetune_bb(f))) * 224
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


class SSDNet(nn.Module):
    """
    Single shot multi object detection
    """
    def __init__(self, num_classes, num_permutations):
        super().__init__()

        # first, take all the layers of resnet up to the last one
        resnet = resnet34(pretrained=True).float()
        self.pretrained = nn.Sequential(*list(resnet.children())[:-2])

        # the last output of the pretrained net has shape (7, 7, 512)
        self.first_conv = nn.Conv2d(512, 256, 3, stride=2, padding=1)
        self.conv = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.conv_out_bb = nn.Conv2d(256, num_permutations * 4, 3, stride=1, padding=1)
        self.conv_out_lab = nn.Conv2d(256, num_permutations * num_classes, 3, stride=1, padding=1)
        # batchnorm stats copied from what is happening in resnet
        self.batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1,
                                         affine=True, track_running_stats=True)
        self.dropout = nn.Dropout()
        self.num_permutations = num_permutations

    @staticmethod
    def flatten(tensor):
        channels, depth, width, height = tensor.shape
        tensor = tensor.permute(0, 2, 3, 1).contiguous()
        return tensor.view(channels, (depth * width * height))

    def forward(self, x):
        x = self.pretrained(x)
        # first convolutional layer
        x = self.dropout(self.batchnorm(self.first_conv(x)))
        bb_4, lab_4 = self.conv_out_bb(x), self.conv_out_lab(x)
        x = self.dropout(self.batchnorm(self.conv(x)))
        bb_2, lab_2 = self.conv_out_bb(x), self.conv_out_lab(x)
        x = self.dropout(self.batchnorm(self.conv(x)))
        bb_1, lab_1 = self.conv_out_bb(x), self.conv_out_lab(x)

        # flatten and concatenate the outputs
        bb = torch.cat([self.flatten(bb_4),
                        self.flatten(bb_2),
                        self.flatten(bb_1)], 1)
        labels = torch.cat([self.flatten(lab_4),
                            self.flatten(lab_2),
                            self.flatten(lab_1)], 1)

        return bb, labels
