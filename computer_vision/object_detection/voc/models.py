import torch
from torch import nn
from torchvision.models import resnet34

from .utils import bbox_to_jaccard, jaccard_to_anchor, anchor_to_class, activations_to_ratios


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
        """
        :param num_classes: num_classes + 1 (for the background)
        :param num_permutations: permutations for each anchor box
        """
        super().__init__()

        # first, take all the layers of resnet up to the last one
        resnet = resnet34(pretrained=True).float()
        self.pretrained = nn.Sequential(*list(resnet.children())[:-2])

        # the last output of the pretrained net has shape (7, 7, 512)
        self.conv_4 = nn.Conv2d(512, 256, 3, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.conv_1 = nn.Conv2d(256, 256, 3, stride=2, padding=1)

        self.conv_out_bb_4 = nn.Conv2d(256, num_permutations * 4, 3, stride=1, padding=1)
        self.conv_out_lab_4 = nn.Conv2d(256, num_permutations * (num_classes + 1), 3, stride=1, padding=1)
        self.conv_out_bb_2 = nn.Conv2d(256, num_permutations * 4, 3, stride=1, padding=1)
        self.conv_out_lab_2 = nn.Conv2d(256, num_permutations * (num_classes + 1), 3, stride=1, padding=1)
        self.conv_out_bb_1 = nn.Conv2d(256, num_permutations * 4, 3, stride=1, padding=1)
        self.conv_out_lab_1 = nn.Conv2d(256, num_permutations * (num_classes + 1), 3, stride=1, padding=1)
        # batchnorm stats copied from what is happening in resnet
        self.batchnorm_4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1,
                                          affine=True, track_running_stats=True)
        self.batchnorm_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1,
                                          affine=True, track_running_stats=True)
        self.batchnorm_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1,
                                          affine=True, track_running_stats=True)
        self.dropout_4 = nn.Dropout()
        self.dropout_2 = nn.Dropout()
        self.dropout_1 = nn.Dropout()

        self.num_permutations = num_permutations

    @staticmethod
    def flatten(tensor):
        channels, depth, width, height = tensor.shape
        tensor = tensor.permute(0, 2, 3, 1).contiguous()
        return tensor.view(channels, (depth * width * height))

    def forward(self, x):
        x = self.pretrained(x)
        # first convolutional layer
        x = self.dropout_4(self.batchnorm_4(self.conv_4(x)))
        bb_4, lab_4 = self.conv_out_bb_4(x), self.conv_out_lab_4(x)
        x = self.dropout_2(self.batchnorm_2(self.conv_2(x)))
        bb_2, lab_2 = self.conv_out_bb_2(x), self.conv_out_lab_2(x)
        x = self.dropout_1(self.batchnorm_1(self.conv_1(x)))
        bb_1, lab_1 = self.conv_out_bb_1(x), self.conv_out_lab_1(x)

        # flatten and concatenate the outputs
        bb = torch.cat([self.flatten(bb_4),
                        self.flatten(bb_2),
                        self.flatten(bb_1)], 1)
        labels = torch.cat([self.flatten(lab_4),
                            self.flatten(lab_2),
                            self.flatten(lab_1)], 1)

        return bb, labels


class SSDLoss(nn.Module):

    def __init__(self, anchors, threshold, num_classes, background_index=20, device=torch.device("cpu")):
        super().__init__()
        self.anchors = anchors
        self.threshold = threshold
        self.num_classes = num_classes
        self.background_index = background_index
        self.device = device

    def forward(self, target_bb_batch, target_label_batch, pred_bb_batch, pred_label_batch):
        # iterate through each example in the batch
        total_bb_loss = 0
        total_label_loss = 0
        for target_bb, target_label, pred_bb, pred_label in zip(target_bb_batch, target_label_batch,
                                                                pred_bb_batch, pred_label_batch):
            jaccard = bbox_to_jaccard(self.anchors, target_bb)
            anchor_val, anchor_idx = jaccard_to_anchor(jaccard)
            anchor_classes, object_indices, objects = anchor_to_class(anchor_val,
                                                                      anchor_idx, target_label,
                                                                      threshold=self.threshold,
                                                                      background_index=self.background_index)

            # turn the anchor classes into targets, and lop off the end so that we aren't training
            # the model to recognize background
            target_label_one_hot = torch.eye(self.num_classes + 1, device=self.device)[anchor_classes][:, :-1]
            pred_label = pred_label.view(-1, (self.num_classes + 1))[:, :-1]
            # get the weight to turn this into a focal loss
            alpha, gamma = 0.25, 1
            pred_sigmoid = torch.sigmoid(pred_label)
            p_t = pred_sigmoid * target_label_one_hot + (1 - pred_sigmoid) * (1 - target_label_one_hot)
            alpha_t = alpha * target_label_one_hot + (1 - alpha) * (1 - target_label_one_hot)
            focal_weight = alpha_t * (1 - p_t).pow(gamma)
            label_loss = nn.functional.binary_cross_entropy_with_logits(pred_label,
                                                                        target_label_one_hot,
                                                                        size_average=False,
                                                                        weight=focal_weight) / self.num_classes

            # next, the bounding box loss
            # first, lets turn the bounding box from pixel values into ratios
            target_bb = target_bb.view(-1, 4) / 224
            # first, lets get the bounding box predictions we care about
            pred_bb = pred_bb.view(-1, 4)[object_indices]

            # and the actual anchor coordinates for the anchors we care about
            relevant_anchors = self.anchors[object_indices] / 224
            # and finally, the bounding boxes
            target_bb = target_bb[objects]
            pred_bb_pixels = activations_to_ratios(pred_bb, relevant_anchors)
            # Finally, its just L1 loss
            bb_loss = torch.nn.functional.l1_loss(pred_bb_pixels, target_bb)
            total_bb_loss += bb_loss
            total_label_loss += label_loss
        return total_bb_loss, total_label_loss
