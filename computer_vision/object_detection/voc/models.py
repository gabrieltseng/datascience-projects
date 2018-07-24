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

        # we will have 4 output classes (xmin, ymin, xmax, ymax)
        self.finetune_sod = nn.Linear(25088, 4)

        # in addition, we will have a multiclass classifier
        self.finetune_label = nn.Linear(25088, num_classes)
        # dim 0 are the batches, dim 1 are the classes
        self.finetune_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        f = self.pretrained(x)
        f = f.view(f.size(0), -1)

        # multiply by 224, to make sure the bounding box coordinates are
        # within the image. This points the neural net in the right direction
        bounding_boxes = self.finetune_softmax(self.finetune_sod(f)) * 224
        labels = self.finetune_softmax(self.finetune_label(f))
        return bounding_boxes, labels