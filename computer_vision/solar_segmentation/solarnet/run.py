import torch
from torch.utils.data import DataLoader

from solarnet.preprocessing import MaskMaker, ImageSplitter
from solarnet.datasets import ClassifierDataset
from solarnet.models import Classifier, train_classifier_epoch


class RunTask:

    @staticmethod
    def make_masks():
        mask_maker = MaskMaker()
        mask_maker.process()

    @staticmethod
    def split_images():
        splitter = ImageSplitter()
        splitter.process()

    @staticmethod
    def train_classifier(num_epochs=5,
                         device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

        model = Classifier()
        if device.type != 'cpu': model = model.cuda()
        dataloader = DataLoader(ClassifierDataset(), batch_size=64, shuffle=True)

        for i in range(num_epochs):
            if i <= 2:
                # we start by finetuning the model
                optimizer = torch.optim.Adam([pam for name, pam in
                                              model.named_parameters() if 'classifier' in name])
            else:
                # then, we train the whole thing
                optimizer = torch.optim.Adam(model.parameters())

            _ = train_classifier_epoch(model, optimizer, dataloader)
