import torch
from torch.utils.data import DataLoader

import numpy as np

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
    def train_classifier(num_epochs=5, val_set=0.1,
                         device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

        model = Classifier()
        if device.type != 'cpu': model = model.cuda()
        dataset = ClassifierDataset()

        # make a train and val set
        train_mask = np.random.rand(len(dataset)) < (1 - val_set)
        val_mask = ~train_mask

        dataset.add_mask(train_mask)
        val_dataset = ClassifierDataset(mask=val_mask)

        train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=64)

        for i in range(num_epochs):
            if i <= 2:
                # we start by finetuning the model
                optimizer = torch.optim.Adam([pam for name, pam in
                                              model.named_parameters() if 'classifier' in name])
            else:
                # then, we train the whole thing
                optimizer = torch.optim.Adam(model.parameters())

            _, _ = train_classifier_epoch(model, optimizer, train_dataloader, val_dataloader)
