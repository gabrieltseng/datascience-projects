import torch
from torch.utils.data import DataLoader

import numpy as np
from pathlib import Path
from tqdm import tqdm

from solarnet.preprocessing import MaskMaker, ImageSplitter
from solarnet.datasets import ClassifierDataset
from solarnet.models import Classifier, train_classifier


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
    def train_classifier(max_epochs=5, val_set=0.1, test_set=0.1,
                         device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

        model = Classifier()
        if device.type != 'cpu': model = model.cuda()
        dataset = ClassifierDataset()

        # make a train and val set
        mask = np.random.rand(len(dataset))
        train_mask = mask < (1 - (val_set + test_set))
        val_mask = (mask > (1 - (val_set + test_set))) & (mask < 1 - test_set)
        test_mask = mask < (1 - test_set)

        dataset.add_mask(train_mask)
        val_dataset = ClassifierDataset(mask=val_mask)
        test_dataset = ClassifierDataset(mask=test_mask)

        train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=64)

        train_classifier(model, train_dataloader, val_dataloader, max_epochs=max_epochs)

        savedir = Path('data/models')
        if not savedir.exists(): savedir.mkdir()
        torch.save(model.state_dict(), savedir / 'classifier.model')

        # save predictions for analysis
        print("Generating test results")
        preds, true = [], []
        with torch.no_grad():
            for test_x, test_y in tqdm(test_dataloader):
                test_preds = model(test_x)
                preds.append(test_preds.squeeze(1).cpu().numpy())
                true.append(test_y.cpu().numpy())

        np.save(savedir / 'classifier_preds.npy', np.concatenate(preds))
        np.save(savedir / 'classifier_true.npy', np.concatenate(true))

