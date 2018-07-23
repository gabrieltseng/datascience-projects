import torch
from torch.utils.data import Dataset, TensorDataset
import cv2
from pathlib import Path

import numpy as np

from voc.utils import xml_to_dict, load_image, keep_largest_box


class VOCDataset(object):
    """Base for VOC dataset.

    All other classes which extend this should implement
    a __getitem__ method and a __len__ method.
    """

    def __init__(self, annotations=Path('VOC2007/Annotations')):
        self.annotations_files = sorted([x for x in annotations.iterdir()])


class ImageDataset(VOCDataset, Dataset):

    def __init__(self, annotations=Path('VOC2007/Annotations'), normalizer=None,
                 resize=None):
        super().__init__()
        self.normalizing_dict = {
            'imagenet': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
            'inception': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
        }
        if normalizer:
            assert normalizer in self.normalizing_dict, "Normalizer not one of {}". \
                format([val for val in self.normalizing_dict.keys()])
        self.normalizer = normalizer
        self.resize = resize

        largest_items = [keep_largest_box(xml_to_dict(ano))['objects'][0]
                        for ano in self.annotations_files]

        # first, the bounding boxes
        bounding_boxes = [item['coordinates'] for item in largest_items]
        self.bounding_boxes = torch.from_numpy(np.asarray(bounding_boxes)).float()

        # next, the item labels
        label_names = [item['name'] for item in largest_items]
        # turn the labels into integer classes
        self.label2class = {val: idx for idx, val in enumerate(set(label_names))}
        label_classes = [self.label2class[item] for item in label_names]
        self.labels = torch.from_numpy(np.asarray(label_classes)).int()

    def get_labels_from_classes(self):
        return self.label2class

    def _normalize(self, image):
        if self.normalizer:
            mean = self.normalizing_dict[self.normalizer]['mean']
            std = self.normalizing_dict[self.normalizer]['std']
            image = (image - mean) / std
        # in addition, roll the axis so that they suit pytorch
        return image.swapaxes(2, 0)

    def _resize(self, image):
        if self.resize:
            width, height = self.resize
            image = cv2.resize(image, (width, height),
                               interpolation=cv2.INTER_AREA)
        return image

    def __getitem__(self, index):
        annotation = xml_to_dict(self.annotations_files[index])
        image_path = annotation['image_path']
        image = self._normalize(self._resize(load_image(image_path)))
        return torch.from_numpy(image).float(), \
            self.bounding_boxes[index], self.labels[index]

    def __len__(self):
        return len(self.annotations_files)
