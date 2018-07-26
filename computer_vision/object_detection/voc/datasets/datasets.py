import torch
from torch.utils.data import Dataset, TensorDataset
import cv2
from pathlib import Path

import numpy as np
import random

from voc.utils import xml_to_dict, load_image, keep_largest_box, normalize
from .transforms import no_change, horizontal_flip, vertical_flip, colour_jitter


class VOCDataset(object):
    """Base for VOC dataset.

    All other classes which extend this should implement
    a __getitem__ method
    """

    def __init__(self, annotations=Path('VOC2007/Annotations'), mask=None):
        annotations_files = np.array(sorted([x for x in annotations.iterdir()]))
        if mask is not None:
            annotations_files = annotations_files[mask]
        self.annotations_files = annotations_files

    def __len__(self):
        return len(self.annotations_files)


class ImageDataset(VOCDataset, Dataset):
    """
    Given a path to annotations in a standard VOC dataset, return a Dataset object
    which yields the images, bounding boxes and labels
    """

    def __init__(self, annotations=Path('VOC2007/Annotations'), mask=None, normalizer=None,
                 resize=None, random_transform=False, device=torch.device("cpu"),
                 label2class=None):
        """
        annotations: a pathlib Path to the annotations (i.e. the xml files)
        mask: if making a train and val dataset, pass a mask to differentiate the training
            and validation examples. The mask should consist of an array of booleans.
        normalizer: one of {'imagenet', 'inception'}; how to normalize the images
        resize: a (height, width) tuple to resize the images to. If not None, the bounding
            box coordinates will also be appropriately resized
        random_transform: boolean, whether or not to randomly transform the images
        device: the device on which the tensors should be. Default is the CPU
        """

        super().__init__(annotations=annotations, mask=mask)

        self.device = device
        self.normalizing_dict = {
            'imagenet': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
            'inception': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
        }
        if normalizer:
            assert normalizer in self.normalizing_dict, "Normalizer not one of {}". \
                format([val for val in self.normalizing_dict.keys()])
        self.normalizer = normalizer
        self.resize = resize
        self.random_transform = random_transform

        largest_items = [keep_largest_box(xml_to_dict(ano))['objects'][0]
                         for ano in self.annotations_files]

        # first, the bounding boxes
        bounding_boxes = [item['coordinates'] for item in largest_items]
        self.bounding_boxes = np.asarray(bounding_boxes)

        # next, the item labels
        label_names = [item['name'] for item in largest_items]
        # turn the labels into integer classes
        if not label2class:
            self.label2class = {val: idx for idx, val in enumerate(set(label_names))}
        else:
            self.label2class = label2class

        label_classes = [self.label2class[item] for item in label_names]
        self.labels = torch.tensor(np.asarray(label_classes), dtype=torch.long,
                                   device=self.device)

    def get_labels_from_classes(self):
        return self.label2class

    def get_normalizer(self):
        return self.normalizing_dict[self.normalizer]

    def _normalize(self, image):
        mean = std = None
        if self.normalizer:
            mean = self.normalizing_dict[self.normalizer]['mean']
            std = self.normalizing_dict[self.normalizer]['std']
        # in addition, roll the axis so that they suit pytorch
        return normalize(image, mean, std)

    def _resize(self, image, bounding_box):
        if self.resize:
            width, height = self.resize
            xmin, ymin, xmax, ymax = bounding_box

            x_ratio = width / image.shape[1]
            y_ratio = height / image.shape[0]
            bounding_box = np.asarray([xmin * x_ratio, ymin * y_ratio,
                                      xmax * x_ratio, ymax * y_ratio])

            image = cv2.resize(image, (width, height),
                               interpolation=cv2.INTER_AREA)

        return image, bounding_box

    def _transform_image(self, image, bounding_box):

        transforms = [
            no_change,
            horizontal_flip,
            vertical_flip,
            colour_jitter,
        ]

        chosen_function = random.choice(transforms)
        image, bounding_box = chosen_function(image, bounding_box)
        return image, bounding_box

    def __getitem__(self, index):
        annotation = xml_to_dict(self.annotations_files[index])
        image_path = annotation['image_path']
        image, bb = self._resize(load_image(image_path),
                                 self.bounding_boxes[index])
        if self.random_transform:
            image, bb = self._transform_image(image, bb)
        image = self._normalize(image)
        return torch.tensor(image, dtype=torch.float, device=self.device), \
            torch.tensor(bb, dtype=torch.double, device=self.device), \
            self.labels[index]
