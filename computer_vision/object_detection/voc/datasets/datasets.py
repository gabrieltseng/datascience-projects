import torch
from torch.utils.data import Dataset, TensorDataset
import cv2
from pathlib import Path

import numpy as np
import random

from voc.utils import xml_to_dict, load_image, keep_largest_box, normalize
from .transforms import no_change, horizontal_flip, colour_jitter, rotate


class VOCDataset(object):
    """Base for VOC dataset.

    All other classes which extend this should implement
    a __getitem__ method
    """

    def __init__(self, annotations=Path('VOC2007/Annotations'), mask=None, normalizer=None,
                 device=torch.device("cpu"), label2class=None):
        annotations_files = np.array(sorted([x for x in annotations.iterdir()]))
        if mask is not None:
            annotations_files = annotations_files[mask]
        self.annotations_files = annotations_files

        self.device = device
        self.normalizing_dict = {
            'imagenet': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
            'inception': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
        }
        if normalizer:
            assert normalizer in self.normalizing_dict, "Normalizer not one of {}". \
                format([val for val in self.normalizing_dict.keys()])
        self.normalizer = normalizer

        objects = [im['objects'] for im in [xml_to_dict(ano) for ano in self.annotations_files]]
        label_names = set([n for l in [[d['name'] for idx, d in i.items()] for i in objects] for n in l])
        self.label_names = label_names
        # turn the labels into integer classes
        if not label2class:
            self.label2class = {val: idx for idx, val in enumerate(label_names)}
        else:
            self.label2class = label2class

    def __len__(self):
        return len(self.annotations_files)

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


class SingleImageDataset(VOCDataset, Dataset):
    """
    Given a path to annotations in a standard VOC dataset, return a Dataset object
    which yields the images, and the bounding box and label for the largest object
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

        super().__init__(annotations=annotations, mask=mask, normalizer=normalizer,
                         device=device, label2class=label2class)

        self.resize = resize
        self.random_transform = random_transform

        largest_items = [keep_largest_box(xml_to_dict(ano))['objects'][0]
                         for ano in self.annotations_files]

        # first, the bounding boxes
        bounding_boxes = [item['coordinates'] for item in largest_items]
        self.bounding_boxes = np.asarray(bounding_boxes)

        label_names = [item['name'] for item in largest_items]
        label_classes = [self.label2class[item] for item in label_names]
        self.labels = torch.tensor(np.asarray(label_classes), dtype=torch.long,
                                   device=self.device)

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
            colour_jitter,
            rotate,
        ]

        chosen_function = random.choice(transforms)
        image, bounding_box = chosen_function(image, [bounding_box])

        # ensure min and max are correct
        xmin, ymin, xmax, ymax = bounding_box[0]
        bounding_box = [min([xmin, xmax]),
                        min([ymin, ymax]),
                        max([xmin, xmax]),
                        max([ymin, ymax])]

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


class MultiImageDataset(VOCDataset):
    """
        Given a path to annotations in a standard VOC dataset, return a Dataset object
        which yields the images, and the bounding box and label for the largest object
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

        super().__init__(annotations=annotations, mask=mask, normalizer=normalizer,
                         device=device, label2class=label2class)

        self.resize = resize
        self.random_transform = random_transform

        # now, lets get the bounding boxes and labels, which will be lists
        objects = [xml_to_dict(image)['objects'] for image in self.annotations_files]

        bbs = []
        labels = []
        for images in objects:
            # this makes sure the order is right
            single_image_bbs = []
            single_image_labels = []

            for idx, object in images.items():
                single_image_bbs.append(object['coordinates'])
                single_image_labels.append(self.label2class[object['name']])
            bbs.append(single_image_bbs)
            labels.append(single_image_labels)
        self.bounding_boxes = bbs
        self.labels = labels

    def _resize(self, image, bounding_boxes):
        if self.resize:
            width, height = self.resize
            bbs_to_return = []
            for bounding_box in bounding_boxes:
                xmin, ymin, xmax, ymax = bounding_box

                x_ratio = width / image.shape[1]
                y_ratio = height / image.shape[0]

                bounding_box = [xmin * x_ratio, ymin * y_ratio,
                                xmax * x_ratio, ymax * y_ratio]

                bbs_to_return.append(bounding_box)

            image = cv2.resize(image, (width, height),
                               interpolation=cv2.INTER_AREA)

        return image, bbs_to_return

    def _transform_image(self, image, bounding_boxes):

        transforms = [
            no_change,
            horizontal_flip,
            colour_jitter,
            rotate,
        ]

        chosen_function = random.choice(transforms)
        image, bounding_boxes = chosen_function(image, bounding_boxes)

        # ensure min and max are correct
        output_bb = []
        for bb in bounding_boxes:
            xmin, ymin, xmax, ymax = bb
            bounding_box = [min([xmin, xmax]),
                            min([ymin, ymax]),
                            max([xmin, xmax]),
                            max([ymin, ymax])]
            output_bb.append(bounding_box)
        return image, output_bb

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
               torch.tensor(self.labels[index], dtype=torch.long, device=self.device)


def collate_im_bb_lab(batch):
    # this allows us to handle the fact that the bbs and
    # labels will be of different sizes
    ims = torch.stack([item[0] for item in batch], 0)
    bbs = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return ims, bbs, labels
