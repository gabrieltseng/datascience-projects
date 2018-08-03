import torch
import cv2
import numpy as np
import itertools
from pathlib import Path
import xml.etree.ElementTree as ET
import copy


def load_image(image_path):
    """
    opencv returns the image in blue green red, not red green blue
    This function corrects for this
    """
    image_array = cv2.imread(image_path.as_posix())

    return cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)


def xml_to_dict(file_path, image_folder_path=Path('VOC2007/JPEGImages')):
    """
    Given a VOC xml annotation, turn it into a dictionary
    we can manipulate
    """
    tree = ET.parse(file_path)  # create an ElementTree object
    root = tree.getroot()
    object_id = 0

    # first, get the image filename
    image_filename = root.find('filename').text
    xml_dict = {'image_path': image_folder_path/image_filename}

    # Then, get all the bounding box coordinates
    xml_dict['objects'] = {}
    for child in root:
        if child.tag == 'object':
            object_name = child.find('name').text
            bounding_box = child.find('bndbox')

            xml_dict['objects'][object_id] = {'name': object_name,
                                              'coordinates':
                                                  [int(bounding_box.find('xmin').text),
                                                   int(bounding_box.find('ymin').text),
                                                   int(bounding_box.find('xmax').text),
                                                   int(bounding_box.find('ymax').text)]}
            object_id += 1
    return xml_dict


def box_size(bounding_box):
    xmin, ymin, xmax, ymax = bounding_box
    return (xmax - xmin) * (ymax - ymin)


def keep_largest_box(annotation):
    """
    Given an annotation, keep only the largest bounding box
    """
    return_annotation = copy.deepcopy(annotation)
    largest_item_idx = max(annotation['objects'],
                           key=(lambda key: box_size(annotation['objects'][key]['coordinates'])))

    largest_item = annotation['objects'][largest_item_idx]

    return_annotation['objects'] = {0: largest_item}
    return return_annotation


def normalize(image, mean=None, std=None):
    """
    Given an image, and a dataset mean and standard deviation,
    normalize the image according to the dataset means and standard deviations

    If no mean and std are passed, this method will just swap the image axes so they
    are suitable for pytorch
    """
    if mean and std:
        image = (image - mean) / std
    # in addition, roll the axis so that they suit pytorch
    return image.swapaxes(2, 0)


def denormalize(image, mean=None, std=None):
    """
    Given a normalized image, and the statistics according to which it was normalized,
    denormalize it
    """
    image = image.swapaxes(0, 2)
    if mean and std:
        image = (image * std) + mean
    return image.astype(int)


def make_anchors(image_height, image_width, anchor_size):
    # anchor size is a fraction of the image height and width
    if isinstance(anchor_size, tuple):
        x_ratio, y_ratio = anchor_size
    else:
        x_ratio = y_ratio = anchor_size

    x_size = image_width * x_ratio
    y_size = image_height * y_ratio
    # first, lets define the top left corner of all the anchor boxes:
    xs = np.arange(0, image_width, x_size)
    ys = np.arange(0, image_height, y_size)
    # get all possible coordinates
    top_right_xy = [list(zip(xs, zip_xy)) for zip_xy in itertools.permutations(xs, len(ys))]

    # and flatten it
    top_right_xy = [coords for sublist in top_right_xy for coords in sublist]

    # the set ensures uniqueness
    anchor_boxes = set([tuple((xmin, ymin, xmin + x_size, ymin + y_size)) for xmin, ymin in top_right_xy])
    return [list(anchor) for anchor in anchor_boxes]


def permute_anchors(anchors, zooms=None, ratios=None, image_dims=(224, 224)):
    """Given a list of anchor boxes, return a list of anchor boxes with the zooms and ratios applies.

    Note that the output is a 3D array, so that each bounding box can be associated with its base bounding box
    """
    num_permutations = len(zooms) * len(ratios)
    output_bbs = []
    for xmin, ymin, xmax, ymax in anchors:
        anchor_width = xmax - xmin
        anchor_height = ymax - ymin
        center = (xmin + (anchor_width / 2), ymin + (anchor_height / 2))
        for zoom in zooms:
            for x_ratio, y_ratio in ratios:
                # ensure the bounding boxes don't go outside of the image
                xmin = max(0, center[0] - (zoom * x_ratio * anchor_width / 2))
                xmax = min(image_dims[0], center[0] + (zoom * x_ratio * anchor_width / 2))
                ymin = max(0, center[1] - (zoom * y_ratio * anchor_width / 2))
                ymax = min(image_dims[1], center[1] + (zoom * y_ratio * anchor_width / 2))
                output_bbs.append([xmin, ymin, xmax, ymax])
    return np.array(output_bbs), num_permutations


def extend_tensor(shape, constant, dtype=torch.float64):
    if isinstance(constant, torch.Tensor):
        constant = constant.item()
    return torch.ones(shape, dtype=dtype) * constant


def anchor_to_class(anchor_val, anchor_idx, labels, threshold=0.5, background_index=21):
    selected_anchors = anchor_val > threshold
    background = torch.nonzero(1-selected_anchors)[:, 0]

    anchor_classes = labels[anchor_idx]
    anchor_classes[background] = background_index
    return anchor_classes


def jaccard_to_anchor(jaccard_overlaps):
    """
    To find which anchor box is associated to an object,
    we need to consider
    - which is the best anchor box for each object
    - which is the best object for each anchor box
    """
    # first, find the best anchor for the bounding box
    bbox_val, bbox_idx = jaccard_overlaps.max(0)
    # then, the best bounding box for each anchor
    anchor_val, anchor_idx = jaccard_overlaps.max(1)

    # next, lets force the best bbox per anchor VALUE (not idx)
    # to be high for the best anchor
    anchor_val[bbox_idx] = 1.99
    for i, o in enumerate(bbox_idx):
        anchor_idx[o]: i
    # anchor_idx: best bbox for each anchor (forced to the bbox
    # for the best anchor for that bbox)
    # anchor_val: jaccard overlap of that bbox (forced to 2 for the
    # best anchors for each bbox)
    return anchor_val, anchor_idx


def bbox_to_jaccard(anchors, bbox):
    """
    Combines jaccard_index and find_anchor for tensors.
    """
    # to begin with, we have to make the two tensors broadcastable
    anchors = anchors.unsqueeze(1)
    bbox = bbox.unsqueeze(0)

    axmin, aymin, axmax, aymax = anchors[:, :, 0], anchors[:, :, 1], anchors[:, :, 2], anchors[:, :, 3]
    bxmin, bymin, bxmax, bymax = bbox[:, :, 0], bbox[:, :, 1], bbox[:, :, 2], bbox[:, :, 3]

    b_area = ((bxmax - bxmin) * (bymax - bymin)).unsqueeze(0)
    a_area = ((axmax - axmin) * (aymax - aymin)).unsqueeze(1)
    a_plus_b = (b_area + a_area)[:, 0, :]

    overlap_width = torch.clamp(torch.min(axmax, bxmax) - torch.max(bxmin, axmin), 0)
    overlap_height = torch.clamp(torch.min(aymax, bymax) - torch.max(aymin, bymin), 0)
    overlaps = overlap_height * overlap_width

    union = a_plus_b - overlaps
    jaccard = overlaps / union
    return jaccard