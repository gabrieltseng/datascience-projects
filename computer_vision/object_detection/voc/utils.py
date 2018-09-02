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
    if image.max() > 1:
        image = image / 255
    if mean and std:
        image = (image - mean) / std
    # in addition, roll the axis so that they suit pytorch
    return image.swapaxes(2, 0).swapaxes(1, 2)


def denormalize(image, mean=None, std=None):
    """
    Given a normalized image, and the statistics according to which it was normalized,
    denormalize it
    """
    image = image.swapaxes(1, 2).swapaxes(0, 2)
    if mean and std:
        image = (image * std) + mean
    image *= 255
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


def activations_to_ratios(bb, anchors):
    """
    We will define the activations from the network as factors, which can be used
    to slightly shift the bounding boxes. This is motivated by the SSD paper
    """
    bb = torch.tanh(bb)
    # get some info about the anchors
    width = anchors[:, 2] - anchors[:, 0]
    height = anchors[:, 3] - anchors[:, 1]
    x_center, y_center = (anchors[:, 0] + (width / 2)), (anchors[:, 1] + (height / 2))

    # bb[:, 2:] will be used to determine shifts in the height and widths of the bounding box
    activated_width = width * torch.exp(bb[:, 0])
    activated_height = height * torch.exp(bb[:, 1])
    # bb[:, :2] will be used to determine shifts in the center of the bounding box
    activated_x_center = x_center + (bb[:, 2] * width)
    activated_y_center = y_center + (bb[:, 3] * height)

    xmin = activated_x_center - (activated_width / 2)
    ymin = activated_y_center - (activated_height / 2)
    xmax = activated_x_center + (activated_width / 2)
    ymax = activated_y_center + (activated_height / 2)
    return torch.stack((xmin, ymin, xmax, ymax), 1)


def nms(boxes, scores, overlap=0.5, top_k=100):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0: return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        keep[count] = i
        count += 1
        if idx.size(0) == 1: break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count