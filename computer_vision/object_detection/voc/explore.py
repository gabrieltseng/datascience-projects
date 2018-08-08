"""
Some methods which make the VOC dataset easier to explore
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random


from pathlib import Path
from .utils import load_image, xml_to_dict, activations_to_ratios, nms


def plot_image(image, bounding_boxes, labels, ax=None):
    show_plot = False
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 10))
        show_plot = True
    ax.imshow(image)

    for bounding_box, label in zip(bounding_boxes, labels):
        xmin, ymin, xmax, ymax = bounding_box
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin),
                                 width, height,
                                 linewidth=1, edgecolor='xkcd:bright green',
                                 facecolor='none', label=label)
        ax.add_patch(rect)
        ax.text(xmin, ymin, label, color='xkcd:bright green', fontsize=12)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if show_plot:
        plt.show()


def show_annotation(annotation):
    """
    Given a dictionary of the annotation,
    plots the corresponding image with the bounding boxes
    """
    # first, load up the image
    image_path = annotation['image_path']
    image_to_plot = load_image(image_path)

    # then, all the bounding boxes and labels
    bounding_boxes = []
    labels = []
    for anno_id, anno in annotation['objects'].items():
        bounding_boxes.append(anno['coordinates'])
        labels.append(anno['name'])
    plot_image(image_to_plot, bounding_boxes, labels)

    plt.show()


def show_random_example(VOC_path=Path('VOC2007'), return_annotation=False):
    """
    Given the VOC directory, plot a random example
    """
    annotations = VOC_path / 'Annotations'
    images = VOC_path / 'JPEGImages'

    random_annotation = random.choice([x for x in annotations.iterdir()])
    print("Showing {}".format(str(random_annotation)))
    annotation = xml_to_dict(random_annotation, image_folder_path=images)
    show_annotation(annotation)
    if return_annotation:
        return annotation


def plot_multiobject_results(im, bb, lab, anchors, label2class, use_nms=True,
                             threshold=0.25, ax=None):
    """Plot the results of a prediction
    """
    if not ax: fig, ax = plt.subplots(figsize=(10, 10))
    class2label = {int(im_class): label for label, im_class in label2class.items()}

    # first, find the bb coordinates given the anchors
    coords = activations_to_ratios(bb.view(-1, 4), anchors) * 224

    # next, find the objects for which the item was more than threshold confident
    num_classes = len(label2class)
    lab = torch.nn.functional.sigmoid(lab.view(-1, (num_classes + 1)))[:, :-1]
    selected_anchors = torch.nonzero(torch.sum(lab > threshold, dim=1))
    if selected_anchors.shape[0] == 0:
        ax.imshow(im)
    else:
        selected_anchors = selected_anchors.squeeze(1)
        selected_labels = lab[selected_anchors]
        selected_label_values, selected_label_idxs = torch.max(selected_labels, dim=1)
        # now, we can use nms to find the boxes we are going to keep
        selected_coords = coords[selected_anchors]
        if use_nms:
            output_boxes, count = nms(selected_coords, selected_label_values)
            output_boxes = output_boxes[:count]

            nms_selected_boxes = selected_coords[output_boxes].detach().cpu().numpy()
            nms_selected_labels = selected_label_idxs[output_boxes]
            nms_selected_label_scores = selected_label_values[output_boxes]
            label_names = ['{}: {}'.format(class2label[idx.item()], round(score.item(), 2))
                           for idx, score in zip(nms_selected_labels, nms_selected_label_scores)]

            plot_image(im, nms_selected_boxes, label_names, ax=ax)
        else:
            label_names = ['{}: {}'.format(class2label[idx.item()], round(score.item(), 2))
                           for idx, score in zip(selected_label_idxs, selected_label_values)]
            plot_image(im, selected_coords.detach().cpu().numpy(), label_names, ax=ax)
