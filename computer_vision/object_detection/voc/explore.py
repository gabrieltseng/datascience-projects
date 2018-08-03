"""
Some methods which make the VOC dataset easier to explore
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random


from pathlib import Path


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
