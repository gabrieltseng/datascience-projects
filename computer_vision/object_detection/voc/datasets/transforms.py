"""
Image transformations, along with their corresponding
bounding box transformations.
"""

import cv2
import numpy as np


def no_change(image, bb):
    return image, bb


def horizontal_flip(image, bb):
    image = cv2.flip(image, 1)
    xmin, ymin, xmax, ymax = bb
    width = image.shape[1]

    # note that xmax becomes xmin,
    # and vice versa
    xmin_flip = width - xmax
    xmax_flip = width - xmin

    bb = [xmin_flip, ymin,
          xmax_flip, ymax]

    return image, bb


def vertical_flip(image, bb):
    image = cv2.flip(image, 0)
    xmin, ymin, xmax, ymax = bb
    height = image.shape[0]

    # note that ymax becomes ymin,
    # and vice versa
    ymin_flip = height - ymax
    ymax_flip = height - ymin

    bb = [xmin, ymin_flip,
          xmax, ymax_flip]

    return image, bb

def colour_jitter(image, bb):
    height, width, _ = image.shape
    zitter = np.zeros_like(image)

    for channel in range(zitter.shape[2]):
        noise = np.random.randint(0, 30, (height, width))
        zitter[:, :, channel] = noise

    image = cv2.add(image, zitter)
    return image, bb
