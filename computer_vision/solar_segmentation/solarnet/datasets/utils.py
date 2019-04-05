import numpy as np

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def normalize(image):
    """Normalized an image (or a set of images), as per
    https://pytorch.org/docs/1.0.0/torchvision/models.html

    Specifically, images are normalized to range [0, 1], and
    then normalized according to ImageNet stats.
    """
    image = image / 255

    # determine if we are dealing with a single image, or a
    # stack of images. If a stack, expected in (batch, channels, height, width)
    source, dest = 0 if len(image.shape) == 3 else 1, -1

    # moveaxis for array broadcasting, and then back so its how pytorch expects it
    return np.moveaxis((np.moveaxis(image, source, dest) - MEAN) * STD, dest, source)


def denormalize(image):
    """Reverses what normalize does
    """
    # determine if we are dealing with a single image, or a
    # stack of images. If a stack, expected in (batch, channels, height, width)
    source, dest = 0 if len(image.shape) == 3 else 1, -1

    image = np.moveaxis((np.moveaxis(image, source, dest) / STD) + MEAN, dest, source)
    return (image * 255).astype(int)
