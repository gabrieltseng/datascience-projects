import rasterio
import numpy as np
import matplotlib.pyplot as plt


def visualize(path_to_tif):
    """Visualize a downloaded file.

    Takes the red, green and blue bands to plot a
    'colour image' of a downloaded tif file.

    Note that this is not a true colour image, since
    this is a complex thing to represents, and instead is a 'basic
    true colour scheme'
    http://www.hdfeos.org/forums/showthread.php?t=736

    Parameters
    ----------
    path_to_tif: str
        string path to the tif file
    """
    data = rasterio.open(path_to_tif)

    arr_red = data.read(1)
    arr_green = data.read(4)
    arr_blue = data.read(3)

    im = np.dstack((arr_red, arr_green, arr_blue))

    im_norm = im / im.max()

    plt.imshow(im_norm)
