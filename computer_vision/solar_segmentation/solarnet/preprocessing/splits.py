import numpy as np
from numpy.random import randint
import pandas as pd
from collections import defaultdict
from pathlib import Path
import rasterio
from tqdm import tqdm


class ImageSplitter:
    """Solar panels cover a relatively small landmass compared to
    the total image sizes, so to make sure there are contiguous solar
    panels in all segments, we will center each image on the solar panel.
    For every image, we will also randomly sample (at a buffer away from the
    solar panels) a number of images which don't contain any solar panels
    to pretrain the network
    """

    def __init__(self, data_folder=Path('data')):
        self.data_folder = data_folder

        # setup; make the necessary folders
        self.processed_folder = data_folder / 'processed'
        if not self.processed_folder.exists(): self.processed_folder.mkdir()

        self.solar_panels = self._setup_folder('solar')
        self.empty = self._setup_folder('empty')

    def _setup_folder(self, folder_name):

        folder_name = self.processed_folder / folder_name
        if not folder_name.exists(): folder_name.mkdir()

        for subfolder in ['org', 'mask']:
            subfolder_name = folder_name / subfolder
            if not subfolder_name.exists(): subfolder_name.mkdir()

        return folder_name

    def read_centroids(self):

        metadata = pd.read_csv(self.data_folder / 'metadata/polygonDataExceptVertices.csv',
                               usecols=['city', 'image_name', 'centroid_latitude_pixels',
                                        'centroid_longitude_pixels'])

        # for each image, we want to know where the solar panel centroids are
        output_dict = defaultdict(lambda: defaultdict(list))

        for idx, row in metadata.iterrows():
            output_dict[row.city][row.image_name].append((
                row.centroid_latitude_pixels, row.centroid_longitude_pixels
            ))
        return output_dict

    @staticmethod
    def adjust_coords(coords, image_radius, org_imsize):
        x, y = coords
        # we make sure that the centroid isn't at the edge of the image
        if x < image_radius: x = image_radius
        elif x > (org_imsize - image_radius): x = org_imsize - image_radius

        if y < image_radius: y = image_radius
        elif y > (org_imsize - image_radius): y = org_imsize - image_radius

        return x, y

    def process(self, imsize=224, empty_ratio=2, org_imsize=5000):

        im_rad = imsize // 2
        centroids_dict = self.read_centroids()

        im_idx = 0
        for city, images in centroids_dict.items():
            print(f"Processing {city}")

            for image_name, centroids in tqdm(images.items()):

                org_file = rasterio.open(self.data_folder / f"{city}/{image_name}.tif").read()
                mask_file = np.load(self.data_folder / f"{city}_masks/{image_name}.npy")

                # first, lets collect the positive examples
                for centroid in centroids:
                    x, y = self.adjust_coords(centroid, im_rad, org_imsize)

                    max_width, max_height = int(x + im_rad), int(y + im_rad)
                    min_width, min_height = max_width - imsize, max_height - imsize

                    np.save(self.solar_panels / f"org/{im_idx}.npy",
                            org_file[:, min_width: max_width, min_height: max_height])
                    np.save(self.solar_panels / f"mask/{im_idx}.npy",
                            mask_file[min_width: max_width, min_height: max_height])

                    im_idx += 1

                # next, the negative examples
                # this is a crude way of doing it, and could probably be improved
                patience, max_patience = 0, 3
                num_empty, max_num_empty = 0, len(centroids) * empty_ratio
                while (patience < max_patience) and (num_empty < max_num_empty):
                    rand_x, rand_y = randint(0, org_imsize - imsize), randint(0, org_imsize - imsize)

                    rand_x_max, rand_y_max = rand_x + imsize, rand_y + imsize
                    # this makes sure no solar panel is present
                    mask_candidate = mask_file[rand_x: rand_x_max, rand_y: rand_y_max]

                    if mask_candidate.sum() == 0:
                        np.save(self.empty / f"org/{im_idx}.npy",
                                org_file[:, rand_x: rand_x_max, rand_y: rand_y_max])
                        np.save(self.empty / f"mask/{im_idx}.npy",
                                mask_candidate)
                        im_idx += 1
                        num_empty += 1
                    else:
                        patience += 1
        print(f"Generated {im_idx} samples")
