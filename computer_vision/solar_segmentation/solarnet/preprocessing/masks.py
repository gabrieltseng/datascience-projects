import pandas as pd
import numpy as np
from matplotlib.path import Path as PolygonPath
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


class MaskMaker:
    """This class looks for all folders defined in FILE2CITY, and
    produces masks for all of the .tif files saved there.
    These files will be saved in <org_folder>_mask/<org_filename>.npy
    """

    def __init__(self, data_folder=Path('data')):
        self.data_folder = data_folder

    def _read_data(self):
        metadata_folder = self.data_folder / 'metadata'

        polygon_pixels = self._csv_to_dict_polygon_pixels(
            pd.read_csv(metadata_folder / 'polygonVertices_PixelCoordinates.csv')
        )
        # TODO: potentially filter on jaccard index
        polygon_images = self._csv_to_dict_image_names(
            pd.read_csv(metadata_folder / 'polygonDataExceptVertices.csv',
                        usecols=['polygon_id', 'city', 'image_name', 'jaccard_index']
                        )
        )
        return polygon_images, polygon_pixels

    def process(self):

        polygon_images, polygon_pixels = self._read_data()

        for city, files in polygon_images.items():
            print(f'Processing {city}')
            # first, we make sure the mask file exists; if not,
            # we make it
            masked_city = self.data_folder / f"{city}_masks"
            if not masked_city.exists(): masked_city.mkdir()

            for image, polygons in tqdm(files.items()):
                masks = []
                for polygon in polygons:
                    masks.append(self.make_mask(polygon_pixels[polygon]))

                np.save(masked_city / f"{image}.npy", np.sum(masks, axis=0))

    @staticmethod
    def _csv_to_dict_polygon_pixels(polygon_pixels):
        output_dict = {}

        for idx, row in polygon_pixels.iterrows():
            vertices = []
            for i in range(1, int(row.number_vertices) + 1):
                vertices.append((row[f"lat{i}"], row[f"lon{i}"]))
            output_dict[int(row.polygon_id)] = vertices
        return output_dict

    @staticmethod
    def _csv_to_dict_image_names(polygon_images):
        output_dict = defaultdict(lambda: defaultdict(list))

        for idx, row in polygon_images.iterrows():
                output_dict[row.city][row.image_name].append(int(row.polygon_id))
        return output_dict

    @staticmethod
    def make_mask(coords, imsize=5000):
        """https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask
        """
        poly_path = PolygonPath(coords)

        x, y = np.mgrid[:imsize, :imsize]
        coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        mask = poly_path.contains_points(coors)

        return mask.reshape(imsize, imsize).astype(float)
