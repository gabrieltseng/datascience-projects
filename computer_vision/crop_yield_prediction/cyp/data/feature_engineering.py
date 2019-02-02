from pathlib import Path
import numpy as np
import pandas as pd
import math

from .utils import load_clean_yield_data as load


class Engineer:
    """
    Take the preprocessed data from the Data Cleaner
    and turn the images into matrices which can be input
    into the machine learning models.

    These matrices can either be histograms, which describe the distributions of
    pixels on each band, and contain 32 bins.
    This turns the band of an image from dim=(width*height) to dim=32.

    They can also be means of each band, which turns the band of an image from
    dim=(width*height) into a scalar value.
    """
    def __init__(self, cleaned_data_path=Path('data/img_output'),
                 yield_data_filepath=Path('data/yield_data.csv'),
                 county_data_filepath=Path('data/county_data.csv')):
        self.cleaned_data = cleaned_data_path
        self.processed_files = self.get_filenames()

        # merge the yield and county data for easier manipulation
        yield_data = load(yield_data_filepath)[['Year', 'State ANSI', 'County ANSI', 'Value']]
        yield_data.columns = ['Year', 'State', 'County', 'Value']
        county_data = pd.read_csv(county_data_filepath)[['CntyFips', 'StateFips',
                                                         'Longitude', 'Latitude']]
        county_data.columns = ['County', 'State', 'Longitude', 'Latitude']
        self.yield_data = yield_data.merge(county_data, how='left', on=['County', 'State'])

    def get_filenames(self):
        """
        Get all the .tif files in the image folder.
        """
        files = []
        for dir_file in Path(self.cleaned_data).iterdir():
            if str(dir_file).endswith('npy'):

                # strip out the directory so its just the filename
                files.append(str(dir_file.parts[-1]))
        return files

    @staticmethod
    def filter_timespan(imcol, start_day=49, end_day=305, composite_period=8, bands=9):
        """
        Author: Jiaxuan You, https://github.com/JiaxuanYou

        Given an image collection containing a year's worth of data,
        filter it between start_day and end_day. If end_day is later than the date
        for which we have data, the image collection is padded with zeros.

        Parameters
        ----------
        imcol: The image collection to be filtered
        start_day: int, default=49
            The earliest day for which to consider data
        end_day: int, default=305
            The last day for which to consider data
        composite_period: int, default=8
            The composite period of the images. Default taken from the composite
            periods of the MOD09A1 and MYD11A2 datasets
        bands: int, default=9
            The number of bands per image. Default taken from the number of bands in the
            MOD09A1 + the number of bands in the MYD11A2 datasets

        Returns
        ----------
        A filtered image collection
        """
        start_index = int(math.floor(start_day / composite_period)) * bands
        end_index = int(math.floor(end_day / composite_period)) * bands

        if end_index > imcol.shape[2]:
            padding = np.zeros((imcol.shape[0], imcol.shape[1], end_index - imcol.shape[2]))
            imcol = np.concatenate((imcol, padding), axis=2)
        return imcol[:, :, start_index: end_index]

    def process(self, num_bands=9, generate='mean', num_bins=32, ):
        """
        Parameters
        ----------
        num_bands: int, default=9
            The number of bands per image. Default taken from the number of bands in the
            MOD09A1 + the number of bands in the MYD11A2 datasets
        generate: str, {'mean', 'histogram'}, default='mean'
            What to generate from the data. If 'mean', calculates a mean
            of all the bands. If 'histogram', calculates a histogram of all
            the bands with num_bins bins for each band.
        num_bins: int, default=32
            If generate=='histogram', the number of bins to generate in the histogram.
        """

        # define all the outputs of this method
        output_images = []
        yields = []
        years = []
        locations = []
        state_county_info = []

        for yield_data in self.yield_data.itertuples():
            year = yield_data.Year
            county = yield_data.County
            state = yield_data.State

            filename = f'{year}_{int(state)}_{int(county)}.npy'
            if filename in self.processed_files:
                image = np.load(self.cleaned_data / filename)
                image = self.filter_timespan(image, start_day=49, end_day=305,
                                             bands=num_bands)

                if generate == 'mean':
                    image = np.sum(image, axis=(0, 1)) / np.count_nonzero(image) * image.shape[2]
                    image[np.isnan(image)] = 0
                elif generate == 'histogram':
                    raise NotImplementedError
                output_images.append(image)
                yields.append(yield_data.Value)
                years.append(year)
                locations.append(np.array([yield_data.Longitude, yield_data.Latitude]))
                state_county_info.append(np.array([county, state]))

                print(f'County: {county}, State: {state}, Year: {year}, Sum of mean: {np.sum(image)}')

        np.savez(self.cleaned_data / 'histogram_all_mean.npz',
                 output_image=np.stack(output_images), output_yield=np.array(yields),
                 output_year=np.array(years), output_locations=np.stack(locations),
                 output_index=np.stack(state_county_info))
        print (f'Finished generating image {generate}s!')
