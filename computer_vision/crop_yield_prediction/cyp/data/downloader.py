import ee
import time
import pandas as pd
from pathlib import Path


class MODISExporter:
    """ A base class to export MODIS data from
    the Google Earth Engine to Google Drive

    Parameters
    ----------

    locations_filepath: pathlib Path, default=Path('data/locations_final.csv')
        A path to the locations being pulled
    collection_id: str, default='MODIS/051/MCD12Q1'
        The ID Earth Engine Image Collection being exported
    """
    def __init__(self, locations_filepath=Path('data/locations_final.csv'),
                 collection_id='MODIS/051/MCD12Q1'):
        self.locations = pd.read_csv(locations_filepath)
        self.collection_id = collection_id

        try:
            ee.Initialize()
            print('The Earth Engine package initialized successfully!')
        except ee.EEException:
            print('The Earth Engine package failed to initialize! '
                  'Have you authenticated the earth engine?')

    @staticmethod
    def _export_one_image(img, folder, name, region, scale, crs):
        # export one image from Earth Engine to Google Drive
        # Author: Jiaxuan You, https://github.com/JiaxuanYou
        task = ee.batch.Export.image(img, name, {
            'driveFolder': folder,
            'driveFileNamePrefix': name,
            'region': region,
            'scale': scale,
            'crs': crs
        })
        task.start()
        while task.status()['state'] == 'RUNNING':
            print('Running...')
            # Perhaps task.cancel() at some point.
            time.sleep(10)

        print(f'Done: {task.staus()}')

    @staticmethod
    def _append_band(current, previous):
        # Transforms an Image Collection with 1 band per Image into a single Image with items as bands
        # Author: Jamie Vleeshouwer

        # Rename the band
        previous = ee.Image(previous)
        current = current.select([0, 1, 2, 3, 4, 5, 6])
        # Append it to the result (Note: only return current item on first element/iteration)
        return ee.Algorithms.If(ee.Algorithms.IsEqual(previous, None), current, previous.addBands(ee.Image(current)))

    def export(self, coordinate_system='EPSG:4326', scale=500, region='county', offset=0.11,
               datefilter=None):
        """Export an Image Collection from Earth Engine to Google Drive

        Parameters
        ----------
            coordinate_system: str, default='EPSG:4326'
                The coordinate system in which to export the data
            scale: int, default=500
                The pixel resolution, as determined by the output.
                https://developers.google.com/earth-engine/scale
            region: str, 'county' or 'offset', default='county'
                One of 'county' or 'offset'; the technique used to
                define the region to be taken from the original data.
            offset: float, default=0.11
                If region == 'offset', the offset value to apply to the
                latitudes and longitudes.
            datefilter: None, or 'default', or tuple of str dates
                Dates within which to export the data
        """
        if datefilter == 'default':
            datefilter = ('2002-12-31', '2016-8-4')

        imgcoll = ee.ImageCollection(self.collection_id) \
            .filterBounds(ee.Geometry.Rectangle(-106.5, 50, -64, 23))
        if datefilter:
            imgcoll.filterDate(datefilter[0], datefilter[1])
        img = imgcoll.iterate(self._append_band)

        # note that the county regions are pulled from Google's Fusion tables. This calls a merge
        # of county geometry and census data:
        # https://fusiontables.google.com/data?docid=1S4EB6319wWW2sWQDPhDvmSBIVrD3iEmCLYB7nMM#rows:id=1
        if region == 'county':
            county_region = ee.FeatureCollection('ft:1S4EB6319wWW2sWQDPhDvmSBIVrD3iEmCLYB7nMM')

        for state_id, county_id, lat, lon in self.locations.values:
            fname = '{}_{}'.format(int(state_id), int(county_id))

            # filter for a county
            if region == 'county':
                file_region = county_region.filterMetadata('StateFips', 'equals', int(state_id))
                file_region = ee.FeatureCollection(file_region).filterMetadata('CntyFips', 'equals', int(county_id))
                file_region = ee.Feature(file_region.first())
            else:
                file_region = str([
                    [lat - offset, lon + offset],
                    [lat + offset, lon + offset],
                    [lat + offset, lon - offset],
                    [lat - offset, lon - offset]])

            while True:
                try:
                    self._export_one_image(img, 'Data_mask', fname, file_region, scale, coordinate_system)
                except:
                    print
                    'retry'
                    time.sleep(10)
                    continue
                break
        print('Finished Exporting!')
