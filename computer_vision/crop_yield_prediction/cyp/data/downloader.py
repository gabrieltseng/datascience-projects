import ee
import time
import pandas as pd
from pathlib import Path


class MODISExporter:
    """ A class to export MODIS data from
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
        self.locations = pd.read_csv(locations_filepath, header=None)
        self.collection_id = collection_id

        try:
            ee.Initialize()
            print('The Earth Engine package initialized successfully!')
        except ee.EEException:
            print('The Earth Engine package failed to initialize! '
                  'Have you authenticated the earth engine?')

    def update_parameters(self, locations_filepath=None, collection_id=None):
        """
        Update the locations file or the collection id
        """
        if locations_filepath is not None:
            self.locations = pd.read_csv(locations_filepath, header=None)
        if collection_id is not None:
            self.collection_id = collection_id

    @staticmethod
    def _export_one_image(img, folder, name, region, scale, crs):
        # export one image from Earth Engine to Google Drive
        # Author: Jiaxuan You, https://github.com/JiaxuanYou
        print(f'Exporting to {folder}/{name}')
        task_dict = {
            'driveFolder': folder,
            'driveFileNamePrefix': name,
            'scale': scale,
            'crs': crs
        }
        if region is not None:
            task_dict.update({
                'region': region
            })
        task = ee.batch.Export.image(img, name, task_dict)
        task.start()
        while task.status()['state'] == 'RUNNING':
            print('Running...')
            # Perhaps task.cancel() at some point.
            time.sleep(10)

        print(f'Done: {task.status()}')

    @staticmethod
    def _append_band(current, previous):
        # Transforms an Image Collection with 1 band per Image into a single Image with items as bands
        # Author: Jamie Vleeshouwer

        # Rename the band
        previous = ee.Image(previous)
        current = current.select([0, 1, 2, 3, 4, 5, 6])
        # Append it to the result (Note: only return current item on first element/iteration)
        return ee.Algorithms.If(ee.Algorithms.IsEqual(previous, None), current, previous.addBands(ee.Image(current)))

    def export(self, folder_name, coordinate_system='EPSG:4326', scale=500, region_type='uscounties', offset=0.11,
               datefilter=None, export_limit=None, min_img_val=None, max_img_val=None):
        """Export an Image Collection from Earth Engine to Google Drive

        Parameters
        ----------
            folder_name: str
                The name of the folder to export the images to in
                Google Drive. If the folder is not there, this process
                creates it
            coordinate_system: str, default='EPSG:4326'
                The coordinate system in which to export the data
            scale: int, default=500
                The pixel resolution, as determined by the output.
                https://developers.google.com/earth-engine/scale
            region_type: str, 'uscounties', 'countygeometries', or 'square', default='county'
                The technique used to define the region to be taken
                from the original data.
                'uscounties', 'countygeometries' and 'world' define two different function tables to
                call the regions from.
            offset: float, default=0.11
                If region == 'square', the offset value to apply to the
                latitudes and longitudes.
            datefilter: None, or 'default', or tuple of str dates
                Dates within which to export the data
            export_limit: int or None, default=None
                If not none, limits the number of files exported to the value
                passed.
            min_img_val = int or None:
                A minimum value to clip the band values to
            max_img_val: int or None
                A maximum value to clip the band values to
        """
        if datefilter == 'default':
            datefilter = ('2002-12-31', '2016-8-4')

        imgcoll = ee.ImageCollection(self.collection_id) \
            .filterBounds(ee.Geometry.Rectangle(-106.5, 50, -64, 23))
        if datefilter:
            imgcoll.filterDate(datefilter[0], datefilter[1])
        img = imgcoll.iterate(self._append_band)
        img = ee.Image(img)

        # "clip" the values of the bands
        if min_img_val is not None:
            # passing en ee.Number creates a constant image
            img_min = ee.Image(ee.Number(min_img_val))
            img = img.min(img_min)
        if max_img_val is not None:
            img_max = ee.Image(ee.Number(max_img_val))
            img = img.max(img_max)

        # note that the county regions are pulled from Google's Fusion tables. This calls a merge
        # of county geometry and census data:
        # https://fusiontables.google.com/data?docid=1S4EB6319wWW2sWQDPhDvmSBIVrD3iEmCLYB7nMM#rows:id=1
        if region_type == 'uscounties':
            region = ee.FeatureCollection('ft:18Ayj5e7JxxtTPm1BdMnnzWbZMrxMB49eqGDTsaSp')
        elif region_type == 'countygeometries':
            region = ee.FeatureCollection('ft:1S4EB6319wWW2sWQDPhDvmSBIVrD3iEmCLYB7nMM')
        elif region_type == 'world':
            region = ee.FeatureCollection('ft:1tdSwUL7MVpOauSgRzqVTOwdfy17KDbw-1d9omPw')

        for data in self.locations.values[: export_limit]:
            if len(data) == 4:
                state_id, county_id, lat, lon = data
                fname = '{}_{}'.format(int(state_id), int(county_id))
            else:
                country, index = data
                fname = f'index{index}'

            # filter for a county
            if region_type == 'uscounties':
                file_region = region.filterMetadata('STATE num', 'equals', state_id)
                file_region = ee.FeatureCollection(file_region).filterMetadata('COUNTY num', 'equals', county_id)
                file_region = file_region.first()
                file_region = file_region.geometry().coordinates().getInfo()[0]
                processed_img = img
            elif region_type == 'countygeometries':
                file_region = region.filterMetadata('StateFips', 'equals', int(state_id))
                file_region = ee.FeatureCollection(file_region).filterMetadata('CntyFips', 'equals', int(county_id))
                file_region = ee.Feature(file_region.first())
                processed_img = img.clip(file_region)
                file_region = None
            elif region_type == 'world':
                file_region = region.filterMetadata('Country', 'equals', country)
                if file_region is None:
                    print(country, index, 'not found')
                    continue
                file_region = file_region.first()
                file_region = file_region.geometry().coordinates().getInfo()[0]
                processed_img = img
            else:
                file_region = str([
                    [lat - offset, lon + offset],
                    [lat + offset, lon + offset],
                    [lat + offset, lon - offset],
                    [lat - offset, lon - offset]])
                processed_img = img

            while True:
                try:
                    self._export_one_image(processed_img, folder_name, fname, file_region, scale, coordinate_system)
                except:
                    print
                    'retry'
                    time.sleep(10)
                    continue
                break
        print('Finished Exporting!')

    def export_all(self, export_limit=None):
        """
        Export all the data. This is equivalent to running the 5/10 scripts on the original github repo.
        """

        # first, make sure the class was initialized correctly
        self.update_parameters(locations_filepath=Path('data/locations_final.csv'),
                               collection_id='MODIS/MOD09A1')

        # pull_MODIS_entire_county_clip.py
        self.export(folder_name='crop_yield/test', datefilter='default',
                    region_type='countygeometries', min_img_val=16000, max_img_val=100,
                    export_limit=export_limit)

        # pull_MODIS_landcover_entire_county_clip.py
        self.update_parameters(locations_filepath=Path('data/subset_locations.csv'),
                               collection_id='MODIS/051/MCD12Q1')
        self.export(folder_name='crop_yield/data_mask', datefilter='default',
                    region_type='countygeometries', export_limit=export_limit)

        # pull_MODIS_temperature_entire_county_clip.py
        self.update_parameters(collection_id='MODIS/MYD11A2')
        self.export(folder_name='crop_yield/data_temperature', datefilter='default',
                    region_type='countygeometries', export_limit=export_limit)

        # pull_MODIS_world.py
        self.update_parameters(locations_filepath=Path('data/world_locations.csv'),
                               collection_id='MODIS/MOD09A1')
        self.export(folder_name='crop_yield/data_world', datefilter='default', region_type='world',
                    min_img_val=5000, max_img_val=0,
                    export_limit=export_limit)
