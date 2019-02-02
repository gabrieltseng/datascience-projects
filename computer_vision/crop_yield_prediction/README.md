## Setup

Running this code requires you to sign up to [Earth Engine](https://developers.google.com/earth-engine/).

Once you have done so, run

```bash
earthengine authenticate
```

and follow the instructions. To test that everything has worked, run

```bash
python -c "import ee; ee.Initialize()"
```

Note that Earth Engine exports files to Google Drive by default (to the same google account used sign up to Earth Engine.)

## MODIS datasets

The satellite data used comes from the Moderate Resolution Imaging Spectroradiometer 
([MODIS](https://en.wikipedia.org/wiki/Moderate_Resolution_Imaging_Spectroradiometer)), aboard the [Terra](https://en.wikipedia.org/wiki/Terra_(satellite))
satellite.

Specifically, the following datasets are used:

#### [MOD09A1: Surface Reflectance](https://lpdaac.usgs.gov/dataset_discovery/modis/modis_products_table/mod09a1_v006)

"An estimate of the surface spectral reflectance of Terra MODIS bands 1 through 7".

Basically, an 'image' of the county as seen from the satellite.

#### [MCD12Q1: Land Cover Type](https://lpdaac.usgs.gov/dataset_discovery/modis/modis_products_table/mcd12q1)

Labels the data according to one of five global land cover classification systems. This is used as a mask, because we only
want to consider pixels associated with farmland.

#### [MYD11A2: Aqua/Land Surface Temperature](https://lpdaac.usgs.gov/dataset_discovery/modis/modis_products_table/myd11a2_v006)

Two more bands which can be used as input data to our models.

## Additional notes so far

The following table is used for county deliminations:

* [Copy of United States Counties](https://fusiontables.google.com/data?docid=1S4EB6319wWW2sWQDPhDvmSBIVrD3iEmCLYB7nMM#rows:id=1)

It is hosted on Google's FusionTables, which will [not be available after December 3rd 2019](https://support.google.com/fusiontables/answer/9185417).
