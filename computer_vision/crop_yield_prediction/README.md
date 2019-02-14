# PyCrop Yield Prediction

A PyTorch implementation of [Jiaxuan You](https://cs.stanford.edu/~jiaxuan/)'s 2017 Crop Yield Prediction Project.

> [Deep Gaussian Process for Crop Yield Prediction Based on Remote Sensing Data](https://cs.stanford.edu/~ermon/papers/cropyield_AAAI17.pdf)

## Introduction

This repo contains a pytorch implementation of the Deep Gaussian Process for Crop Yield Prediction. It draws from the
original [Tensorflow implementation](https://github.com/JiaxuanYou/crop_yield_prediction).

Neural networks treat each datapoint as independent and identically distributed, making a prediction for each test point
in isolation. In the case of predicting crop yields, counties are likely to be correlated based on their spatial and
temporal proximity in a way which the input data may not capture. For instance, soil quality is likely to be similar
across geographically close counties.

Gaussian Processes make predictions on test points by considering their proximity to training points (where proximity is
defined in terms of time and space), allowing the model to improve by uncovering spatial and temporal correlations.

In this pipeline, a Deep Gaussian Process is used to predict soybean yields in US counties.

## Setup

[Anaconda](https://www.anaconda.com/download/#macos) running python 3.7 is used as the package manager. To get set up
with an environment, install Anaconda from the link above, and (from this directory) run

```bash
conda env create -f environment.yml
```
This will create an environment named `crop_yield_prediction` with all the necessary packages to run the code. To 
activate this environment, run

```bash
conda activate crop_yield_prediction
```

Running this code also requires you to sign up to [Earth Engine](https://developers.google.com/earth-engine/). Once you 
have done so, active the `crop_yield_prediction` environment and run

```bash
earthengine authenticate
```

and follow the instructions. To test that everything has worked, run

```bash
python -c "import ee; ee.Initialize()"
```

Note that Earth Engine exports files to Google Drive by default (to the same google account used sign up to Earth Engine.)

To download the data used in the paper (MODIS images of the top 11 soybean producing states in the US) requires
just over **110 Gb** of storage. This can be done in steps - the export class allows for checkpointing.

## Pipeline

The main entrypoint into the pipeline is [`run.py`](run.py). The pipeline is split into 4 major components. Note that
each component reads files from the previous step, and saves all files that later steps will need, into the 
[`data`](data) folder.

Parameters which can be passed in each step are documented in [`run.py`](run.py). The default parameters are all taken
from the original repository.

[Python Fire](https://github.com/google/python-fire) is used to generate command line interfaces.

#### Exporting

```bash
python run.py export
```

Exports data from the Google Earth Engine to Google Drive. Note that to make the export more efficient, all the bands
from a county - across all the export years - are concatenated, reducing the number of files to be exported.

#### Preprocessing

```bash
python run.py process
```

Takes the exported and downloaded data, and splits the data by year. In addition, the temperature and reflection `tif` 
files are merged, and the mask is applied so only farmland is considered. Files are saved as `.npy` files.

The size of the processed files is 

#### Feature Engineering

```bash
python run.py engineer
``` 
Take the processed `.npy` files and generate histogams which can be input into the models. The total size of the `.npy`
files is **97 GB**. Running with the flag `delete_when_done=True` will delete the `.tif` files as they get processed. 

#### Model training

```bash
python run.py train_cnn
```
and
```bash
python run.py train_rnn
```

Trains CNN and RNN models, respectively, with a Gaussian Process. The trained models are saved in 
`data/models/<model_type>` and results are saved in csv files in those folders. If a Gaussian Process is used, the
results of the model without a Gaussian Process are also saved for analysis.


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
