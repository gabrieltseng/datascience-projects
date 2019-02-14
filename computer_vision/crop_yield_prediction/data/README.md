## Data

This directory stores useful files, and is the location to which Earth Engine Exports should be saved.

### Contents:

#### [`yield_data.csv`](yield_data.csv)

This is the yield data from the USDA website, which measures soybean yields in bushels per acre. The data can be reproduced
at the following [link](https://quickstats.nass.usda.gov/#5A65CCEF-B75F-366D-AA20-5632E0073EA1).

Click on `Get Data`, and then `Spreadsheet`, to generate a copy of the csv file. This file is analogous to
[`yield_final.csv`](https://github.com/JiaxuanYou/crop_yield_prediction/blob/master/2%20clean%20data/yield_final.csv) in
the original repository.

#### [`county_data.csv`](county_data.csv)

This is the county data from a US 2010 Census [fusion table](https://support.google.com/fusiontables/answer/2571232), 
which we also use to delineate the county borders when exporting the MODIS data from the Earth Engine. It is used for 
its latitude and longitude data, which allows the distance between counties to be approximated. The data can be 
reproduced at the following [link](https://fusiontables.google.com/data?docid=1S4EB6319wWW2sWQDPhDvmSBIVrD3iEmCLYB7nMM#rows:id=1).

Note that fusion tables will [not be available after December 3rd 2019](https://support.google.com/fusiontables/answer/9185417).

Click on `File`, and then `Download`, and download all rows as a CSV file.

#### [`counties.svg`](counties.svg)

A map of the U.S., with county delineations. Taken from the 
[original repository](https://github.com/JiaxuanYou/crop_yield_prediction/blob/master/6%20result_analysis/counties.svg), 
and useful for analysis.
