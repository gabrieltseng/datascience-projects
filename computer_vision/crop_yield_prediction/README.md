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

## Additional notes so far

The following two tables are used for county deliminations:

* [Copy of United States Counties](https://fusiontables.google.com/data?docid=18Ayj5e7JxxtTPm1BdMnnzWbZMrxMB49eqGDTsaSp#rows:id=3)
* [Merge of County Geometry and Census Data with County KML and US County Census Data 2010](https://fusiontables.google.com/data?docid=1S4EB6319wWW2sWQDPhDvmSBIVrD3iEmCLYB7nMM#rows:id=1)

Both are hosted on Google's FusionTables, which will [not be available after December 3rd 2019](https://support.google.com/fusiontables/answer/9185417).
