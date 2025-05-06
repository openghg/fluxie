# Inverse Modelling Intercomparison Tool

This repository contains functions to compare inverse models developed under the PARIS project, and a notebook to allow for easy use of these functions.

Follow the steps below to run the notebook and plot model results.

## Installation

Clone the repository and install fluxy:
```
git clone https://github.com/openghg/fluxy.git
cd fluxy
pip install -e .
```
Note: in the ICOS Jupyter Hub, you might need to restart the kernel so that package fluxy is found.

## Prepare input files
### 1. Flux and concentration netCDF files with model results

Data format must be in agreement with the PARIS-AVENGERS-EYECLIMA template. Mandatory variables are described below. The full variable list can be found in the cdl files in folder templates/.

#### a) Flux file

| Dimension variables | Type   | Units                          | Description                                      |
|:--------------------|:-------|:-------------------------------|:-------------------------------------------------|
| longitude           | double | degrees_east                   | Longitude of grid cell centre
| latitude            | double | degrees_north                  | Latitude of grid cell centre
| time                | double | days since 1970-01-01 00:00:00 | Mid of flux interval in UTC
| time_bnds           | double | days since 1970-01-01 00:00:00 | Start and end points of each flux interval in UTC
| country             | string | -                              | Country ISO 3166-1 alpha-3 code

| Grid variables             | Type  | Units (1)   | Dimensions                | Description                                  |
|:---------------------------|:------|:------------|:--------------------------|:---------------------------------------------|
| flux_total_prior           | float | mol m-2 s-1 | time, latitude, longitude | Prior total `<species>` fluxes
| flux_total_posterior       | float | mol m-2 s-1 | time, latitude, longitude | Posterior total `<species>` fluxes
| stdev_flux_total_prior     | float | mol m-2 s-1 | time, latitude, longitude | Standard deviation of prior total `<species>` fluxes
| stdev_flux_total_posterior | float | mol m-2 s-1 | time, latitude, longitude | Standard deviation of posterior total `<species>` fluxes

(1) Any SI unit of the type "amount length-2 time-1" and "mass length-2 time-1" is valid. All grid variables should have the same units.

| By-country variables               | Type  | Units (2) | Dimensions    | Description                                      |
|:-----------------------------------|:------|:----------|:--------------|:-------------------------------------------------|
| flux_total_prior_country           | float | kg yr-1   | time, country | Country-total prior `<species>` fluxes
| flux_total_posterior_country       | float | kg yr-1   | time, country | Country-total posterior `<species>` fluxes
| stdev_flux_total_prior_country     | float | kg yr-1   | time, country | Standard deviation of country-total prior `<species>` fluxes
| stdev_flux_total_posterior_country | float | kg yr-1   | time, country | Standard deviation of country-total posterior `<species>` fluxes

(2) Any SI unit of the type "mass time-1" is valid. All by-country variables should have the same units.

| Auxiliary variables | Type  | Units  |  Dimensions                  | Description                                |
|:--------------------|:------|:-------|:-----------------------------|:-------------------------------------------|
| country_fraction    | float | -      | country, latitude, longitude | Fraction of grid cell associated to country
| cell_area           | float | m2     | latitude, longitude          | Surface area of gird cell

#### b) Concentration file

| Characterising variables | Type   | Units                          | Dimensions | Description                                      |
|:-------------------------|:-------|:-------------------------------|:-----------|:-------------------------------------------------|
| longitude                | double | degrees_east                   | index      | Sample longitude in decimal degrees
| latitude                 | double | degrees_north                  | index      | Sample latitude in decimal degrees
| time                     | double | days since 1970-01-01 00:00:00 | index      | Time of mid of observation interval in UTC
| time_bnds                | double | days since 1970-01-01 00:00:00 | index      | Start and end points of each time step
| altitude                 | float  | m                              | index      | Sample altitude in meters above sea level
| number_of_identifier     | short  | -                              | index      | Index of identifier of observing platform
| assimilation_flag        | short  | -                              | index      | Flag indicating whether observation was used in inversion/assimilation (0: not used; 1: used)

| Observation variables | Type   | Units (3) | Dimensions | Description                                      |
|:----------------------|:-------|:----------|:-----------|:-------------------------------------------------|
| platform              | string | -         | index      | Identifier of observing platform
| mf_observed           | float  | mol mol-1 | index      | Observed mole fraction of `<species>` in dry air
| stdev_mf_total        | float  | mol mol-1 | index      | Total model-data-mismatch uncertainty applied in inversion

| Simulated variables | Type  | Units (3) | Dimensions | Description                                      |
|:--------------------|:------|:----------|:-----------|:-------------------------------------------------|
| mf_prior            | float | mol mol-1 | index      | Prior simulated mole fraction of `<species>` in dry air
| mf_posterior        | float | mol mol-1 | index      | Posterior simulated mole fraction of `<species>` in dry air
| mf_bc_prior         | float | mol mol-1 | index      | Prior simulated boundary condition mole fraction including site bias
| mf_bc_posterior     | float | mol mol-1 | index      | Posterior simulated boundary condition mole fraction including site bias

(3) ppm, ppb and ppt are also valid units.

#### c) File naming

Filenames should follow the following format:  
- Flux file: `<inversionModel>_<optional_identifying_tags>_<species>_<inversionFrequency>.nc`  
- Concentration file: `<inversionModel>_<optional_identifying_tags>_<species>_<inversionFrequency>_concentration.nc`  

`<inversionFrequency>` should be equal to "yearly" or "monthly".

For easy traceability and nice automatic labels, consider replacing `<optional_identifying_tags>` by `<transportModel>_<domain>_<prior>`.

e.g.:  
InTEM_NAME_EUROPE_EDGAR_hfc134a_yearly.nc  
InTEM_NAME_EUROPE_EDGAR_hfc134a_yearly_concentration.nc

The following folder structure is expected:
`/path/to/data/<inversionModel>/<species>/`

### 2. Regions information (file "regions_info.json")

Example file located in folder configs/.

| Variables     | Type                      | Description  |
|:--------------|:--------------------------|:-------------|
| country_codes | dict[str,str]             | Country names and respective ISO 3166-1 alpha-3 codes. |
| regions       | dict[str,str]             | Regions corresponding to aggregation of countries. |
| point_source  | dict[str,list]            | Latitude/longitude coordinates of points of interest. |

### 3. Models information (file "models_info.json")

Example file located in folder configs/.

| Variables     | Type                      | Description  |
|:--------------|:--------------------------|:-------------|
| filename_tags | dict[str,str] (optional)  | Dictionary of keys that point to long filename strings. <br> Used to reduce the sequence of name tags that constitute the model run name. <br> e.g. if {"std" : "4sites_baseline_optimized"}, model run name "InTEM_NAME_EDGAR_std" points to "InTEM_NAME_EDGAR_4sites_baseline_optimized". <br> Note that `<model>` can be used as a generic tag which will be replaced by the inversion model name in lower case. |
| model_labels  | dict[str, str] (optional) | Dictionary with model run names and respective labels to use in the plot. <br> If not defined, the label is created automatically from the model run names. |
| species_name  | dict[str,dict] (optional) | Species name that should replace `<species>` in the filename. <br> By default, `<species>` is assumed equal to the value specified in the notebook (e.g. "hfc134a"). <br> Use this dictionary to specify model specific species name. Dictionary keys should correspond to `<inversionModel>`. |
| standard_run  | dict[str,dict]            | Name tags (`<transportModel>_<prior>_<optional_tags>`) that identify the standard run for all models and each gas. <br> These runs are considered when summing country fluxes from all HFCs or PFCs (e.g. option species="all_hfc"). <br> To use the name tags specified under "default", specify only the `<inversionModel>` name in the notebook (e.g. models=["InTEM","RHIME"]). <br> Define other dictionary keys (e.g. "longrun") to specify a different set of model runs. You can point to these runs by specifying `<inversionModel>_<key_name>` (e.g. models=["InTEM_longrun"]). Missing species will be taken from the "default" dictionary. |

### 4. Species information (file "species_info.json")

Example file located in folder configs/.

It contains a dictionary of species (or group of species) pointing to various properties/print settings.

| Keys          | Type                   | Description                                                                                                         |
|:--------------|:-----------------------|:--------------------------------------------------------------------------------------------------------------------|
| species_print | str                    | Species name used in the plot axis (LaTeX format).                                                                  |
| gwp           | float (optional)       | Global Warming Potential. <br> Used to convert country fluxes to mass of CO2 equivalent.                            |
| molar_mass    | float (optional)       | Species molar mass (g mol-1). <br> Used to apply mol<->g conversion to fluxes.                                      |
| list_species  | list of str (optional) | List of species which define a given group of species. <br> Used to plot sum of country fluxes over various species.|

### 5. Sites information (file "site_info.json")

Example file located in folder configs/.

It contains a dictionary of stations (station designation code) pointing to the respective observation network (e.g. ICOS).  
For each pair station/observation network, there is a dictionary of station specifications:

| Keys                | Type                   | Description                               |
|:--------------------|:-----------------------|:------------------------------------------|
| latitude            | float                  | Latitude of the station (degrees N)       |
| longitude           | float                  | Longitude of the station (degrees E)      |
| height_station_masl | float (optional)       | Station height (meters above sea level)   |
| long_name           | str (optional)         | Station long name                         |
| height              | list of str (optional) | Inlet heights (meters above ground level) |
| height_name         | list of str (optional) | Inlet heights name                        |

### 6. netCDF file with country fluxes from bottom-up inventory (optional)

The inventory files must be in the following location: `/path/to/data/inventory/`  
The filenames should follow the following format: `<inventory_identifier>_<species>_<year>.nc` (e.g. "UNFCCC_inventory_hfc134a_2024.nc")  
Data format must be in agreement with the PARIS-AVENGERS-EYECLIMA template for inventory files.

### 7. netCDF file with baseline timestamps (optional)

The files with baseline timestamps must be in the following location: `/path/to/data/<baseline_identifier>/`  
The filenames should follow the following format: `<stationID>_<baseline_identifier>.nc` (e.g. "JFJ_InTEM_baseline_timestamps.nc")  
Data format must be in agreement with the PARIS-AVENGERS-EYECLIMA template for baseline timestamp files.

## Run the notebook

The notebook that allows you plot the different variables of interest is located in:
`scripts/PARIS_inversion_results.ipynb`

1. In the first notebook cell, specify the path to the data and the `experiments` dictionary which points to the model runs you want to plot.

The model runs are identified by providing the output name tags:
`<inversionModel>_<optional_identifying_tags>` (e.g.: "InTEM_NAME_EUROPE_EDGAR")

Note that `<species>` and `<inversionFrequency>` are not specified in the model run name because they are defined in designated variables in the notebook.
If the sequence of name tags is too long, a simplified name tag can be defined in `filename_tags` in models_info.json (see #3).

2. The notebook is organized in 3 sections numbered 1 to 3. At the top of each section, specify the models you want to plot, species name, start/end dates, etc (plotting options are described in front of each variable). Run the top cell to read in the data and select values between the chosen dates.

3. In the subsequent cells under each numbered section, edit the plotting options according to your preference and run the cells to produce various plots.
