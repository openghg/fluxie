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

Data format must be in agreement with the PARIS-AVENGERS-EYECLIMA template.

Filenames should follow the following format:  
`<inversionModel>_<transportModel>_<domain>_<prior>_<optional_tags>_<species>_<inversionFrequency>(_concentration).nc`

Note that the part within parenthesis refers to the concentration file only.
`<inversionFrequency>` should be equal to "yearly" or "monthly".

e.g.:  
InTEM_NAME_EUROPE_EDGAR_4sites_hfc134a_yearly.nc  
InTEM_NAME_EUROPE_EDGAR_4sites_hfc134a_yearly_concentration.nc

The following folder structure is expected:
`/path/to/data/<inversionModel>/<species>/`

### 2. Regions information (file "regions_info.json")

Example file located in folder configs/.

| Variables     | Type                      | Description  |
|:--------------|:--------------------------|:-------------|
| domain        | str                       | Domain name tag used in the filename (e.g. "EUROPE"). |
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

The model runs are identified by providing the following sequence of name tags:
`<inversionModel>_<transportModel>_<prior>_<optional_tags>` (e.g.: "InTEM_NAME_EDGAR_4sites")

Note that `<domain>` is not specified in the name of the model run because it is defined in regions_info.json (see #2).
`<species>` and `<inversionFrequency>` are not specified in the model run name because they are defined in designated variables in the notebook.
If the sequence of name tags is too long, a simplified name tag can be defined in `filename_tags` in models_info.json (see #3).

2. The notebook is organized in 3 sections numbered 1 to 3. At the top of each section, specify the models you want to plot, species name, start/end dates, etc (plotting options are described in front of each variable). Run the top cell to read in the data and select values between the chosen dates.

3. In the subsequent cells under each numbered section, edit the plotting options according to your preference and run the cells to produce various plots.
