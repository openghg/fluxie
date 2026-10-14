# Configuration files

fluxie uses configuration files for information that does not depend on the
model outputs. The default configuration files are in the
[`configs/`](https://github.com/openghg/fluxie/tree/devel/configs) folder of the
repository and are read with [`read_config_files`][fluxie.io.read_config_files]:

```python
from fluxie.io import read_config_files

config_data = read_config_files()  # default configs/ folder
config_data = read_config_files("path/to/my/configs")  # your own configs
```

`config_data` is a dictionary whose keys are the file names without extension
(e.g. `config_data["species_info"]`). It is passed to most fluxie functions.

All `.json`, `.yaml` and `.yml` files of the folder are read. To use your own
configuration, copy the `configs/` folder and edit the files.

## Regions information

Example file: `configs/regions_info.json`

| Variables     | Type                      | Description  |
|:--------------|:--------------------------|:-------------|
| country_codes | dict[str,str]             | Country names and respective ISO 3166-1 alpha-3 codes. |
| regions       | dict[str,str]             | Regions corresponding to aggregation of countries. |
| point_source  | dict[str,list]            | Latitude/longitude coordinates of points of interest. |

## Models information

Example file: `configs/models_info.json`

| Variables     | Type           | Description  |
|:--------------|:---------------|:-------------|
| filename_tags | dict[str,str]  | Dictionary of keys that point to long filename strings. <br> Used to reduce the sequence of name tags that constitute the model run name. <br> e.g. if {"std" : "4sites_baseline_optimized"}, model run name "InTEM_NAME_EDGAR_std" points to "InTEM_NAME_EDGAR_4sites_baseline_optimized". <br> Note that `<model>` can be used as a generic tag which will be replaced by the inversion model name in lower case. |
| model_labels  | dict[str,str]  | Dictionary with model run names and respective labels to use in the plot. <br> If not defined, the label is created automatically from the model run names. |
| species_name  | dict[str,dict] | Species name that should replace `<species>` in the filename. <br> By default, `<species>` is assumed equal to the value specified in the notebook (e.g. "hfc134a"). <br> Use this dictionary to specify model specific species name. Dictionary keys should correspond to `<inversionModel>`. |
| standard_run  | dict[str,dict] | Name tags (`<transportModel>_<prior>_<optional_tags>`) that identify the standard run for all models and each gas. <br> These runs are considered when summing country fluxes from all HFCs or PFCs (e.g. option species="all_hfc"). <br> To use the name tags specified under "default", specify only the `<inversionModel>` name in the notebook (e.g. models=["InTEM","RHIME"]). <br> Define other dictionary keys (e.g. "longrun") to specify a different set of model runs. You can point to these runs by specifying `<inversionModel>_<key_name>` (e.g. models=["InTEM_longrun"]). Missing species will be taken from the "default" dictionary. |

## Species information

Example file: `configs/species_info.json`

It contains a dictionary of species (or group of species) pointing to various properties/print settings.

| Keys          | Type        | Description                                                                                                           |
|:--------------|:------------|:----------------------------------------------------------------------------------------------------------------------|
| species_print | str         | Species name used in the plot axis (LaTeX format).                                                                    |
| gwp           | float       | Global Warming Potential. <br> GWP-100 values from IPCC AR5, used to convert country fluxes to mass of CO2 equivalent.|
| molar_mass    | float       | Species molar mass (g mol-1). <br> Used to apply mol<->g conversion to fluxes.                                        |
| list_species  | list of str | List of species which define a given group of species. <br> Used to plot sum of country fluxes over various species.  |

## Sites information

Example file: `configs/site_info.json`

It contains a dictionary of stations (station designation code) pointing to the respective observation network (e.g. ICOS).  
For each pair station/observation network, there is a dictionary of station specifications:

| Keys                | Type        | Description                               |
|:--------------------|:------------|:------------------------------------------|
| latitude            | float       | Latitude of the station (degrees N)       |
| longitude           | float       | Longitude of the station (degrees E)      |
| height_station_masl | float       | Station height (meters above sea level)   |
| long_name           | str         | Station long name                         |
| height              | list of str | Inlet heights (meters above ground level) |
| height_name         | list of str | Inlet heights name                        |

## Sectors

Example file: `configs/sectors.yml`

| Keys                      | Description |
|:--------------------------|:------------|
| sectors_groups            | Groups of sectors, e.g. all the road transport sectors in `Road Transport`. Used by [`group_sectors`][fluxie.operators.sectors.group_sectors]. |
| sector_groups_per_species | Sector groups to use for each species. |
| sector_ordering           | Order in which the sectors are stacked in the plots. |
| colors_of_sector          | Colors of the sectors in the plots. |

Sector groups are used for instance in the [eddy covariance plots](../gallery.md#eddy-covariance-fluxes).
