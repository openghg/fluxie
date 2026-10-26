# Input data format

fluxie reads model outputs from netCDF files following a common template.
If you are adding a new model to fluxie, you need to produce netCDF files
compatible with this template.

## Model outputs

The files must follow the PARIS-AVENGERS-EYECLIMA template available in the folder
[`data/templates/`](https://github.com/openghg/fluxie/tree/devel/data/templates)
(see files `PAR-AVE-EYE_inversion_flux_output.cdl` and `PAR-AVE-EYE_inversion_concentration_output.cdl`).
A summary of the most important variables is given in
[`README_templates.md`](https://github.com/openghg/fluxie/blob/devel/data/templates/README_templates.md).
Examples of files following the template can be found in the `data/tests` folder.

The filenames must have the following format:

- Flux file: `<inversionModel>_<optional_identifying_tags>_<species>_<inversionFrequency>.nc`  
- Concentration file: `<inversionModel>_<optional_identifying_tags>_<species>_<inversionFrequency>_concentrations.nc`

`<inversionFrequency>` should be equal to "yearly" or "monthly".

For easy traceability and nice automatic labels, consider replacing `<optional_identifying_tags>` by `<transportModel>_<domain>_<prior>`.

e.g.:

- InTEM_NAME_EUROPE_EDGAR_hfc134a_yearly.nc
- InTEM_NAME_EUROPE_EDGAR_hfc134a_yearly_concentrations.nc

The following folder structure is expected: `/path/to/data/<inversionModel>/<species>/`

## Bottom-up inventory country fluxes (optional)

The inventory files must be in the following location: `/path/to/data/inventory/`  
The filenames should follow the following format: `<inventory_identifier>_<species>_<year>.nc` (e.g. "UNFCCC_inventory_hfc134a_2024.nc")  
The data format must follow the PARIS-AVENGERS-EYECLIMA template for inventory files: `data/templates/PAR-AVE-EYE_inventory.cdl`

## Baseline timestamps (optional)

The files with baseline timestamps must be in the following location: `/path/to/data/baseline_timestamps/`  
The filenames should follow the following format: `<stationID>_<baseline_identifier>_baseline_timestamps.nc` (e.g. "JFJ_InTEM_baseline_timestamps.nc")  
The data format must follow the PARIS-AVENGERS-EYECLIMA template for baseline timestamp files: `data/templates/PAR-AVE-EYE_baseline_timestamps.cdl`

## Prior sector fluxes (optional)

To split the total country fluxes into sectors with
[`scale_by_sector_proportions`][fluxie.operators.flux_scale_by_sector_proportions.scale_by_sector_proportions],
fluxie uses the sectoral fluxes of a gridded prior inventory.

The files must be in the following location: `/path/to/data/sector_flux/`  
The filenames should follow the following format: `<domain>_<inventory>_<species>_yearly_flux_sectors.nc` (e.g. "EUROPE_EDGAR_ch4_yearly_flux_sectors.nc")  
The data format must follow the template `data/templates/PAR-AVE-EYE_sector_flux.cdl`.
The grid cell areas are read from `configs/<domain>_cell_area.nc` (template `data/templates/PAR-AVE-EYE_cell_area.cdl`).

## Eddy covariance fluxes

Eddy covariance files follow the template `data/templates/ec_flux.cdl` and are
named `<model>_<species>_eddy_flux.nc` (e.g. `EDDY_HARDAU_co2_eddy_flux.nc`).
See `data/tests/ecflux` for an example.
