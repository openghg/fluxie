# PARIS-AVENGERS-EYECLIMA output template description 

The most important variables are described below. Please refer to the cdl files for a complete description.

## 1. Flux file

| Dimension variables | Old name | Units                          | Description                                      |
|:--------------------|:---------|:-------------------------------|:-------------------------------------------------|
| longitude           | -        | degrees_east                   | Longitude of grid cell centre
| latitude            | -        | degrees_north                  | Latitude of grid cell centre
| time                | -        | days since 1970-01-01 00:00:00 | Mid of flux interval in UTC
| time_bnds           | (absent) | days since 1970-01-01 00:00:00 | Start and end points of each flux interval in UTC
| country             | -        | -                              | Country ISO 3166-1 alpha-3 code
| *Optional if using percentile instead of stdev (non-Gaussian PDFs)*
| percentile          | -        | -                              | Percentile of flux pdf

| Grid variables                                 | Old name | Units (1)   | Dimensions                | Description                                  |
|:-----------------------------------------------|:---------|:------------|:--------------------------|:---------------------------------------------|
| flux_total_prior                               | -        | mol m-2 s-1 | time, latitude, longitude | Prior total `<species>` fluxes
| flux_total_posterior                           | -        | mol m-2 s-1 | time, latitude, longitude | Posterior total `<species>` fluxes
| stdev_flux_total_prior                         | (absent) | mol m-2 s-1 | time, latitude, longitude | Standard deviation of prior total `<species>` fluxes
| stdev_flux_total_posterior                     | (absent) | mol m-2 s-1 | time, latitude, longitude | Standard deviation of posterior total `<species>` fluxes
| *Optional variables*
| flux_total_prior_inversion_grid                | -        | mol m-2 s-1 | time, latitude, longitude | Prior total `<species>` fluxes on the reduced inversion grid
| flux_total_posterior_inversion_grid            | -        | mol m-2 s-1 | time, latitude, longitude | Posterior total `<species>` fluxes on the reduced inversion grid
| stdev_flux_total_prior_inversion_grid          | (absent) | mol m-2 s-1 | time, latitude, longitude | Standard deviation of prior total `<species>` fluxes on the reduced inversion grid
| stdev_flux_total_posterior_inversion_grid      | (absent) | mol m-2 s-1 | time, latitude, longitude | Standard deviation of posterior total `<species>` fluxes on the reduced inversion grid
| *Alternative to stdev for non-Gaussian PDFs*
| percentile_flux_total_prior                    | - | mol m-2 s-1 | time, percentile, latitude, longitude | Percentile of prior total `<species>` fluxes
| percentile_flux_total_posterior                | - | mol m-2 s-1 | time, percentile, latitude, longitude | Percentile of posterior total `<species>` fluxes
| percentile_flux_total_prior_inversion_grid     | - | mol m-2 s-1 | time, percentile, latitude, longitude | Percentile of prior total `<species>` fluxes on the reduced inversion grid
| percentile_flux_total_posterior_inversion_grid | - | mol m-2 s-1 | time, percentile, latitude, longitude | Percentile of posterior total `<species>` fluxes on the reduced inversion grid

(1) fluxy accepts any SI unit of the type "amount length-2 time-1" and "mass length-2 time-1". All grid variables should have the same units.

| By-country variables               | Old name  | Units (2) | Dimensions    | Description                                      |
|:-----------------------------------|:------|:----------|:--------------|:-------------------------------------------------|
| flux_total_prior_country           | float | kg yr-1   | time, country | Country-total prior `<species>` fluxes
| flux_total_posterior_country       | float | kg yr-1   | time, country | Country-total posterior `<species>` fluxes
| stdev_flux_total_prior_country     | float | kg yr-1   | time, country | Standard deviation of country-total prior `<species>` fluxes
| stdev_flux_total_posterior_country | float | kg yr-1   | time, country | Standard deviation of country-total posterior `<species>` fluxes
| Alternative to stdev for non-Gaussian PDFs
| percentile_flux_total_prior_country
| percentile_flux_total_posterior_country

(2) fluxy accepts any SI unit of the type "mass time-1". All by-country variables should have the same units.

| Auxiliary variables | Old name  | Units  |  Dimensions                  | Description                                |
|:--------------------|:------|:-------|:-----------------------------|:-------------------------------------------|
| country_fraction    | float | -      | country, latitude, longitude | Fraction of grid cell associated to country
| cell_area           | float | m2     | latitude, longitude          | Surface area of gird cell

## 2. Concentration file

| Characterising variables | Old name   | Units                          | Dimensions | Description                                      |
|:-------------------------|:-------|:-------------------------------|:-----------|:-------------------------------------------------|
| longitude                | double | degrees_east                   | index      | Sample longitude in decimal degrees
| latitude                 | double | degrees_north                  | index      | Sample latitude in decimal degrees
| time                     | double | days since 1970-01-01 00:00:00 | index      | Time of mid of observation interval in UTC
| time_bnds                | double | days since 1970-01-01 00:00:00 | index      | Start and end points of each time step
| altitude                 | float  | m                              | index      | Sample altitude in meters above sea level
| number_of_identifier     | short  | -                              | index      | Index of identifier of observing platform
| assimilation_flag        | short  | -                              | index      | Flag indicating whether observation was used in inversion/assimilation (0: not used; 1: used)

| Observation variables | Old name   | Units (3) | Dimensions | Description                                      |
|:----------------------|:-------|:----------|:-----------|:-------------------------------------------------|
| platform              | string | -         | index      | Identifier of observing platform
| mf_observed           | float  | mol mol-1 | index      | Observed mole fraction of `<species>` in dry air
| stdev_mf_total        | float  | mol mol-1 | index      | Total model-data-mismatch uncertainty applied in inversion

| Simulated variables | Old name  | Units (3) | Dimensions | Description                                      |
|:--------------------|:------|:----------|:-----------|:-------------------------------------------------|
| mf_prior            | float | mol mol-1 | index      | Prior simulated mole fraction of `<species>` in dry air
| mf_posterior        | float | mol mol-1 | index      | Posterior simulated mole fraction of `<species>` in dry air
| mf_bc_prior         | float | mol mol-1 | index      | Prior simulated boundary condition mole fraction including site bias
| mf_bc_posterior     | float | mol mol-1 | index      | Posterior simulated boundary condition mole fraction including site bias

(3) fluxy also accepts ppm, ppb and ppt.
