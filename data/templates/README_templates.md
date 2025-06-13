# PARIS-AVENGERS-EYECLIMA output template description 

The most important variables are described below. Please refer to the cdl files for a complete description.

## 1. Flux file

| Dimension variables | Units                          | Description                                      |
|:--------------------|:-------------------------------|:-------------------------------------------------|
| longitude           | degrees_east                   | Longitude of grid cell centre
| latitude            | degrees_north                  | Latitude of grid cell centre
| time                | days since 1970-01-01 00:00:00 | Mid of flux interval in UTC
| time_bnds           | days since 1970-01-01 00:00:00 | Start and end points of each flux interval in UTC
| country             | -                              | Country ISO 3166-1 alpha-3 code
| *Optional if using percentile instead of stdev (non-Gaussian PDFs)*
| percentile          | -                              | Percentile of flux pdf

| Grid variables                                 | Units (1)   | Dimensions                | Description                                  |
|:-----------------------------------------------|:------------|:--------------------------|:---------------------------------------------|
| flux_total_prior                               | mol m-2 s-1 | time, latitude, longitude | Prior total `<species>` fluxes
| flux_total_posterior                           | mol m-2 s-1 | time, latitude, longitude | Posterior total `<species>` fluxes
| stdev_flux_total_prior                         | mol m-2 s-1 | time, latitude, longitude | Standard deviation of prior total `<species>` fluxes
| stdev_flux_total_posterior                     | mol m-2 s-1 | time, latitude, longitude | Standard deviation of posterior total `<species>` fluxes
| *Optional variables*
| flux_total_prior_inversion_grid                | mol m-2 s-1 | time, latitude, longitude | Prior total `<species>` fluxes on the reduced inversion grid
| flux_total_posterior_inversion_grid            | mol m-2 s-1 | time, latitude, longitude | Posterior total `<species>` fluxes on the reduced inversion grid
| stdev_flux_total_prior_inversion_grid          | mol m-2 s-1 | time, latitude, longitude | Standard deviation of prior total `<species>` fluxes on the reduced inversion grid
| stdev_flux_total_posterior_inversion_grid      | mol m-2 s-1 | time, latitude, longitude | Standard deviation of posterior total `<species>` fluxes on the reduced inversion grid
| *Alternative to stdev for non-Gaussian PDFs*
| percentile_flux_total_prior                    | mol m-2 s-1 | time, percentile, latitude, longitude | Percentile of prior total `<species>` fluxes
| percentile_flux_total_posterior                | mol m-2 s-1 | time, percentile, latitude, longitude | Percentile of posterior total `<species>` fluxes
| percentile_flux_total_prior_inversion_grid     | mol m-2 s-1 | time, percentile, latitude, longitude | Percentile of prior total `<species>` fluxes on the reduced inversion grid
| percentile_flux_total_posterior_inversion_grid | mol m-2 s-1 | time, percentile, latitude, longitude | Percentile of posterior total `<species>` fluxes on the reduced inversion grid

(1) fluxy accepts any SI unit of the type "amount length-2 time-1" and "mass length-2 time-1". However, please make sure that all grid variables have the same units.

| By-country variables                    | Units (2) | Dimensions    | Description                                      |
|:----------------------------------------|:----------|:--------------|:-------------------------------------------------|
| flux_total_prior_country                | kg yr-1   | time, country | Country-total prior `<species>` fluxes
| flux_total_posterior_country            | kg yr-1   | time, country | Country-total posterior `<species>` fluxes
| stdev_flux_total_prior_country          | kg yr-1   | time, country | Standard deviation of country-total prior `<species>` fluxes
| stdev_flux_total_posterior_country      | kg yr-1   | time, country | Standard deviation of country-total posterior `<species>` fluxes
| *Alternative to stdev for non-Gaussian PDFs*
| percentile_flux_total_prior_country     | kg yr-1   | time, percentile, country | Percentiles of country-total prior `<species>` fluxes
| percentile_flux_total_posterior_country | kg yr-1   | time, percentile, country | Percentiles of country-total posterior `<species>` fluxes
| *Optional*
| covariance_flux_total_posterior_country | kg2 yr-2  | time, country, country    | Covariance of country-total posterior `<species>` fluxes

(2) fluxy accepts any SI unit of the type "mass time-1" for country flux variables and "mass2 time-2" for the covariance variable. However, please make sure that all by-country variables have the same units and that the covariance variable has the respective squared units.

| Auxiliary variables | Units  |  Dimensions                  | Description                                |
|:--------------------|:-------|:-----------------------------|:-------------------------------------------|
| country_fraction    | -      | country, latitude, longitude | Fraction of grid cell associated to country
| cell_area           | m2     | latitude, longitude          | Surface area of gird cell

## 2. Concentration file

| Characterising variables | Units                          | Dimensions | Description                                      |
|:-------------------------|:-------------------------------|:-----------|:-------------------------------------------------|
| longitude                | degrees_east                   | index      | Sample longitude in decimal degrees
| latitude                 | degrees_north                  | index      | Sample latitude in decimal degrees
| time                     | days since 1970-01-01 00:00:00 | index      | Time of mid of observation interval in UTC
| time_bnds                | days since 1970-01-01 00:00:00 | index      | Start and end points of each time step
| altitude                 | m                              | index      | Sample altitude in meters above sea level
| number_of_identifier     | -                              | index      | Index of identifier of observing platform
| assimilation_flag        | -                              | index      | Flag indicating whether observation was used in inversion/assimilation (0: not used; 1: used)

| Observation variables | Units (3) | Dimensions | Description                                      |
|:----------------------|:----------|:-----------|:-------------------------------------------------|
| platform              | -         | platform   | Identifier of observing platform
| mf_observed           | mol mol-1 | index      | Observed mole fraction of `<species>` in dry air

| Simulated variables     | Units (3) | Dimensions | Description                                      |
|:------------------------|:----------|:-----------|:-------------------------------------------------|
| mf_prior                | mol mol-1 | index      | Prior simulated mole fraction of `<species>` in dry air
| mf_posterior            | mol mol-1 | index      | Posterior simulated mole fraction of `<species>` in dry air
| mf_bc_prior             | mol mol-1 | index      | Prior simulated boundary condition mole fraction including site bias
| mf_bc_posterior         | mol mol-1 | index      | Posterior simulated boundary condition mole fraction including site bias
| *Optional*
| stdev_mf_prior          | mol mol-1 | index      | Standard deviation of prior simulated mole fractions due to state vector uncertainty
| stdev_mf_posterior      | mol mol-1 | index      | Standard deviation of posterior simulated mole fractions due to state vector uncertainty
| mf_bias_prior	          | mol mol-1 | index      | Prior simulated mole fraction site bias
| mf_bias_posterior       | mol mol-1 | index      | Posterior simulated mole fraction site bias
| mf_outer_prior          | mol mol-1 | index      | Prior simulated mole fraction contribution from distant regions
| mf_outer_posterior      | mol mol-1 | index      | Posterior simulated mole fraction contribution from distant regions
| *Alternative to stdev for non-Gaussian PDFs*
| percentile_mf_prior     | mol mol-1 | index, percentile | Percentile of prior simulated mole fraction due to state vector uncertainty uncertainty
| percentile_mf_posterior | mol mol-1 | index, percentile | Percentile of posterior simulated mole fraction due to state vector uncertainty

| Uncertainty variables           | Units (3) | Dimensions | Description                                      |
|:--------------------------------|:----------|:-----------|:-------------------------------------------------|
| stdev_mf_total                  | mol mol-1 | index      | Total model-data-mismatch uncertainty applied in inversion
| *Optional*
| stdev_mf_observed_repeatability | mol mol-1 | index      | Repeatability uncertainty of observed mole fraction
| stdev_mf_observed_variability	  | mol mol-1 | index      | Variability of observed mole fraction within aggregation interval
| stdev_mf_model                  | mol mol-1 | index      | Model uncertainty of simulated mole fraction

(3) fluxy also accepts ppm, ppb and ppt. However, please make sure that all variables have the same units.
