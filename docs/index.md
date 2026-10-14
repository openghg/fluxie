# Flux Intercomparison Environment (fluxie)

**fluxie** is a Python package to compare greenhouse gas fluxes and mole fractions
from atmospheric inverse models.
It reads model outputs written in a common netCDF format, processes them
(selection, unit conversion, resampling, regional aggregation, statistics) and
produces publication-ready figures.

fluxie was born (as *fluxy*) during the [PARIS project](https://horizoneurope-paris.eu/)
to compare the results of the European inversion systems, and is now used in
several intercomparison projects.

<div class="grid cards" markdown>

-   :material-download: **[Installation](installation.md)**

    Install fluxie on your machine or on the ICOS Jupyter Hub.

-   :material-rocket-launch: **[Quick start](quickstart.md)**

    Make your first plots with the test data.

-   :material-image-multiple: **[Gallery](gallery.md)**

    All the available plots, with the code to produce them.

-   :material-book-open-variant: **[User guide](user_guide/principles.md)**

    How fluxie works and how to format your model outputs.

-   :material-api: **[API reference](api/index.md)**

    Documentation of all functions.

-   :material-account-group: **[Contributing](developer/contributing.md)**

    Add your own plots and improve fluxie.

</div>

## What can fluxie plot?

- Timeseries of country and regional total fluxes, with uncertainties and bottom-up inventories
- Sectoral breakdown of country fluxes
- Maps of posterior and prior fluxes, and comparisons between models, groups of models and periods
- Observed and modelled mole fraction timeseries at measurement sites
- Statistics and Taylor diagrams of modelled vs observed mole fractions
- Eddy covariance fluxes compared with inventories

## Citing fluxie

If you use fluxie in your research, please [cite it](citing.md) and add your
publication to the [Hall of fame](hall_of_fame.md#add-your-publication).
