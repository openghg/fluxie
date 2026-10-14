# Quick start

The easiest way to try fluxie is to run the example notebook
[`scripts/example_basics.ipynb`](https://github.com/openghg/fluxie/blob/devel/scripts/example_basics.ipynb) from the repository.
It uses the test data in `data/tests` and goes through the main plots.
See [Notebooks](user_guide/notebooks.md) for the other example notebooks.

## A first plot

The following script reads the flux outputs of three test models and plots
their country fluxes.

```python
from fluxie.config import set_model_colors, set_model_labels
from fluxie.io import read_config_files, read_model_output
from fluxie.operators.select import slice_flux
from fluxie.plots.flux_timeseries import plot_country_flux
from fluxie.test_utils import data_dir

species = "hfc134a"
models = [
    "InTEM_NAME_EUROPE_EDGAR_old_format",
    "ELRIS_NAME_EUROPE_EDGAR",
    "RHIME_NAME_EUROPE_EDGAR_old_format",
]

# 1. Read the configuration and the model outputs
config_data = read_config_files()
ds_all_flux = read_model_output(
    data_dir, "flux", species, models, config_data, period="yearly"
)

# 2. Select the period of interest and convert the units
ds_all_flux = slice_flux(
    ds_all_flux,
    config_data,
    "2018-01-01",
    "2024-01-01",
    species=species,
    country_flux_units_print="Gg yr-1",
)

# 3. Plot
fig = plot_country_flux(
    ds_all_flux,
    species,
    plot_regions=["GERMANY", "UK"],
    config_data=config_data,
    model_colors=set_model_colors(models),
    model_labels=set_model_labels(models, config_data, False),
)
fig.savefig("country_flux.png")
```

## Next steps

- Browse the [gallery](gallery.md) to see all the available plots.
- Read [How fluxie works](user_guide/principles.md) to understand the workflow.
- Format your own model outputs following [Input data format](user_guide/data_format.md).
- Use the [analysis notebooks](user_guide/notebooks.md) to compare your models.
