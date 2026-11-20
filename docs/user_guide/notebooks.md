# Notebooks

The [`scripts/`](https://github.com/openghg/fluxie/tree/devel/scripts) folder of
the repository contains notebooks showing how to use fluxie. They are a good
starting point for your own analysis.

| Notebook | Description |
|:--|:--|
| [`example_basics.ipynb`](https://github.com/openghg/fluxie/blob/devel/scripts/example_basics.ipynb) | Tutorial: main plots, using the test data. |
| [`example_ecflux.ipynb`](https://github.com/openghg/fluxie/blob/devel/scripts/example_ecflux.ipynb) | Tutorial: eddy covariance fluxes, using the test data. |
| [`PARIS_inversion_results.ipynb`](https://github.com/openghg/fluxie/blob/devel/scripts/PARIS_inversion_results.ipynb) | Full intercomparison of inversion results, as used in the PARIS project. |

## Running the intercomparison notebook

The notebook `scripts/PARIS_inversion_results.ipynb` lets you plot all the
variables of interest for your model runs.

1. In the first notebook cell, specify the path to the data and the `experiments`
   dictionary, which points to the model runs you want to plot.

    The model runs are identified by their name tags:
    `<inversionModel>_<optional_identifying_tags>` (e.g. `"InTEM_NAME_EUROPE_EDGAR"`).

    `<species>` and `<inversionFrequency>` are not part of the model run name
    because they are set in dedicated variables of the notebook.
    If the sequence of name tags is too long, a shorter tag can be defined in
    `filename_tags` in `models_info.json` (see [Configuration files](configuration.md#models-information)).

2. The notebook is organized in 3 numbered sections. At the top of each section,
   specify the models you want to plot, the species, start/end dates, etc.
   (plotting options are described next to each variable).
   Run the top cell to read in the data and select values between the chosen dates.

3. In the following cells of each section, edit the plotting options to your
   preference and run the cells to produce the plots.

!!! tip
    Set `presentation_mode = True` for bigger fonts (slides), and `False` for
    smaller fonts (text documents).
