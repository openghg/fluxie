# API reference

The API reference is generated from the docstrings of the fluxie functions.

| Section | Content |
|:--|:--|
| [Plots](plots.md) | Functions producing figures: flux timeseries, flux maps, mole fraction timeseries, statistics, eddy covariance fluxes. |
| [Input/Output](io.md) | Reading model outputs, configuration files and templates. |
| [Operators](operators.md) | Selection, unit conversion, resampling, regions, sectors and statistics on the datasets. |
| [Configuration](config.md) | Plot styles, model colors and labels. |

A typical workflow chains these modules: read with [`fluxie.io`](io.md),
prepare with [`fluxie.operators`](operators.md), then plot with
[`fluxie.plots`](plots.md).
See [How fluxie works](../user_guide/principles.md) for details.
