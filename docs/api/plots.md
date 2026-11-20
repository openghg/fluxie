# Plot functions

Every plot in fluxie is produced by a single function that takes a dictionary of
model datasets (see [How fluxie works](../user_guide/principles.md)) and returns a
matplotlib figure.

!!! tip
    The [gallery](../gallery.md) shows the output of these functions
    on the test data, together with the code used to produce them.

## Flux timeseries

[Gallery examples](../gallery.md#flux-timeseries)

::: fluxie.plots.flux_timeseries.plot_country_flux

::: fluxie.plots.flux_timeseries.plot_country_sector_flux_bar

::: fluxie.plots.flux_timeseries.plot_all_species_stacked_bar

## Flux maps

[Gallery examples](../gallery.md#flux-maps)

::: fluxie.plots.flux_map.plot_flux_map

::: fluxie.plots.flux_map.plot_flux_map_model_comparison

::: fluxie.plots.flux_map.plot_flux_map_over_time

::: fluxie.plots.flux_map.plot_flux_map_combined_models_comparison

::: fluxie.plots.flux_map.plot_flux_map_period_comparison

## Mole fraction timeseries

[Gallery examples](../gallery.md#mole-fraction-timeseries)

::: fluxie.plots.mf_timeseries.plot_timeseries

::: fluxie.plots.mf_timeseries.plot_sites_timeseries

::: fluxie.plots.mf_timeseries.plot_sites_list_mf

::: fluxie.plots.mf_timeseries.plot_histogram

::: fluxie.plots.mf_timeseries.plot_mf_timeseries

## Statistics

[Gallery examples](../gallery.md#statistics)

::: fluxie.plots.mf_stats.plot_stats_mf

::: fluxie.plots.mf_stats.plot_taylor_diagram

::: fluxie.plots.stats.plot_stats

## Correlation

::: fluxie.plots.correlation.plot_correlation

## Eddy covariance fluxes

[Gallery examples](../gallery.md#eddy-covariance-fluxes)

::: fluxie.plots.ec_flux.sectorial_stack.plot_stacked
