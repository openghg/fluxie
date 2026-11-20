# Operators

Functions that select, convert, resample and combine the model datasets before plotting.

## Selection

::: fluxie.operators.select.slice_flux

::: fluxie.operators.select.slice_mf

::: fluxie.operators.select.slice_site

::: fluxie.operators.select.slice_height

::: fluxie.operators.select.get_site_index

::: fluxie.operators.select.get_unique_sites

::: fluxie.operators.select.get_intake_height

::: fluxie.operators.select.get_unique_site_height_pairs

::: fluxie.operators.select.clean_timeseries_missing_data

::: fluxie.operators.select.check_site_list

## Conversion and scaling

::: fluxie.operators.convert.scale_variables

::: fluxie.operators.convert.get_variables

::: fluxie.operators.convert.get_unit_type_and_conversion_to_base

::: fluxie.operators.convert.get_units_conversion_factor

::: fluxie.operators.convert.convert_units_co2eq

::: fluxie.operators.flux_scale_by_sector_proportions.open_and_align_sector_dataset

::: fluxie.operators.flux_scale_by_sector_proportions.create_cell_area

::: fluxie.operators.flux_scale_by_sector_proportions.scale_by_sector_proportions

## Resampling and alignment

::: fluxie.operators.flux_align_dataset.align_time

::: fluxie.operators.flux_align_dataset.align_lat_lon

::: fluxie.operators.flux_align_dataset.align_map_data

::: fluxie.operators.flux_map_resample.resample_over_period

::: fluxie.operators.flux_map_resample.average_over_period

::: fluxie.operators.flux_timeseries_resample.resample_flux

::: fluxie.operators.rolling_mean.calc_rolling_mean

## Regions, sectors and statistics

::: fluxie.operators.regions.extract_region_flux

::: fluxie.operators.regions.extract_region_inventory_flux

::: fluxie.operators.regions.format_plot_regions

::: fluxie.operators.sectors.sectors_group_from_config_or_dict

::: fluxie.operators.sectors.group_sectors

::: fluxie.operators.mf.compute_mf_difference

::: fluxie.operators.mf.stats_mf

::: fluxie.operators.stats.stats_observed_vs_simulated

## Combination and difference utilities

::: fluxie.operators.flux_combine.combine_dataset

::: fluxie.operators.flux_combine.combine_map_dataset

::: fluxie.operators.flux_map_diff.define_var_plot

::: fluxie.operators.flux_map_diff.make_model_diff_ds

::: fluxie.operators.flux_prepare_inventory.retrieve_inventories

::: fluxie.operators.ecflux.filter_ecflux
