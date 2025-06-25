import math
import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter
from matplotlib.dates import YearLocator, MonthLocator

from fluxy import config
from fluxy.operators.regions import extract_region_flux
from fluxy.operators.rolling_mean import calc_rolling_mean
from fluxy.operators.flux_timeseries_resample import resample_flux
from fluxy.operators.flux_combine import combine_dataset
from fluxy.operators.flux_prepare_inventory import retrieve_inventories
from fluxy.plots.utils import update_list_params

logger = logging.getLogger(__name__)


def determine_subplots_arrangement(subplot_number: int) -> tuple[int, int]:
    """
    Determine number of columns and rows for the figure given the number of subplots to make.
    Args:
        subplot_number: number of subplots to make.
    Returns:
        n_cols,n_rows: number of columns and rows for the figure.
    """
    if subplot_number < 4:
        n_cols = subplot_number
        n_rows = 1
    elif subplot_number == 4:
        n_cols = 2
        n_rows = 2
    elif subplot_number in [5, 6]:
        n_cols = 3
        n_rows = 2
    elif subplot_number > 6:
        n_cols = 4
        n_rows = math.ceil(subplot_number / 4)
    return n_cols, n_rows


def prepare_data_to_plot(
    ds_all_region: dict[str, xr.Dataset],
    model_labels: dict[str, str],
    model_colors: dict[str, list],
    plot_separate: bool | list[bool] = True,
    plot_combined: bool | list[bool] = False,
    resample: str | list[str] | None = None,
    rolling_mean: bool = False,
    resample_uncert_correlation: bool = False,
    plot_resample_and_original: bool = False,
) -> dict[str, xr.Dataset]:
    """
    Create a single xarray dataset for each set of data to be plotted.

    Args:
        ds_region: xarray datasets of fluxes, scaled and sliced between
            chosen dates.
        plot_separate: If True, plots model result as separate line. List must be of same size as models, e.g. [True, False, False].
            If a single boolean is provided, the same flag is assumed for all models.
        plot_combined: If True, the model is included in combined average result to be plotted. List must be of same size as models, e.g. [False, True, True].
            If a single boolean is provided, the same flag is assumed for all models.
        resample: Option to be passed to resample built-in function of xarray Dataset. For yearly average, 'YS' option should be used; 'QS-DEC' for seasonaly average.
            See http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
        rolling_mean : If True, calculates a rolling mean (xx years) for each of the data to plot.
        resample_uncert_correlation: If True, calculates the resampled uncertainty as the mean from all averaged periods.
            If False, recalculates uncertainty assuming no correlation between all averaged periods, by taking the square root of the summed variances, divided by the number of averaging periods.
        plot_resample_and_original: If True, plots both the resampled data and the data as its original frequency. If False, only plots the resampled data.

    Returns:
        ds_to_plot : dictionnary of datasets to plot
    """

    # Convert some inputs to list and check there size
    plot_separate, plot_combined, resample = update_list_params(
        [plot_separate, plot_combined, resample],
        expected_size=len(ds_all_region.keys()),
    )

    model_colors = model_colors.copy()
    # Assign default color list and label to input dataset

    for m in ds_all_region.keys():
        ds_all_region[m].attrs["model_label"] = model_labels.get(m, m)
        if m not in model_colors.keys():
            model_colors[m] = config.get_default_colors()
        ds_all_region[m].attrs["model_colors"] = model_colors[m] 
    map_model_colors = {f"c{i}": m for i, m in enumerate(model_colors.values())}

    # Prepare list of dataset to plot
    ds_to_plot = dict()

    if not any(resample) or plot_resample_and_original:
        ds_original_flux = {
            m: v for (i, (m, v)) in enumerate(ds_all_region.items()) if plot_separate[i]
        }
        ds_to_plot.update(ds_original_flux)

    if any(resample):
        ds_resampled = resample_flux(
            ds_all_region, resample, resample_uncert_correlation
        )
        ds_to_plot.update(
            {
                m: v
                for (i, (m, v)) in enumerate(ds_resampled.items())
                if plot_separate[i]
            }
        )

    if any(plot_combined):
        if all([resamp for comb, resamp in zip(plot_combined, resample) if comb]):
            ds_combined = combine_dataset(ds_resampled, plot_combined)
            ds_combined["combined"].attrs[
                "model_label"
            ] = "PARIS mean (from resampled data)"
        else:
            ds_combined = combine_dataset(ds_all_region, plot_combined)
            ds_combined["combined"].attrs["model_label"] = "PARIS mean"
        ds_to_plot.update(ds_combined)

    if rolling_mean:
        ds_to_plot = {m: calc_rolling_mean(ds) for m, ds in ds_to_plot.items()}

    # Determine plot color and label of each dataset
    color_usage = {k: 0 for k in map_model_colors.keys()}
    for m in ds_to_plot.keys():
        if m == "combined":
            include_label = "PARIS mean"
            model_color = "black"
        else:
            include_label = ds_to_plot[m].attrs.get("model_label", None)
            key_mc = [
                k
                for k in map_model_colors.keys()
                if map_model_colors[k] == ds_to_plot[m].attrs["model_colors"]
            ][0]
            nb = color_usage[key_mc]
            model_color = map_model_colors[key_mc][nb % len(map_model_colors[key_mc])]
            color_usage[key_mc] = color_usage[key_mc] + 1

        if "_resample" in m:
            include_label += " (resampled)"

        ds_to_plot[m].attrs["model_label"] = include_label
        ds_to_plot[m].attrs["model_color"] = model_color

    return ds_to_plot


def plot_country_flux(
    ds_all: dict[str, xr.Dataset],
    species: str,
    plot_regions: list[str] = [],
    config_data: dict[str, dict] = {},
    model_colors: dict[str, str] = {},
    model_labels: dict[str, str] = {},
    start_date: str | None = None,
    end_date: str | None = None,
    annex_mode: bool = False,
    plot_inventory: bool = True,
    inventory_years: list[str] | None = None,
    data_dir: str | None = None,
    fix_y_axes: bool = False,
    add_prior: bool = True,
    add_prior_unc: bool = False,
    set_global_leg: bool = False,
    country_codes_as_titles: bool = False,
    plot_separate: bool | list[bool] = True,
    plot_combined: bool | list[bool] = False,
    resample: str | list[str] | None = None,
    resample_uncert_correlation: bool = False,
    plot_resample_and_original: bool = False,
    return_res: bool = False,
    rolling_mean: bool = False,
) -> Figure | list:
    """
    Timeseries plot of prior and posterior country fluxes, from list of
    areas in plot_regions.

    Args:
        ds_all: xarray datasets of fluxes, scaled and sliced between
            chosen dates.
        species: Gas species, e.g. 'ch4'.
        plot_regions: Country or regions to plot, e.g. ['UNITED KINGDOM','SWITZERLAND']
        config_data: Dictionary with settings read from json file. Use json filenames as keys.
        model_colors: Models and corresponding colours used to plot the model.
        start_date: Start dates of the data to plot (used to slice inventory data).
        end_date: Start dates of the data to plot (used to slice inventory data).
        annex_mode: If True, replace the labels with more concise versions for National Inventory Report Annexes.
        scale_co2eq: If True, adapt y-axis label to CO2-eq.
        plot_inventory: If True, plots inventory flux estimates as bars in each plot.
        inventory_years: List of inventory data from different years to include. If None, only plots the most recent inventory data.
        data_dir: Path to top data directory, used to read inventory data files.
        fix_y_axes: If True, uses a consistent y axis for all plots.
        add_prior: If True, plots prior as dashed lines.
        add_prior_unc: If True, plots prior uncertainty as shaded area.
        set_global_leg: If True, plots one single legend instead of one legend per subplot.
        country_codes_as_titles: If True, uses list of country codes as titles, instead of the region names.
        plot_separate: If True, plots model result as separate line. List must be of same size as models, e.g. [True, False, False].
            If a single boolean is provided, the same flag is assumed for all models.
        plot_combined: If True, the model is included in combined average result to be plotted. List must be of same size as models, e.g. [False, True, True].
            If a single boolean is provided, the same flag is assumed for all models.
        resample: Option to be passed to resample built-in function of xarray Dataset. For yearly average, 'YS' option should be used; 'QS-DEC' for seasonaly average.
            See http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
        resample_uncert_correlation: If True, calculates the resampled uncertainty as the mean from all averaged periods.
            If False, recalculates uncertainty assuming no correlation between all averaged periods, by taking the square root of the summed variances, divided by the number of averaging periods.
        plot_resample_and_original: If True, plots both the resampled data and the data as its original frequency. If False, only plots the resampled data.
        period_override: Inversion periods to include, to override the standards in species_info.json. Must be the same length as models, e.g. ['monthly',None,'yearly']
        return_res: Wheter or not including a dictionnary with the results as output
        rolling_mean : If True, calculates a rolling mean (xx years) for each of the data to plot.
    Returns:
        fig: A plot per country/region.
        res_dict : If return_res, return also a dictionnary containaing the plotted results

    """
    s_data = config_data.get("species_info", {})
    r_data = config_data.get("regions_info", {})

    if not plot_regions:
        # Read all countries given in the dss and take the intersection of models
        plot_regions = set.intersection(
            *(set(ds["country"].values) for ds in ds_all.values())
        )

    if return_res:
        res_dict: dict[str, dict] = {country: dict() for country in plot_regions}

    max_cf = np.zeros(len(plot_regions))
    min_x, max_x = np.datetime64("2100-01-01", "D"), np.datetime64("1900-01-01", "D")
    linewidth, alpha = (1.0, 0.7) if annex_mode else (1.5, 1.0)

    # Create figure
    n_cols, n_rows = determine_subplots_arrangement(len(plot_regions))

    if "all" in species:
        units = {ds.posterior.units for ds in ds_all.values()}
    else:
        units = {ds["flux_total_posterior_country"].units for ds in ds_all.values()}
    if len(units) == 1:
        unit = list(units)[0]
    else:
        raise ValueError(
            f"Inconsistency in the units from the different datasets : {units} are present. Only one is expected."
        )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        sharex=True,
        constrained_layout=True,
        figsize=(n_cols * 6, n_rows * 4),
    )
    for i, country in enumerate(plot_regions):

        ax = axes if (n_rows, n_cols) == (1, 1) else axes.flatten()[i]

        if plot_inventory:
            inventories_to_plot = retrieve_inventories(
                data_dir,
                country,
                species,
                start_date,
                end_date,
                unit,
                s_data,
                r_data,
                inventory_years,
            )
            for i_inv, inventory in enumerate(inventories_to_plot):
                ax.bar(
                    inventory.time,
                    inventory,
                    np.timedelta64(340 - i_inv * 20, "D"),
                    edgecolor=inventory.plot_color,
                    align="edge",
                    fill=False,
                    label=f"Inventory {inventory.year}",
                    zorder=0,
                )
                if return_res:
                    res_dict[country][f"inventory_{inventory.year}"] = {
                        "time": inventory.time.values,
                        "value": inventory.values,
                    }
                min_x = min(inventory.time.min(skipna=True), min_x)
                max_x = max(inventory.time.max(skipna=True), max_x)
                max_cf[i] = np.nanmax((max_cf[i], inventory.max(skipna=True)))

        ds_all_region = extract_region_flux(ds_all, country, r_data)
        ds_to_plot = prepare_data_to_plot(
            ds_all_region,
            model_labels,
            model_colors,
            plot_separate=plot_separate,
            plot_combined=plot_combined,
            resample=resample,
            rolling_mean=rolling_mean,
            plot_resample_and_original=plot_resample_and_original,
            resample_uncert_correlation=resample_uncert_correlation,
        )

        for m, ds_region in ds_to_plot.items():

            ax.plot(
                ds_region.time,
                ds_region.posterior,
                label=ds_region.attrs["model_label"],
                color=ds_region.attrs["model_color"],
            )
            ax.fill_between(
                ds_region.time,
                ds_region.posterior_lower,
                ds_region.posterior_upper,
                alpha=0.3,
                color=ds_region.attrs["model_color"],
            )
            max_cf[i] = np.nanmax(
                (
                    max_cf[i],
                    ds_region.posterior_upper.max(skipna=True),
                    ds_region.posterior.max(skipna=True),
                )
            )
            min_x = min(ds_region.time.min(skipna=True), min_x)
            max_x = max(ds_region.time.max(skipna=True), max_x)

            if return_res:
                res_dict[country][m] = {
                    "time": ds_region.time.values.astype("datetime64[ns]"),
                    "mean": ds_region.posterior.values,
                    "min": ds_region.posterior_lower.values,
                    "max": ds_region.posterior_upper.values,
                }

            if add_prior:
                ax.plot(
                    ds_region.time,
                    ds_region.prior,
                    label=ds_region.attrs["model_label"] + " prior",
                    color=ds_region.attrs["model_color"],
                    linestyle="dashed",
                    linewidth=linewidth,
                    alpha=alpha,
                )
                max_cf[i] = np.nanmax((max_cf[i], ds_region.prior.max(skipna=True)))

            if add_prior_unc:
                ax.fill_between(
                    ds_region.time,
                    ds_region.prior_lower,
                    ds_region.prior_upper,
                    alpha=0.1,
                    color=ds_region.attrs["model_color"],
                )
                max_cf[i] = np.nanmax(
                    (max_cf[i], ds_region.prior_upper.max(skipna=True))
                )

        ax.set_ylabel(
            f"{s_data.get(species, {}).get('species_print', species)}"
            f" ({unit.replace('2','$_{{2}}$').replace('-1','$^{{-1}}$')})"
        )

        # set legend if needed
        if not set_global_leg:
            ncol = 3 if annex_mode else 2
            leg = ax.legend(ncol=ncol, borderpad=0.4, columnspacing=1.0)
            for l in leg.legend_handles[: (-1 if plot_inventory else None)]:
                l.set_linewidth(3.0)

        # set title
        country_equivalent = {
            "NW_EU2": "NW EUROPE",
            "CW_EU": "CENTRAL W EUROPE",
            "NW_EU_CONTINENT": "NW CONTINENTAL EUROPE",
        }
        print_country = country_equivalent.get(country, country)

        if country_codes_as_titles and country in r_data["regions"].keys():
            ax.set_title(f'{print_country}\n{r_data["regions"][country]}')
        else:
            ax.set_title(f"{print_country}")

        # set grid
        ax.grid(visible=True, which="major", alpha=0.4)

    # set xticks
    year_range = max_x.dt.year.values - min_x.dt.year.values
    min_x, max_x = min_x.values, max_x.values
    if "yearly" in [ds.attrs["frequency"] for ds in ds_to_plot.values()] \
        or resample == "year":
        min_x = min_x.astype("datetime64[Y]")
        max_x = max_x.astype("datetime64[Y]") + np.timedelta64(1, "Y")
    xlim = [min_x - (max_x-min_x)/50, max_x + (max_x-min_x)/50]

    # if max_x - min_x > np.timedelta64(8, "Y"):
    if year_range > 8:
        max_x = max_x.astype("datetime64[Y]")
        min_x = min_x.astype("datetime64[Y]")
        step = int(year_range) // 8 + 1
        xticks = np.arange(min_x, max_x, step=np.timedelta64(step, "Y"))
        if (max_x-min_x)%np.timedelta64(step, "Y")==0:
            xticks = np.append(xticks,max_x)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks.astype("datetime64[Y]"))
        ax.xaxis.set_major_locator(YearLocator())

    else:
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_major_locator(YearLocator())

    ax.set_xlim(xlim)

    if set_global_leg:
        ncol = 0
        if plot_separate or resample:
            ncol = len(ds_all.keys())
        if plot_combined and plot_separate:
            ncol = math.floor(len(ds_all.keys()) / 2) + 2
        elif plot_combined:
            ncol = 3
        if plot_inventory:
            ncol = ncol + 1

        if n_rows > 1:
            legend_loc = (0.5, 1.1)
        else:
            legend_loc = (0.5, 1.15)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=ncol,
            borderpad=0.4,
            columnspacing=1.0,
            bbox_to_anchor=legend_loc,
        )

    fac = 1.1 if set_global_leg else 1.2

    # loop through plots again to fix min/max y-axis values
    for i, country in enumerate(plot_regions):
        if fix_y_axes == True:
            fig.axes[i].set_ylim([0, np.nanmax(max_cf) * fac])
        elif type(fix_y_axes) == list:
            fig.axes[i].set_ylim(fix_y_axes)
        elif fix_y_axes == False:
            fig.axes[i].set_ylim([0, max_cf[i] * fac])

    logger.info(
        "NOTE: If all the data is not within axis limits, adjust the set_ylim parameter"
    )

    if return_res:
        return fig, res_dict
    else:
        return fig
