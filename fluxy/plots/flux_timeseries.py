import math
import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from typing import Tuple

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.dates import YearLocator, MonthLocator
from matplotlib import __version__ as mplt_version

from fluxy import config
from fluxy.operators.regions import extract_region_flux
from fluxy.operators.rolling_mean import calc_rolling_mean
from fluxy.operators.flux_timeseries_resample import resample_flux
from fluxy.operators.flux_combine import combine_dataset
from fluxy.operators.flux_prepare_inventory import retrieve_inventories
from fluxy.plots.utils import update_list_params

logger = logging.getLogger(__name__)


def format_plot_regions(
    plot_regions: str | list[str] | None, ds_all: dict[str, xr.Dataset] | None
) -> list[str]:
    """
    Format plot_regions into a list of regions. If plot_regions originally None, read the country names from ds_all.
    Args:
        plot_regions: (list of) regions
        ds_all: dictionnary containing dataset form which the regions will be determine if plot_regions=None.
    Returns
        plot_regions: list of regions
    """

    if not plot_regions:
        # Read all countries given in the dss and take the intersection of models
        plot_regions = set.intersection(
            *(set(ds["country"].values) for ds in ds_all.values())
        )
    if not isinstance(plot_regions, list):
        plot_regions = [plot_regions]

    return plot_regions


def get_unit(ds_all: dict[str, xr.Dataset]) -> str:
    """
    Determine unit of posterior estimations from datasets. If incoherencies between datasets, an error is raised.
    Args:
        ds_all: dictionnary of datasets from which read the units.
    Returns:
        unit: unit of posterior variables in dataset.
    """

    if all(["flux_total_posterior_country" in ds for ds in ds_all.values()]):
        units = {ds["flux_total_posterior_country"].units for ds in ds_all.values()}
    elif all(["posterior" in ds for ds in ds_all.values()]):
        units = {ds.posterior.units for ds in ds_all.values()}
    else:
        raise ValueError(
            "Did not find variable 'posterior' or 'flux_total_posterior_country' in every dataset. Thus couldn't determine unit."
        )

    if len(units) == 1:
        unit = list(units)[0]
    else:
        raise ValueError(
            f"Inconsistency in the units from the different datasets : {units} are present. Only one is expected."
        )

    return unit


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


def create_fig_and_axes(nb_subplots: int) -> Tuple[Figure, Axes]:
    """
    Create matplotib figure and axes object based on the number of subplots asked for.
    Args:
        nb_subplots: number of subplot wanted (should be the number of subregion to plot when used in plot_country_flux)
    Returns:
        fig, axes: fig and flattened axes
    """

    n_cols, n_rows = determine_subplots_arrangement(nb_subplots)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        sharex=True,
        constrained_layout=True,
        figsize=(n_cols * 6, n_rows * 4),
    )
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    return fig, axes


def prepare_data_to_plot(
    ds_all_region: dict[str, xr.Dataset],
    model_labels: dict[str, str],
    model_colors: dict[str, list],
    plot_separate: bool | list[bool] = True,
    plot_combined: bool | list[bool] = False,
    resample: str | list[str] | None = None,
    rolling_mean: bool | list[bool] = False,
    resample_uncert_correlation: bool = False,
    plot_resample_and_original: bool = False,
    aggreg_month: bool = False,
) -> dict[str, xr.Dataset]:
    """
    Create a single xarray dataset for each set of data to be plotted.

    Args:
        ds_all_region: xarray datasets of fluxes, scaled and sliced between
            chosen dates.
        model_labels: labels to associate to each dataset. should have the same keys as ds_all_region.
        model_colors: colors to associate to each dataset. should have the same keys as ds_all_region.
        plot_separate: If True, plots model result as separate line. List must be of same size as models, e.g. [True, False, False].
            If a single boolean is provided, the same flag is assumed for all models.
        plot_combined: If True, the model is included in combined average result to be plotted. List must be of same size as models,
            e.g. [False, True, True].
            If a single boolean is provided, the same flag is assumed for all models.
        resample: Option to be passed to resample built-in function of xarray Dataset. For yearly average, 'YS' option should be used;
            'QS-DEC' for seasonaly average.
            See http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
        rolling_mean : If True, calculates a rolling mean (xx years) for each of the data to plot.
        resample_uncert_correlation: If True, calculates the resampled uncertainty as the mean from all averaged periods.
            If False, recalculates uncertainty assuming no correlation between all averaged periods, by taking the square root of the summed
            variances, divided by the number of averaging periods.
        plot_resample_and_original: If True, plots both the resampled data and the data as its original frequency. If False, only plots the
            resampled data.
        aggreg_month: if True, plot the data aggregated by month. Used to study seasonnal cycle.

    Returns:
        ds_to_plot : dictionnary of datasets to plot
    """

    # Convert some inputs to list and check there size
    plot_separate, plot_combined, resample, rolling_mean = update_list_params(
        [plot_separate, plot_combined, resample, rolling_mean],
        expected_size=len(ds_all_region.keys()),
    )

    # Timely aggregate the data when necessary
    if aggreg_month:
        if any(resample):
            raise ValueError(
                f"`resample` and `aggreg_month` cannot be both set to True. Please set only one of them to True."
            )
        ds_all_region = {
            k: ds.groupby(f"time.month").mean().rename({"month": "time"})
            for k, ds in ds_all_region.items()
        }

    # Assign default color list and label to input dataset
    model_colors = model_colors.copy()
    for m in ds_all_region.keys():
        ds_all_region[m].attrs["model_label"] = model_labels.get(m, m)
        if m not in model_colors.keys():
            model_colors[m] = config.get_default_colors()
        ds_all_region[m].attrs["model_colors"] = model_colors[m]
    map_model_colors = {f"c{i}": m for i, m in enumerate(model_colors.values())}

    # Prepare list of dataset to plot
    ds_to_plot = dict()

    # Add original datasets to plot
    if not any(resample) or plot_resample_and_original:
        ds_original_flux = {
            m: v for (i, (m, v)) in enumerate(ds_all_region.items()) if plot_separate[i]
        }
        ds_to_plot.update(ds_original_flux)

    # Add resampled datasets to plot
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

    # Apply rolling mean when necessary
    for m, rm, ps, rs in zip(
        ds_all_region.keys(), rolling_mean, plot_separate, resample
    ):
        if rm & ps:  # if rolling_mean and plot_separate
            ds_to_plot[m] = calc_rolling_mean(
                ds_to_plot[m + "_resample"] if rs else ds_to_plot[m]
            )

    # Add combined dataset to plot
    if any(plot_combined):
        if all([resamp for comb, resamp in zip(plot_combined, resample) if comb]):
            ds_to_combine = {
                m: calc_rolling_mean(ds) if rm else ds
                for rm, (m, ds) in zip(rolling_mean, ds_resampled.items())
            }
            ds_combined = combine_dataset(ds_to_combine, plot_combined)
            ds_combined["combined"].attrs[
                "model_label"
            ] = "PARIS mean (from resampled data)"
        else:
            ds_to_combine = {
                m: calc_rolling_mean(ds) if rm else ds
                for rm, (m, ds) in zip(rolling_mean, ds_all_region.items())
            }
            ds_combined = combine_dataset(ds_to_combine, plot_combined)
            ds_combined["combined"].attrs["model_label"] = "PARIS mean"
        ds_to_plot.update(ds_combined)

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

        if ("_resample" in m) and plot_resample_and_original:
            include_label += " (resampled)"

        ds_to_plot[m].attrs["model_label"] = include_label
        ds_to_plot[m].attrs["model_color"] = model_color

    return ds_to_plot


def add_posterior_plot(
    ax: Axes, ds_region: xr.Dataset, highlighted_line: bool
) -> dict[str, dict]:
    """
    Plot the posterior data on the axis. The variable posterior of the dataset ds_region is plotted as a line (color and label found in the dataset
    attributes) and the uncertainty (variables lower_posterior, upper_posterior in the dataset) is plotted as a semi-transparent filled space.
    Axes:
        ax: axes on which to plot
        ds_region: dataset containing posterior data
        highlighted_line: if True, the linewidth is made bigger (3.0) than when False (1.5). Typicall used for the annexes to highlight the PARIS mean.
    Returns:
        res_dict: dictionnary containing posterior data plotted - 4 keys: "time", "mean", "min" (lower uncertainty) and "max"  (upper uncertainty).
    """

    linew = 3 if highlighted_line else 1.5

    ax.plot(
        ds_region.time,
        ds_region.posterior,
        label=ds_region.attrs["model_label"],
        color=ds_region.attrs["model_color"],
        linewidth=linew,
    )
    ax.fill_between(
        ds_region.time,
        ds_region.posterior_lower,
        ds_region.posterior_upper,
        alpha=0.2,
        color=ds_region.attrs["model_color"],
    )

    res_dict = {
        "time": ds_region.time.values.astype("datetime64[ns]"),
        "mean": ds_region.posterior.values,
        "min": ds_region.posterior_lower.values,
        "max": ds_region.posterior_upper.values,
    }

    return res_dict


def add_prior_plot(
    ax: Axes, ds_region: xr.Dataset, annex_mode: bool, add_prior_unc: bool
) -> dict[str, dict]:
    """
    Plot the variable prior of the dataset ds_region on the axis.
    Axes:
        ax: axes on which to plot
        ds_region: dataset containing prior data
        annex_bool: if True, the linewidth is made slighty smaller (1.0) and a transparency of 0.7 is applied to the prior uncertainty.
            If False, linewidth is set to standard value (1.5) and no transparency s applied to the prior uncertainty (alpha=1.0).
        add_prior_unc: if True add prior uncertainty on the plot as a semi-transparent filled space.
    Returns:
        res_dict: dictionnary containing prior data plotted - 2 keys: "time" and "mean", 4 if add_prior_unc: "min" and "max" added.
    """
    linewidth, alpha = (1.0, 0.7) if annex_mode else (1.5, 1.0)

    ax.plot(
        ds_region.time,
        ds_region.prior,
        label=ds_region.attrs["model_label"] + " prior",
        color=ds_region.attrs["model_color"],
        linestyle="dashed",
        linewidth=linewidth,
        alpha=alpha,
    )

    res_dict = {
        "time": ds_region.time.values.astype("datetime64[ns]"),
        "mean": ds_region.prior.values,
    }

    if add_prior_unc:
        ax.fill_between(
            ds_region.time,
            ds_region.prior_lower,
            ds_region.prior_upper,
            alpha=0.1,
            color=ds_region.attrs["model_color"],
        )
        res_dict["min"] = ds_region.prior_lower.values
        res_dict["max"] = ds_region.prior_upper.values

    return res_dict


def add_inventory_barplot(
    ax: Axes,
    data_dir: str,
    country: str,
    species: str,
    start_date: str,
    end_date: str,
    unit: str,
    s_data: dict,
    r_data: dict,
    inventory_years: list[str] | None,
    inventory_filename: str,
    sector: str | list[str],
) -> dict[str, dict]:
    """
    Retrieve and plot the inventories as bar plots. If multiple inventories are plotted, the older the inventory is, the smaller
    the used width of the bar is and the whiter is the grey of the bar.
    The inventory data will be looked for using {dat_dor}/inventory/{inventory_filename}_{species}_{inventory_year[x]}
    Args:
        ax: axis on which plot the inventory.
        data_dir: path to the inventory. Should contain a subdirectory named 'inventory'.
        country: country of which we want the inventory.
        species: species form which we want the inventory.
        start_date: earliest date of the inventory to use.
        end_date: earliest date of the inventory to use.
        unit: units in which to plot the inventory.
        s_data: dictionnary containing species data. See configs/species_infos.json.
        r_data: dictionnary containing ragions data. See configs/species_infos.json.
        inventory_years: list of years of publication of the inventory versions we want to use.
        inventory_filename: name of inventory file.
        sector: sector we want to plot.
    Returns:
        res_dict: dictionnary containing inventory data plotted - 2 keys: "time", "value".
    """
    res_dict = dict()

    if isinstance(start_date, list):
        start_date_inv = str(min([np.datetime64(date) for date in start_date]))
    else:
        start_date_inv = start_date
    if isinstance(end_date, list):
        end_date_inv = str(max([np.datetime64(date) for date in end_date]))
    else:
        end_date_inv = end_date
    inventories_to_plot = retrieve_inventories(
        data_dir,
        country,
        species,
        start_date_inv,
        end_date_inv,
        unit,
        s_data,
        r_data,
        inventory_years,
        inventory_filename,
        sector=sector,
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

        res_dict[f"inventory_{inventory.year}"] = {
            "time": inventory.time.values,
            "value": inventory.values,
        }
    return res_dict


def add_ylim(
    axes: list[Axes],
    plot_regions: list[str],
    res_dict: dict[str, dict],
    fix_y_axes: bool | list[float] | None,
    set_global_leg: bool,
):
    """
    Add limits of y axes based on results in res_dict, or the values given in fig_y_axes if it's a list.
    Args:
        axes: list of axes to set the ylim to.
        plot_regions: list of regions corresponding to the axes (should be the same length and order).
        res_dict: dictionnary containing the data plotted. Should have one key per regions plotted, the values being dictionnaries with 3 keys: "inventory", "posterior" and "prior";
            whose values are the output of add_inventory_barplot, add_posterior_plot, add_prior_plot). The data stored in them is used to infer the ylims.
        fix_y_axes: if list, use it as params to ax.set_ylim; if bool and True, all subplots have the same y lim (the max value that can be found in res_dict); else the max of the data
            plotted in each subplots is used.
        set_global_leg: if True (and thus one common legend is plotted for all subplots in add_legend), a zoom of only 1.1 is made on the ymax, else it is 1.2 to make space for the legend.
    """

    if isinstance(fix_y_axes, list):
        for ax in axes:
            ax.set_ylim(*fix_y_axes)
        return

    max_cf = []
    fac = 1.1 if set_global_leg else 1.2

    for ax, country in zip(axes, plot_regions):
        maxs_inventory = [
            np.nanmax(inv["value"])
            for inv in res_dict["inventory"].get(country, dict()).values()
        ]
        maxs_posterior = [
            np.nanmax(post["max"]) for post in res_dict["posterior"][country].values()
        ]
        maxs_prior = [
            np.nanmax(prior.get("max", prior.get("mean", np.nan)))
            for prior in res_dict["prior"][country].values()
        ]

        max_cf.append(np.nanmax([*maxs_inventory, *maxs_posterior, *maxs_prior]))

        if not fix_y_axes:
            ax.set_ylim(0, max_cf[-1] * fac)

    for ax in axes:
        if fix_y_axes:
            ax.set_ylim(0, np.nanmax(max_cf) * fac)


def add_ylabel(ax: Axes, s_data: dict[str, dict], species: str, sector: str, unit: str):
    """
    Add label to y axis.
    Args:
        ax: axis to set the y label to.
        s_data: dictionnary containing info for each species (like name to use for label). See configs/species_infos.json.
        species: name of the species.
        sector: name of sector plotted.
        unit: unit of data plotted.
    """
    ax.set_ylabel(
        f"{s_data.get(species, {}).get('species_print', species)} {sector.title() if sector != 'total' else ''}"
        f" ({unit.replace('2','$_{{2}}$').replace('-1','$^{{-1}}$')})"
    )


def add_xlims_and_ticks(
    ax: Axes, yearly_freq: bool, res_dict: dict[str, dict], aggreg_month: bool
):
    """
    Add x limits, ticks and ticks labels to matplotlib axes. Optimize them by looking at if they are monthly, yearly, or monthly aggregated, and covered time range.
    Args:
        ax: axis to set xlim and xticks to.
        yearly_freq: set to True if the data plotted have a yearly frequency.
        res_dict: dictionnary containing the data plotted. Should have one key per regions plotted, the values being dictionnaries with 3 keys: "inventory", "posterior" and "prior";
            whose values are the output of add_inventory_barplot, add_posterior_plot, add_prior_plot). The time data stored in them is used to infer the xlims.
        aggreg_month: if True, the data plotted are supposed to be a monthly aggregated so 12 stciks are created, whose labels are the 3 first letters of each month.
    """
    if aggreg_month:
        ax.set_xticks(np.arange(1, 13))
        ax.set_xticklabels(
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "may",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
        )
        return

    min_x, max_x = np.datetime64("2100-01-01", "D"), np.datetime64("1900-01-01", "D")
    for country in res_dict["posterior"].keys():
        for m in res_dict["posterior"][country].keys():
            post_time = res_dict["posterior"][country][m]["time"]
            prior_time = res_dict["prior"][country].get(m, {"time": [min_x, max_x]})[
                "time"
            ]
            min_x = np.nanmin([*post_time, *prior_time, min_x])
            max_x = np.nanmax([*post_time, *prior_time, max_x])

        if country in res_dict["inventory"].keys():
            for inv_year in res_dict["inventory"][country].values():
                min_x = np.nanmin([*inv_year["time"], min_x])
                max_x = np.nanmax([*inv_year["time"], max_x])

    # set xticks
    year_range = max_x.astype("datetime64[Y]") - min_x.astype("datetime64[Y]")
    if yearly_freq:
        min_x = min_x.astype("datetime64[Y]")
        max_x = max_x.astype("datetime64[Y]") + np.timedelta64(1, "Y")
    xlim = [min_x - (max_x - min_x) / 50, max_x + (max_x - min_x) / 50]

    if year_range > np.timedelta64(8, "Y"):
        max_x = max_x.astype("datetime64[Y]")
        min_x = min_x.astype("datetime64[Y]")
        step = int(year_range) // 8 + 1
        xticks: np.ndarray = np.arange(min_x, max_x, step=np.timedelta64(step, "Y"))
        if (max_x - min_x) % np.timedelta64(step, "Y") == 0:
            xticks = np.append(xticks, max_x)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks.astype("datetime64[Y]"))
        ax.xaxis.set_major_locator(YearLocator())

    else:
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_major_locator(YearLocator())

    ax.set_xlim(xlim)


def add_legend(
    fig: Figure, set_global_leg: bool, annex_mode: bool, plot_inventory: bool
):
    """
    Set legend.
    Args:
        fig: figure object for which we want to make the legend
        set_global_legend: if True, set one legend object for all subplots, else plot one legend per subplot
        annex_mode: if in annex_mode,
        plot_inventory: used only if set_global_legend is False. When True, avoid enhancing the width of the last object. Not sure that if it is really
            used and if we shouldn't suppress this.
    """

    if set_global_leg:

        if isinstance(fig.axes[0], list):
            legend_loc = (0.5, 1.1)
        else:
            legend_loc = (0.5, 1.15)
        handles, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=(
                len(labels) if len(labels) <= 6 else len(labels) // 2 + len(labels) % 2
            ),
            borderpad=0.4,
            columnspacing=1.0,
            bbox_to_anchor=legend_loc,
        )

    else:
        for ax in fig.axes:
            _, labels = ax.get_legend_handles_labels()
            ncol = len(labels) + 1 if annex_mode else 2
            leg = ax.legend(ncol=ncol, borderpad=0.4, columnspacing=1.0)

            handle_name = (
                "legend_handles"
                if int(mplt_version.split(".")[0]) >= 3
                and int(mplt_version.split(".")[1]) >= 7
                else "legendHandles"
            )
            for l in leg.__getattribute__(handle_name)[
                : (-1 if plot_inventory else None)
            ]:
                l.set_linewidth(3.0)


def add_title(ax: Axes, country: str, r_data: dict, country_codes_as_titles: bool):
    """
    Add title to matplotlib axes either as the region code or as the full region name whose flux are plotted.
    Args:
        ax: axis to set title to.
        country: country name, should correspond to the data plotted on the axes.
        r_data: dictionnary containing infos on regions (like decomposition of "super-region2 -like BENELUX- or full name). See configs/regions_info.json.
        country_codes_as_titles: If True, write the list of country code in the title, under the region name.
    """

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


def plot_country_flux(
    ds_all: dict[str, xr.Dataset],
    species: str,
    plot_regions: list[str] | str = [],
    config_data: dict[str, dict] = {},
    model_colors: dict[str, list] = {},
    model_labels: dict[str, str] = {},
    start_date: str | None = None,
    end_date: str | None = None,
    annex_mode: bool = False,
    plot_inventory: bool = True,
    inventory_years: list[str] | None = None,
    inventory_filename: str = "UNFCCC_inventory",
    data_dir: str | None = None,
    fix_y_axes: bool | list[float] = False,
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
    rolling_mean: bool | list[bool] = False,
    aggreg_month: bool = False,
    sector: str = "total",
) -> Figure | tuple[Figure, dict[str, dict]]:
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
        inventory_filename: Name of inventory file: {inventory_filename}_{species}_{inventory_year}
        data_dir: Path to top data directory, used to read inventory data files.
        fix_y_axes: If True, uses a consistent y axis for all plots. If list of 2 floats, use them as min and max of all the y axes.
        add_prior: If True, plots prior as dashed lines.
        add_prior_unc: If True, plots prior uncertainty as shaded area.
        set_global_leg: If True, plots one single legend instead of one legend per subplot.
        country_codes_as_titles: If True, write the list of country codes in the titles, under the region names.
        plot_separate: If True, plots model result as separate line. List must be of same size as models, e.g. [True, False, False].
            If a single boolean is provided, the same flag is assumed for all models.
        plot_combined: If True, the model is included in combined average result to be plotted. List must be of same size as models, e.g. [False, True, True].
            If a single boolean is provided, the same flag is assumed for all models.
        resample: Option to be passed to resample built-in function of xarray Dataset. For yearly average, 'YS' option should be used; 'QS-DEC' for seasonaly average.
            See http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
        resample_uncert_correlation: If True, calculates the resampled uncertainty as the mean from all averaged periods.
            If False, recalculates uncertainty assuming no correlation between all averaged periods, by taking the square root of the summed variances, divided by the number of averaging periods.
        plot_resample_and_original: If True, plots both the resampled data and the data as its original frequency. If False, only plots the resampled data.
        return_res: Wheter or not including a dictionnary with the results as output
        rolling_mean : If True, calculates a rolling mean (xx years) for each of the data to plot.
        aggreg_month: if True, plot the data aggregated by month. Used to study seasonnal cycle.
    Returns:
        fig: A plot per country/region.
        res_dict : If return_res, return also a dictionnary containing the plotted results

    """
    if aggreg_month and plot_inventory:
        logger.warning(
            "`plot_inventory` is not yet supported for monthly aggregate plots (`aggreg_month=True`). `plot_inventory` is set to False."
        )
        plot_inventory = False

    s_data = config_data.get("species_info", {})
    r_data = config_data.get("regions_info", {})

    plot_regions = format_plot_regions(plot_regions, ds_all)
    unit = get_unit(ds_all)

    inventory_data = dict()
    posterior_data = {m: dict() for m in plot_regions}
    prior_data = {m: dict() for m in plot_regions}

    # Sel data
    ds_all = {k: ds.sel(time=slice(start_date, end_date)) for k, ds in ds_all.items()}

    # Create figure
    fig, axes = create_fig_and_axes(len(plot_regions))

    # Iterate over each axes and regions
    for ax, country in zip(axes, plot_regions):

        # prepare model results
        ds_all_region = extract_region_flux(ds_all, country, r_data, sector=sector)
        ds_to_plot = prepare_data_to_plot(
            ds_all_region=ds_all_region,
            model_labels=model_labels,
            model_colors=model_colors,
            plot_separate=plot_separate,
            plot_combined=plot_combined,
            resample=resample,
            rolling_mean=rolling_mean,
            plot_resample_and_original=plot_resample_and_original,
            resample_uncert_correlation=resample_uncert_correlation,
            aggreg_month=aggreg_month,
        )

        # plot posterior and prior (if requested)
        for m, ds_region in ds_to_plot.items():
            highlighted_post = (m == "combined") & annex_mode
            posterior_data[country][m] = add_posterior_plot(
                ax, ds_region, highlighted_post
            )

            if add_prior:
                prior_data[country][m] = add_prior_plot(
                    ax, ds_region, annex_mode, add_prior_unc
                )

        # plot inventory
        if plot_inventory:
            inventory_data[country] = add_inventory_barplot(
                ax,
                data_dir,
                country,
                species,
                start_date,
                end_date,
                unit,
                s_data,
                r_data,
                inventory_years,
                inventory_filename,
                sector,
            )

        # set y label
        add_ylabel(ax, s_data, species, sector, unit)

        # set grid
        ax.grid(visible=True, which="major", alpha=0.4)

        # set ax title
        add_title(ax, country, r_data, country_codes_as_titles)

    res_dict: dict[str, dict] = {
        "inventory": inventory_data,
        "posterior": posterior_data,
        "prior": prior_data,
    }
    add_ylim(axes, plot_regions, res_dict, fix_y_axes, set_global_leg)
    yearly_freq = (
        "yearly" in [ds.attrs["frequency"] for ds in ds_to_plot.values()]
        or resample == "year"
    )
    add_xlims_and_ticks(axes[-1], yearly_freq, res_dict, aggreg_month)

    add_legend(fig, set_global_leg, annex_mode, plot_inventory)

    logger.info(
        "NOTE: If all the data is not within axis limits, adjust the fix_y_axes parameter"
    )

    if return_res:
        return fig, res_dict
    else:
        return fig


def plot_country_sector_flux_bar(
    ds_all: dict[str, xr.Dataset],
    species: str,
    plot_region: str,
    config_data: dict[str, dict] = {},
    model_colors: dict[str, str] = {},
    model_labels: dict[str, str] = {},
    plot_inventory_or_prior: str = "inventory",
    inventory_years: list[str] | None = None,
    inventory_filename: str = "UNFCCC_inventory",
    data_dir: str | None = None,
    fix_y_axes: bool = False,
    resample: str | list[str] | None = None,
    resample_uncert_correlation: bool = False,
    rolling_mean: bool = False,
    sectors: list[str] = ["agriculture", "waste", "energy", "industry"],
) -> Figure | list:
    """
    Stacked bar plot of posterior fluxes, split by sector, for a single region, for a range of models.
    Option to plot either the prior fluxes, split by sector, as a separate plot.
    Or plot the inventory fluxes, split by sector (when this data becomes available for all countries).

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
        inventory_filename: Name of inventory file: {inventory_filename}_{species}_{inventory_year}
        data_dir: Path to top data directory, used to read inventory data files.
        fix_y_axes: If True, uses a consistent y axis for all plots.
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
        return_res: Wheter or not including a dictionnary with the results as output
        rolling_mean : If True, calculates a rolling mean (xx years) for each of the data to plot.
    Returns:
        fig: A plot per country/region.
        res_dict : If return_res, return also a dictionnary containaing the plotted results
    """
    # sector_colors = {'agriculture':'darkgreen',
    #                'waste':'purple',
    #                'industry':'darkblue',
    #                'energy':'dodgerblue'}

    s_data = config_data.get("species_info", {})
    r_data = config_data.get("regions_info", {})
    sector_colors = config.get_default_sector_colors()

    max_cf = 0
    width = 1

    country_equivalent = {
        "NW_EU2": "NW EUROPE",
        "CW_EU": "CENTRAL W EUROPE",
        "NW_EU_CONTINENT": "NW CONTINENTAL EUROPE",
    }

    print_country = country_equivalent.get(plot_region, plot_region)

    # Create figure

    if plot_inventory_or_prior == "inventory":
        n_plots = len(ds_all.keys()) + 1
    elif plot_inventory_or_prior == "prior":
        n_plots = len(ds_all.keys()) * 2

    n_rows, n_cols = determine_subplots_arrangement(n_plots)

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

    for i, m in enumerate(ds_all.keys()):

        if n_rows == 1:
            ax_data = axes[i]
            if plot_inventory_or_prior == "prior":
                ax_comp = axes[1]
            if plot_inventory_or_prior == "inventory":
                ax_comp = axes[-1]
        else:
            ax_data = axes[i, 0]
            ax_comp = axes[i, 1]

        for s, sector in enumerate(sectors):

            ds_all_region = extract_region_flux(
                {m: ds_all[m]}, plot_region, r_data, sector=sector
            )
            ds_to_plot = prepare_data_to_plot(
                ds_all_region,
                model_labels,
                model_colors,
                plot_separate=True,
                plot_combined=True,
                resample=resample,
                rolling_mean=rolling_mean,
                plot_resample_and_original=False,
                resample_uncert_correlation=resample_uncert_correlation,
            )

            if resample != None:
                m_extract = m + "_resample"
            else:
                m_extract = m

            if s == 0:
                total_s = np.zeros(ds_to_plot[m_extract].time.shape[0])
                if plot_inventory_or_prior == "prior":
                    total_s_comp = np.zeros(ds_to_plot[m_extract].time.shape[0])

            xticks = np.arange(ds_to_plot[m_extract].time.values.shape[0])

            if resample == "season":
                xtick_labels = ds_to_plot[m_extract].time.values.astype("datetime64[M]")
            else:
                xtick_labels = ds_to_plot[m_extract].time.values.astype("datetime64[Y]")

            ax_data.bar(
                xticks,
                ds_to_plot[m_extract].posterior,
                label=sector.title(),
                color=sector_colors[sector],
                bottom=total_s,
                alpha=0.7,
                width=width,
            )
            total_s += ds_to_plot[m_extract].posterior

            if plot_inventory_or_prior == "prior":

                ax_comp.bar(
                    xticks,
                    ds_to_plot[m_extract].prior,
                    label=sector.title(),
                    color=sector_colors[sector],
                    bottom=total_s_comp,
                    alpha=0.7,
                    width=width,
                )
                total_s_comp += ds_to_plot[m_extract].prior

        if resample != None:
            ax_data.set_xticks(xticks)
            ax_data.set_xticks(xticks, minor=True)
            ax_data.set_xticklabels(xtick_labels)

            ax_comp.set_xticks(xticks)
            ax_comp.set_xticks(xticks, minor=True)
            ax_comp.set_xticklabels(xtick_labels)

        else:
            ax_data.set_xticks(xticks[::12])
            ax_data.set_xticks(xticks, minor=True)
            ax_data.set_xticklabels(xtick_labels[::12])

            ax_comp.set_xticks(xticks[::12])
            ax_comp.set_xticks(xticks, minor=True)
            ax_comp.set_xticklabels(xtick_labels[::12])

        max_cf = np.nanmax((max_cf, np.nanmax(total_s)))
        if plot_inventory_or_prior == "prior":
            max_cf = np.nanmax((max_cf, np.nanmax(total_s_comp)))

            ax_comp.set_ylabel(
                f"Prior\n{print_country} {s_data.get(species, {}).get('species_print', species)}"
                f" ({unit.replace('2','$_{{2}}$').replace('-1','$^{{-1}}$')})"
            )

        ax_data.set_ylabel(
            f"{model_labels[m]}\n{print_country} {s_data.get(species, {}).get('species_print', species)}"
            f" ({unit.replace('2','$_{{2}}$').replace('-1','$^{{-1}}$')})"
        )

        leg = ax_data.legend(ncol=2, borderpad=0.4, columnspacing=1.0)
        if plot_inventory_or_prior == "prior":
            leg = ax_comp.legend(ncol=2, borderpad=0.4, columnspacing=1.0)

    if plot_inventory_or_prior == "inventory":

        ax_comp = axes[-1]

        for s, sector in enumerate(sectors):
            inventories_to_plot = retrieve_inventories(
                data_dir,
                plot_region,
                species,
                (ds_to_plot[m_extract].time.values[0].astype("datetime64[Y]"))
                - np.timedelta64(1, "Y"),
                ds_to_plot[m_extract].time.values[-1],
                unit,
                s_data,
                r_data,
                inventory_years,
                inventory_filename,
                sector=sector,
            )[0]

            if ds_to_plot[m_extract].attrs["frequency"] == "monthly":
                logger.info(
                    f"{m_extract} inversion is monthly, "
                    + "so annual inventory value applied to each month"
                )
                # inventories_to_plot = inventories_to_plot.resample(time='1M',origin='start').ffill()
                inventories_to_plot = inventories_to_plot.reindex_like(
                    ds_to_plot[m_extract], method="ffill"
                )

            elif ds_to_plot[m_extract].attrs["frequency"] == "yearly":
                logger.info(
                    f"{m_extract} inversion is yearly, "
                    + "so no adjustments made to inventory data"
                )

            if s == 0:
                total_s_comp = np.zeros(inventories_to_plot.values.shape)

            ax_comp.bar(
                np.arange(inventories_to_plot.time.values.shape[0]),
                inventories_to_plot,
                label=sector.title(),
                color=sector_colors[sector],
                bottom=total_s_comp,
                width=width,
                alpha=0.7,
            )

            total_s_comp += inventories_to_plot.values

        ax_comp.set_xticks(xticks[::12])
        ax_comp.set_xticks(xticks, minor=True)
        ax_comp.set_xticklabels(xtick_labels[::12])
        ax_comp.set_ylabel(
            f"{print_country} Inventory {s_data.get(species, {}).get('species_print', species)}"
            f" ({unit.replace('2','$_{{2}}$').replace('-1','$^{{-1}}$')})"
        )

        leg = ax_comp.legend(ncol=2, borderpad=0.4, columnspacing=1.0)
        max_cf = np.nanmax((max_cf, np.nanmax(total_s_comp)))

    for i, m in enumerate(ds_all.keys()):

        if n_rows == 1:
            ax_data = axes[i]
            if plot_inventory_or_prior == "prior":
                ax_comp = axes[1]
            if plot_inventory_or_prior == "inventory":
                ax_comp = axes[-1]
        else:
            ax_data = axes[i, 0]
            ax_comp = axes[i, 1]

        if fix_y_axes == True:
            ax_data.set_ylim([0, max_cf * 1.2])
            ax_comp.set_ylim([0, max_cf * 1.2])

    return fig
