import math
import logging
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from typing import Tuple
from datetime import date, datetime, timedelta
from calendar import isleap, month_abbr, monthrange

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.dates import YearLocator, MonthLocator
from matplotlib.ticker import NullFormatter
from matplotlib import __version__ as mplt_version

from fluxy import config
from fluxy.operators.regions import extract_region_flux
from fluxy.operators.rolling_mean import calc_rolling_mean
from fluxy.operators.flux_timeseries_resample import resample_flux
from fluxy.operators.flux_combine import combine_dataset
from fluxy.operators.flux_prepare_inventory import retrieve_inventories
from fluxy.plots.utils import update_list_params

logger = logging.getLogger(__name__)

country_equivalent = {
    "NW_EU2": "NW EUROPE",
    "CW_EU": "CENTRAL W EUROPE",
    "NW_EU_CONTINENT": "NW CONTINENTAL EUROPE",
}


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


def get_posterior_unit(ds_all: dict[str, xr.Dataset]) -> str:
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


def create_fig_and_axes(
    nb_subplots: int, transpose: bool = False
) -> Tuple[Figure, Axes]:
    """
    Create matplotib figure and axes object based on the number of subplots asked for.
    Args:
        nb_subplots: number of subplot wanted (should be the number of subregion to plot when used in plot_country_flux)
    Returns:
        fig, axes: fig and flattened axes
    """

    n_cols_rows = determine_subplots_arrangement(nb_subplots)
    if transpose:
        n_rows, n_cols = n_cols_rows
    else:
        n_cols, n_rows = n_cols_rows

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
    combined_models_dict: dict[str, list[str]] | None = None,
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
        combined_models_dict: dictionnary defining the different combined models to plot. Keys are the labels of the combined results.
            Only used if plot_combined is set to True.
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

    if isinstance(plot_combined, bool) and plot_combined:
        is_plot_combined_single_true = True
    else:
        is_plot_combined_single_true = False

    # Convert some inputs to list and check their size
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
            model = m + "_resample" if rs else m
            ds_to_plot[model] = calc_rolling_mean(ds_to_plot[model])

    # Add combined dataset(s) to plot
    if any(plot_combined):
        if is_plot_combined_single_true:
            if combined_models_dict is None:
                combined_models_dict = {"Mean": list(ds_all_region.keys())}
            else:
                combined_model_list = sum(combined_models_dict.values(), [])
                check_missing_models = set(combined_model_list) - set(ds_all_region.keys())
                if check_missing_models:
                    raise ValueError(
                        f"Models in `combined_model_list` are not available: {check_missing_models}. "
                        f"Available models: {list(ds_all_region.keys())}"
                    )
        else:
            if combined_models_dict is not None:
                logger.warning(
                    "`combined_models_dict` will be re-written according to the elements of `plot_combined`. Label 'Mean' will be used in the plots."
                    " To combine the models listed in `combined_models_dict`, please set `plot_combined = True`."
                )
            combined_models_dict = {
                "Mean": [
                    m for (i, m) in enumerate(ds_all_region.keys()) if plot_combined[i]
                ]
            }

        combined_resample = [resamp for comb, resamp in zip(plot_combined, resample) if comb]
        use_resampled = ( 
            len(unique_resample := set(combined_resample)) == 1
            and unique_resample not in ({None}, {False})
        )
        if use_resampled:
                combined_models_dict = {
                    group_label: [f"{model}_resample" for model in model_list]
                    for group_label, model_list in combined_models_dict.items()
                }
                ds_to_combine = {
                    m: calc_rolling_mean(ds) if rm else ds
                    for rm, (m, ds) in zip(rolling_mean, ds_resampled.items())
                }
        else:
            ds_to_combine = {
                m: calc_rolling_mean(ds) if rm else ds
                for rm, (m, ds) in zip(rolling_mean, ds_all_region.items())
            }

        for group_label, model_list in combined_models_dict.items():
                combine_mask = [model in model_list for model in ds_to_combine.keys()]
                ds_combined = combine_dataset(ds_to_combine, combine_mask)
                ds_combined["combined"].attrs["model_label"] = group_label
                if any('_resample' in s for s in model_list) and plot_resample_and_original:
                    ds_combined["combined"].attrs["model_label"] += " (resampled)"
                new_key = group_label.replace(" ", "_")
                ds_combined = {f"combined_{new_key}": ds_combined["combined"]} # rename key to include group label
                ds_to_plot.update(ds_combined)

    # Determine plot color and label of each dataset
    color_usage = {k: 0 for k in map_model_colors.keys()}
    i_comb = 0
    for m in ds_to_plot.keys():
        if "combined" in m:
            include_label = ds_to_plot[m].attrs.get("model_label", None)
            model_color = config.mean_color_palette[i_comb % len(config.mean_color_palette)]
            i_comb += 1
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
            ds_to_plot[m].attrs["model_label"] += " (resampled)"

        ds_to_plot[m].attrs["model_color"] = model_color

    return ds_to_plot


def add_posterior_plot(
    ax: Axes, ds_region: xr.Dataset, highlighted_line: bool, add_post_unc: bool
) -> dict[str, dict]:
    """
    Plot the posterior data on the axis. The variable posterior of the dataset ds_region is plotted as a line (color and label found in the dataset
    attributes) and the uncertainty (variables lower_posterior, upper_posterior in the dataset) is plotted as a semi-transparent filled space.
    Axes:
        ax: axes on which to plot
        ds_region: dataset containing posterior data
        highlighted_line: if True, the linewidth is made bigger (3.0) than when False (1.5). Typicall used for the annexes to highlight the PARIS mean.
        add_post_unc: if True, plots model uncertainty.
    Returns:
    Returns:
        res: dataframe with one line per timestamp and 9 columns ("type", "model", "sector", "country", "species", 
            "time", "mean_val", "min_unc", "max_unc")
    """

    linew = 3 if highlighted_line else 1.5

    time_as_datetime = ds_region.time.values.astype("datetime64[D]").tolist()

    ax.plot(
        time_as_datetime,
        ds_region.posterior,
        label=ds_region.attrs["model_label"],
        color=ds_region.attrs["model_color"],
        linewidth=linew,
    )

    if add_post_unc:
        ax.fill_between(
            time_as_datetime,
            ds_region.posterior_lower,
            ds_region.posterior_upper,
            alpha=0.2,
            color=ds_region.attrs["model_color"],
        )

    res = pd.DataFrame(
        {
            "type": ["posterior",] * ds_region.time.size,
            "model": [ds_region.attrs["model_label"],] * ds_region.time.size,
            "sector": [ds_region.attrs["sector"],] * ds_region.time.size,
            "country": [ds_region.attrs["country"],] * ds_region.time.size,
            "species": [ds_region.attrs["species"],] * ds_region.time.size,
            "time": time_as_datetime,
            "mean_val": ds_region.posterior.values,
            "min_unc": ds_region.posterior_lower.values,
            "max_unc": ds_region.posterior_upper.values,
        }
    )
    return res


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
        res: dataframe with one line per timestamp and 7-9 columns ("type", "model", "sector", "country", "species", 
            "time", "mean_val" and "min_unc", "max_unc" if add_prior_unc)
    """
    linewidth, alpha = (1.0, 0.7) if annex_mode else (1.5, 1.0)

    time_as_datetime = ds_region.time.values.astype("datetime64[D]").tolist()

    ax.plot(
        time_as_datetime,
        ds_region.prior,
        label=ds_region.attrs["model_label"] + " prior",
        color=ds_region.attrs["model_color"],
        linestyle="dashed",
        linewidth=linewidth,
        alpha=alpha,
    )

    res = pd.DataFrame(
        {
            "type": ["prior",] * ds_region.time.size,
            "model": [ds_region.attrs["model_label"],] * ds_region.time.size,
            "sector": [ds_region.attrs["sector"],] * ds_region.time.size,
            "country": [ds_region.attrs["country"],] * ds_region.time.size,
            "species": [ds_region.attrs["species"],] * ds_region.time.size,
            "time": time_as_datetime,
            "mean_val": ds_region.prior.values,
        }
    )
    if add_prior_unc:
        ax.fill_between(
            time_as_datetime,
            ds_region.prior_lower,
            ds_region.prior_upper,
            alpha=0.1,
            color=ds_region.attrs["model_color"],
        )
        res["min_unc"] = ds_region.prior_lower.values
        res["max_unc"] = ds_region.prior_upper.values

    return res


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
    annex_mode: bool,
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
        annex_mode: If True, replace Inventory label with a more concise version for National Inventory Report Annexes.
    Returns:
        res: dataframe with one line per timestamp and 7 columns ("type", "model", "sector", "country", "species", 
            "time", "mean_val")
    """

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
        sectors=sector,
    )

    res = pd.DataFrame()
    for i_inv, inventory in enumerate(inventories_to_plot):
        time_as_datetime = inventory.time.values.astype("datetime64[D]").tolist()

        ax.bar(
            time_as_datetime,
            inventory,
            timedelta(days=340 - i_inv * 20),
            edgecolor=inventory.plot_color,
            align="edge",
            fill=False,
            label=(
                f"NID {inventory.year}" if annex_mode else f"Inventory {inventory.year}"
            ),
            zorder=0,
        )

        tmp = pd.DataFrame(
            {
                "type": ["inventory",] * inventory.time.size,
                "model": [f"inventory_{inventory.year}",] * inventory.time.size,
                "sector": [sector,] * inventory.time.size,
                "country": [country,] * inventory.time.size,
                "species": [species,] * inventory.time.size,
                "time": time_as_datetime,
                "mean_val": inventory.values,
            }
        )
        res = pd.concat([res, tmp], ignore_index=True)

    return res


def add_sector_barplot(
    ax: Axes, ds_sector: xr.Dataset, variable: str, bottom_values: np.ndarray | float
) -> dict[str, dict]:
    """
    Add a layer to the stacked barplot. The layer correspond to a sector.
    Axes:
        ax: axes on which to plot
        ds_sector: dataset for the sector of interest
        variable: variable to plot (either "posterior" or "prior")
        bottom_values: bottom values passed as argument toax.bar. Correspond to the previous heights of the stacks.
    Returns:
        res: dataframe with one line per timestamp and 7 columns ("type", "model", "sector", "country", "species", 
            "time", "mean_val")
    """

    sector_colors = config.get_default_sector_colors()
    sector = str(ds_sector.sector.values)

    freq = ds_sector.attrs.get("frequency", "unknown")

    if variable == "inv_data" or freq in ["year", "yearly"]:
        time_as_datetime = ds_sector.time.values.astype("datetime64[Y]").tolist()
        width = [
            timedelta(days=366) if isleap(date.year) else timedelta(days=365)
            for date in time_as_datetime
        ]
        offset = timedelta(days=183)
    elif freq == "monthly":
        time_as_datetime = ds_sector.time.values.astype("datetime64[M]").tolist()
        width = [
            timedelta(days=monthrange(date.year, date.month)[1])
            for date in time_as_datetime
        ]
        offset = timedelta(days=15)
    else:
        time_as_datetime = ds_sector.time.values.astype("datetime64[D]").tolist()
        width = [
            (time_as_datetime[i + 1] - time_as_datetime[i - 1]) / 2
            for i in range(1, len(time_as_datetime) - 1)
        ]
        width = [np.mean(width), *width, np.mean(width)]
        offset = timedelta(days=0)

    ax.bar(
        time_as_datetime,
        ds_sector[variable].values,
        label=sector.title(),
        color=sector_colors[sector],
        bottom=bottom_values,
        alpha=0.7,
        width=width,
        align="edge",
    )

    res = pd.DataFrame(
        {
            "type": [variable,] * ds_sector.time.size,
            "model": [ds_sector.attrs["model_label"],] * ds_sector.time.size,
            "sector": [sector,] * ds_sector.time.size,
            "country": [ds_sector.attrs["country"],] * ds_sector.time.size,
            "species": [ds_sector.attrs["species"],] * ds_sector.time.size,
            "time": np.array(time_as_datetime) + offset,
            "mean_val": ds_sector[variable].values + bottom_values,
        }
    )

    return res


def prepare_inventory_sector_barplot(
    sectors: list[str],
    start_date: str,
    end_date: str,
    data_dir: str,
    plot_region: str,
    species: str,
    unit: str,
    s_data: dict[str, dict],
    r_data: dict[str, dict],
    inventory_years: str | list[str] | None,
    inventory_filename: str,
) -> list[xr.Dataset]:
    """
    Prepare the inventory for the sector barplot.
    Call fluxy.operators.flux_prepare_inventory.retrieve_inventories and change the outputed dataarrays in to dataset (with variable name "inv_data").

    Args:
        sectors: List of emissions sectors
        start_date: Start date of the data to plot.
        end_date: End date of the data to plot.
        data_dir: directory which contains the data (should have inside a directory named 'inventory').
        plot_region: Region of interest.
        species: Gas species, e.g. 'ch4'.
        unit: unit in which the inventory should be converted.
        s_data: Dictionary of species with information for plotting (read from json file).
        r_data: Dictionary with country and region names (read from json file).
        inventory_years: List of inventory data from different years to include. If None, only plots the most recent inventory data.
        inventory_filename: Name of inventory file: {inventory_filename}_{species}_{inventory_year}
    Returns:
        inventories_list : list of inventory data to be plotted.

    """

    inventories = retrieve_inventories(
        data_dir,
        plot_region,
        species,
        start_date,
        end_date,
        unit,
        s_data,
        r_data,
        inventory_years,
        inventory_filename,
        sectors=sectors,
    )

    inventories = [inv.to_dataset(name="inv_data") for inv in inventories]

    return inventories


def add_ylim(
    axes: list[Axes],
    dim: str,
    values: list[str],
    plotted_data_df: pd.DataFrame,
    fix_y_axes: bool | list[float] | None,
    set_global_leg: bool = False,
):
    """
    Add limits to y axes based on results in res_dict, or the values given in fig_y_axes if it's a list.

    Args:
        axes: list of axes to add the ylim to.
        dim: dimension following which the list of axes is made.
        values: list of values taken by the dimension and corresponding to the axes (should be the same length and order).
        res_dict: dictionnary containing the data plotted. Should have one key per regions plotted, the values being dictionnaries with 3 keys: "inventory", "posterior" and "prior";
            whose values are the output of add_inventory_barplot, add_posterior_plot, add_prior_plot). The data stored in them is used to infer the ylims.
        fix_y_axes: if list, use it as params to ax.set_ylim; if bool and True, all subplots have the same y lim (the max value that can be found in res_dict); else the max of the data
            plotted in each subplots is used.
        set_global_leg: if True (and thus one common legend is plotted for all subplots in add_legend), a zoom of only 1.1 is made on the ymax, else it is 1.2 to make space for the legend.
    """

    if isinstance(fix_y_axes, list):
        if isinstance(fix_y_axes[0], list):
            if len(fix_y_axes) != len(values):
                raise ValueError(
                    "'fix_y_axes' must be a boolean, a list with 2 floats, or a list of lists of the same length as `values`."
                )
            for i, ax in enumerate(axes):
                ax.set_ylim(*fix_y_axes[i])
        else:
            if len(fix_y_axes) != 2:
                raise ValueError(
                    "'fix_y_axes' must be a boolean, a list with 2 floats, or a list of lists of the same length as `values`."
                )
            for ax in axes:
                ax.set_ylim(*fix_y_axes)
        return

    max_cf = []
    fac = 1.1 if set_global_leg else 1.2

    for ax, val in zip(axes, values):
        df_country = plotted_data_df[plotted_data_df[dim] == val]
        max_country = np.nanmax(
            df_country[df_country.columns.intersection(["mean_val", "max_unc"])]
        )

        max_cf.append(max_country)

    for i, ax in enumerate(axes):
        if fix_y_axes:
            ax.set_ylim(0, np.nanmax(max_cf) * fac)
        else:
            ax.set_ylim(0, max_cf[i] * fac)


def add_ylabel(
    ax: Axes,
    s_data: dict[str, dict],
    species: str,
    unit: str,
    plot_type: str,
    **kwargs: str | int,
):
    """
    Add label to y axis.
    Args:
        ax: axis to add the y label to.
        s_data: dictionnary containing info for each species (like name to use for label). See configs/species_infos.json.
        species: name of the species.
        sector: name of sector plotted.
        unit: unit of data plotted.
    """
    if plot_type == "country_plot":
        sector = kwargs["sector"]
        ax.set_ylabel(
            f"{s_data.get(species, {}).get('species_print', species)} {sector.title() if sector != 'total' else ''}"
            f" ({unit.replace('2','$_{{2}}$').replace('-1','$^{{-1}}$')})"
        )
    # sector plot - posterior
    elif "sector_barplot" in plot_type:

        print_country = country_equivalent.get(kwargs["region"], kwargs["region"])

        if plot_type.split("-")[1] == "posterior":
            ax.set_ylabel(
                f"{kwargs['label']}\n{print_country} {s_data.get(species, {}).get('species_print', species)}"
                f" ({unit.replace('2','$_{{2}}$').replace('-1','$^{{-1}}$')})"
            )
        elif plot_type.split("-")[1] == "prior":
            ax.set_ylabel(
                f"Prior\n{print_country} {s_data.get(species, {}).get('species_print', species)}"
                f" ({unit.replace('2','$_{{2}}$').replace('-1','$^{{-1}}$')})"
            )
        elif plot_type.split("-")[1] == "inventory":
            ax.set_ylabel(
                f"{kwargs['inventory_filename'].replace('_',' ')} {kwargs['year']}\n{print_country} {s_data.get(species, {}).get('species_print', species)}"
                f" ({unit.replace('2','$_{{2}}$').replace('-1','$^{{-1}}$')})"
            )


def add_xlims_and_ticks(
    ax: Axes, yearly_freq: bool, plotted_data_df: dict[str, dict], aggreg_month: bool
):
    """
    Add x limits, ticks and ticks labels to matplotlib axes. Optimize them by looking at if they are monthly, yearly, or monthly aggregated, and covered time range.
    Args:
        ax: axis to add xlim and xticks to.
        yearly_freq: set to True if the data plotted have a yearly frequency.
        res_dict: dictionnary containing the data plotted. Should have one key per regions plotted, the values being dictionnaries with 3 keys: "inventory", "posterior" and "prior";
            whose values are the output of add_inventory_barplot, add_posterior_plot, add_prior_plot). The time data stored in them is used to infer the xlims.
        aggreg_month: if True, the data plotted are supposed to be a monthly aggregated so 12 stciks are created, whose labels are the 3 first letters of each month.
    """
    if aggreg_month:
        ax.set_xticks(np.arange(1, 13))
        ax.set_xticklabels(list(month_abbr)[1:])
        return

    min_x = np.nanmin(plotted_data_df["time"])
    max_x = np.nanmax(plotted_data_df["time"])

    # set xticks
    year_range = date(max_x.year, 1, 1) - date(min_x.year, 1, 1)
    if yearly_freq:
        min_x = date(min_x.year, 1, 1)
        max_x = date(max_x.year + 1, 1, 1)
    xlim = [min_x - (max_x - min_x) / 50, max_x + (max_x - min_x) / 50]

    if year_range > timedelta(days=8 * 365.25) or yearly_freq:
        min_x = date(min_x.year, 1, 1)
        max_x = date(max_x.year + 1, 1, 1)
        step = (max_x.year - min_x.year) // 8 + 1
        xticks = np.array(
            [date(year, 1, 1) for year in range(min_x.year, max_x.year, step)]
        )
        if (max_x.year - min_x.year) % step == 0:
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
    Add legend.
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
            ncol = 3 if annex_mode else 2
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
        ax: axis to add title to.
        country: country name, should correspond to the data plotted on the axes.
        r_data: dictionnary containing infos on regions (like decomposition of "super-region2 -like BENELUX- or full name). See configs/regions_info.json.
        country_codes_as_titles: If True, write the list of country code in the title, under the region name.
    """

    # set title
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
    plot_separate_unc: bool | None = None,
    plot_combined: bool | list[bool] = False,
    plot_combined_unc: bool | None = None,
    combined_models_dict: dict[str, list[str]] | None = None,
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
        plot_separate_unc: If True, plots separate models uncertainty.
            If None, will default to True if any value in plot_separate is True.
            If explicitly True/False, that value is used.
        plot_combined: If True, the model is included in combined average result to be plotted. List must be of same size as models, e.g. [False, True, True].
            If a single boolean is provided, the same flag is assumed for all models.
        plot_combined_unc: If True, plots combined average model uncertainty.
            If None, will default to True if any value in plot_combined is True.
            If explicitly True/False, that value is used.
        combined_models_dict: dictionnary defining the different combined models to plot. Keys are the name of the combined model.
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
        res_dict : If return_res, return also a dataframe containing the plotted results. The columns of this dataframe are "type" (possible values "prior"/"posterior"/"inventory"),
            "model", "sector", "country", "species", "time", "mean_val", "min_unc", "max_unc".
    """
    if aggreg_month and plot_inventory:
        logger.warning(
            "`plot_inventory` is not yet supported for monthly aggregate plots (`aggreg_month=True`). `plot_inventory` is set to False."
        )
        plot_inventory = False

    s_data = config_data.get("species_info", {})
    r_data = config_data.get("regions_info", {})


    plot_regions = format_plot_regions(plot_regions, ds_all)
    unit = get_posterior_unit(ds_all)

    plotted_data_df = pd.DataFrame()

    # Compute default for plot_separate_unc and plot_combined_unc if not given
    plot_separate_unc = (
        np.any(plot_separate) if plot_separate_unc is None else plot_separate_unc
    )
    plot_combined_unc = (
        np.any(plot_combined) if plot_combined_unc is None else plot_combined_unc
    )

    # Sel data
    ds_all = {k: ds.sel(time=slice(start_date, end_date)) for k, ds in ds_all.items()}

    # Create figure
    fig, axes = create_fig_and_axes(len(plot_regions))

    # Iterate over each axes and regions
    for ax, country in zip(axes, plot_regions):

        # prepare model results
        ds_all_region = extract_region_flux(ds_all, country, r_data, sectors=sector)
        ds_to_plot = prepare_data_to_plot(
            ds_all_region=ds_all_region,
            model_labels=model_labels,
            model_colors=model_colors,
            plot_separate=plot_separate,
            plot_combined=plot_combined,
            combined_models_dict=combined_models_dict,
            resample=resample,
            rolling_mean=rolling_mean,
            plot_resample_and_original=plot_resample_and_original,
            resample_uncert_correlation=resample_uncert_correlation,
            aggreg_month=aggreg_month,
        )

        # plot posterior and prior (if requested)
        for m, ds_region in ds_to_plot.items():
            highlighted_post = ("combined" in m) & annex_mode
            add_post_unc = (("combined" in m) & plot_combined_unc) |  (("combined" not in m) & plot_separate_unc)
            posterior_df = add_posterior_plot(
                ax, ds_region, highlighted_post, add_post_unc
            )
            plotted_data_df = pd.concat(
                [plotted_data_df, posterior_df], ignore_index=True
            )

            if add_prior:
                prior_df = add_prior_plot(ax, ds_region, annex_mode, add_prior_unc)
                plotted_data_df = pd.concat(
                    [plotted_data_df, prior_df], ignore_index=True
                )

        # plot inventory
        if plot_inventory:
            inventory_df = add_inventory_barplot(
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
                annex_mode,
            )
            plotted_data_df = pd.concat(
                [plotted_data_df, inventory_df], ignore_index=True
            )

        # set y label
        add_ylabel(ax, s_data, species, unit, plot_type="country_plot", sector=sector)

        # set grid
        ax.grid(visible=True, which="major", alpha=0.4)

        # set ax title
        add_title(ax, country, r_data, country_codes_as_titles)

    add_ylim(axes, "country", plot_regions, plotted_data_df, fix_y_axes, set_global_leg)
    yearly_freq = (
        "yearly" in [ds.attrs["frequency"] for ds in ds_to_plot.values()]
        or resample == "year"
    )
    add_xlims_and_ticks(axes[-1], yearly_freq, plotted_data_df, aggreg_month)

    add_legend(fig, set_global_leg, annex_mode, plot_inventory)

    logger.info(
        "NOTE: If all the data is not within axis limits, adjust the fix_y_axes parameter"
    )

    if return_res:
        return fig, plotted_data_df
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
    fix_y_axes: bool = True,
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
        res_dict : If return_res, return also a dictionnary containing the plotted results
    """
    plot_type = "sector_barplot"
    s_data = config_data.get("species_info", {})
    r_data = config_data.get("regions_info", {})
    unit = get_posterior_unit(ds_all)

    plotted_data_df = pd.DataFrame()

    # prepare data
    ds_all_region = extract_region_flux(ds_all, plot_region, r_data, sectors=sectors)
    ds_to_plot = prepare_data_to_plot(
        ds_all_region,
        model_labels,
        model_colors,
        resample=resample,
        rolling_mean=rolling_mean,
        resample_uncert_correlation=resample_uncert_correlation,
    )

    freqs = [ds.attrs["frequency"] for ds in ds_to_plot.values()]

    if plot_inventory_or_prior == "inventory":
        start_date = str(min([ds.time.values.min() for ds in ds_to_plot.values()]))[:10]
        end_date = str(max([ds.time.values.max() for ds in ds_to_plot.values()]))[:10]
        inv_plot_data = prepare_inventory_sector_barplot(
            sectors,
            start_date,
            end_date,
            data_dir,
            plot_region,
            species,
            unit,
            s_data,
            r_data,
            inventory_years,
            inventory_filename,
        )

    # create figure
    if plot_inventory_or_prior == "inventory":
        n_plots = len(ds_to_plot.keys()) + len(inventory_years)
        fig, axes = create_fig_and_axes(n_plots)
    elif plot_inventory_or_prior == "prior":
        n_plots = len(ds_to_plot.keys()) * 2
        fig, axes = create_fig_and_axes(n_plots, transpose=True)

    for i, (m, ds) in enumerate(ds_to_plot.items()):

        if plot_inventory_or_prior == "prior":
            ax_data = axes[2 * i]
            ax_comp = axes[2 * i + 1]
        else:
            ax_data = axes[i]

        # plot posterior (and eventually prior)
        former_sector = None
        for sector in sectors:
            vars_to_plot = (
                ["posterior", "prior"]
                if plot_inventory_or_prior == "prior"
                else ["posterior"]
            )
            for var, ax in zip(vars_to_plot, [ax_data, ax_comp]):
                bottom_values = (
                    plotted_data_df[
                        (plotted_data_df.sector == former_sector)
                        & (plotted_data_df.type == var)
                        & (plotted_data_df.model == ds.attrs["model_label"])
                    ].mean_val
                    if former_sector
                    else 0
                )
                res = add_sector_barplot(ax, ds.sel(sector=sector), var, bottom_values)
                plotted_data_df = pd.concat([plotted_data_df, res], ignore_index=True)

            former_sector = sector

        # set y_label inventory_filename, year
        add_ylabel(
            ax_data,
            s_data,
            species,
            unit,
            plot_type=f"{plot_type}-posterior",
            region=plot_region,
            label=ds.attrs["model_label"],
        )
        if plot_inventory_or_prior == "prior":
            ax_comp.legend(ncol=2, borderpad=0.4, columnspacing=1.0)
            add_ylabel(
                ax_comp,
                s_data,
                species,
                unit,
                plot_type=f"{plot_type}-prior",
                region=plot_region,
            )

    # plot inventory sector bar
    if plot_inventory_or_prior == "inventory":
        for year, (i, inv) in zip(inventory_years, enumerate(inv_plot_data)):
            ax = axes[-i - 1]
            former_sector = None
            for sector in sectors:
                bottom_values = (
                    plotted_data_df[
                        (plotted_data_df.sector == former_sector)
                        & (plotted_data_df.type == "inventory")
                        & (plotted_data_df.model == f"inventory_{year}")
                    ].mean_val
                    if former_sector
                    else 0
                )
                res = add_sector_barplot(ax, ds.sel(sector=sector), var, bottom_values)
                plotted_data_df = pd.concat([plotted_data_df, res], ignore_index=True)
                former_sector = sector
            ax.legend(ncol=2, borderpad=0.4, columnspacing=1.0)
            add_ylabel(
                ax,
                s_data,
                species,
                unit,
                plot_type=f"{plot_type}-inventory",
                region=plot_region,
                inventory_filename=inventory_filename,
                year=year,
            )

    # plot grid and legend
    for ax in axes:
        ax.grid(visible=True, which="major", alpha=0.4)
        ax_data.legend(ncol=2, borderpad=0.4, columnspacing=1.0)

    # set y lim
    if not fix_y_axes:
        fix_y_axes = True
        logger.info(
            "Switching `fix_y_axes` as the the country are the same for every suplots iin a sector plots, and thus the ODG of the max values should be the same."
        )
    add_ylim(
        axes,
        "model",
        [ds.attrs["model_label"] for ds in ds_to_plot.values()],
        plotted_data_df,
        fix_y_axes,
    )

    # set xlim and xticks
    yearly_freq = ("year" in freqs) or ("yearly" in freqs)

    add_xlims_and_ticks(axes[-1], yearly_freq, plotted_data_df, aggreg_month=False)

    return fig, plotted_data_df
