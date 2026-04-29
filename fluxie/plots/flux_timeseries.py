import os
import math
import logging
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from typing import Literal, Tuple
from pathlib import Path
from datetime import date, datetime, timedelta
from calendar import isleap, month_abbr, monthrange

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.dates import YearLocator, MonthLocator
from matplotlib.ticker import NullFormatter
from matplotlib import __version__ as mplt_version

from scipy.optimize import curve_fit

from fluxie import config
from fluxie.operators.regions import extract_region_flux, format_plot_regions
from fluxie.operators.rolling_mean import calc_rolling_mean
from fluxie.operators.flux_timeseries_resample import resample_flux
from fluxie.operators.flux_combine import combine_dataset
from fluxie.operators.flux_prepare_inventory import retrieve_inventories
from fluxie.operators.convert import convert_units_co2eq
from fluxie.plots.utils import update_list_params

logger = logging.getLogger(__name__)

country_equivalent = {
    "NW_EU2": "NW EUROPE",
    "CW_EU": "CENTRAL W EUROPE",
    "NW_EU_CONTINENT": "NW CONTINENTAL EUROPE",
}


def get_unit(ds_all: dict[str, xr.Dataset]) -> str:
    """
    Determine unit of posterior estimations from datasets. If incoherencies between datasets, an error is raised.
    Args:
        ds_all: dictionnary of datasets from which read the units.
    Returns:
        unit: unit of posterior variables in dataset.
    """

    variables_to_check = [
        "flux_total_posterior_country",
        "posterior",
        "flux_total_prior_country",
        "prior",
    ]

    for var in variables_to_check:
        if all([var in ds for ds in ds_all.values()]):
            units = {ds[var].units for ds in ds_all.values()}
            if len(units) != 1:
                raise ValueError(
                    f"Inconsistency in the units from the different datasets for variable '{var}': {units} are present. "
                    "Only one is expected."
                )
            unit = list(units)[0]
            return unit

    raise ValueError(
        f"Did not find any of the expected variables {variables_to_check} in every dataset. Thus couldn't determine unit."
    )


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
        axes = axes.flatten()[:nb_subplots]
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
    only_overlapping: bool = True,
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
        only_overlapping: if True, only includes data from years when all models/species available. If False,
            includes all available data.
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
        ["plot_separate", "plot_combined", "resample", "rolling_mean"],
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
            m: v for (i, (m, v)) in enumerate(ds_all_region.items()) #if plot_separate[i]
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
                #if plot_separate[i]
            }
        )

    # Apply rolling mean when necessary
    for m, rm, ps, rs in zip(
        ds_all_region.keys(), rolling_mean, plot_separate, resample
    ):
        if rm: #& ps:  # if rolling_mean and plot_separate
            model = m + "_resample" if rs else m
            ds_to_plot[model] = calc_rolling_mean(ds_to_plot[model])

    # Add combined dataset(s) to plot
    if any(plot_combined):
        if is_plot_combined_single_true:
            if combined_models_dict is None:
                combined_models_dict = {"Mean": list(ds_to_plot.keys())}
            else:
                combined_model_list = sum(combined_models_dict.values(), [])
                check_missing_models = set(combined_model_list) - set(
                    ds_all_region.keys()
                )
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
                    m for (i, m) in enumerate(ds_to_plot.keys()) if plot_combined[i]
                ]
            }

        combined_resample = [
            resamp for comb, resamp in zip(plot_combined, resample) if comb
        ]
        use_resampled = len(
            unique_resample := set(combined_resample)
        ) == 1 and unique_resample not in ({None}, {False})
        if use_resampled:
            combined_models_dict = {
                group_label: [f"{model}_resample" for model in model_list]
                for group_label, model_list in combined_models_dict.items()
            }

        ds_to_combine = {
            m: ds #calc_rolling_mean(ds) if rm else ds
            for rm, (m, ds) in zip(
                rolling_mean, (ds_resampled if use_resampled else ds_to_plot).items()
            )
        }

        for i,(group_label, model_list) in enumerate(combined_models_dict.items()):
            combine_mask = [model in model_list for model in ds_to_combine.keys()]
            ds_combined = combine_dataset(ds_to_combine, combine_mask, only_overlapping)
            ds_combined["combined"].attrs["model_label"] = group_label
            if any("_resample" in s for s in model_list) and plot_resample_and_original:
                ds_combined["combined"].attrs["model_label"] += " (resampled)"
            new_key = group_label.replace(" ", "_")
            ds_combined = {
                f"combined_{new_key}": ds_combined["combined"]
            }  # rename key to include group label
            #ds_to_plot.update(ds_combined)
            if plot_separate[i] == False:
                ds_to_plot = ds_combined.copy()
            else:
                ds_to_plot.update(ds_combined)

    # Determine plot color and label of each dataset
    color_usage = {k: 0 for k in map_model_colors.keys()}
    i_comb = 0
    for m in ds_to_plot.keys():
        include_label = ds_to_plot[m].attrs.get("model_label", None)
        if "combined" in m:
            model_color = config.mean_color_palette[
                i_comb % len(config.mean_color_palette)
            ]
            i_comb += 1
        else:
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


def linear(x, m, c):
    return m * x + c


def add_line_plot(
    ax: Axes,
    ds: xr.Dataset,
    variable: Literal["posterior", "prior"],
    highlighted_line: bool = False,
    add_unc: bool = False,
    unit: str = None,
    plot_trends: bool = False,
) -> dict[str, dict]:
    """
    Plot the posterior/prior data on the axis. The variable posterior/prior of the dataset ds is plotted as a line (color and label found in the dataset
    attributes) and the uncertainty (variables lower_posterior/prior, upper_posterior/prior in the dataset) is plotted as a semi-transparent filled space.
    Args:
        ax: axes on which to plot
        ds: dataset containing posterior/prior data
        variable: whether the posterior or prior should be plotted
        highlighted_line: for posterior/prior, if True, the linewidth is made bigger (3.0/1.5) than when False (1.5/1.0). Typicaly used for the annexes to highlight the PARIS mean.
        add_unc: if True, plots model uncertainty.
    Returns:
        res: dataframe with one line per timestamp and 9 columns ("type", "model", "sector", "country", "species",
            "time", "mean_val", "min_unc", "max_unc")
    """
    if variable == "posterior":
        kwargs_plot = dict(
            ls="-",
            lw=3 if highlighted_line else 1.5,
            alpha=1.0,
            label=ds.attrs["model_label"],
        )
        alpha_unc = 0.2
    elif variable == "prior":
        kwargs_plot = dict(
            ls="--",
            lw=1.5 if highlighted_line else 1.0,
            alpha=1.0 if highlighted_line else 0.7,
            label=ds.attrs["model_label"] + " prior",
        )
        alpha_unc = 0.1
    else:
        raise ValueError(
            "Available options for 'variable' parameter are 'prior' and 'posterior'."
        )

    time_as_datetime = ds.time.values.astype("datetime64[D]").tolist()

    ax.plot(
        time_as_datetime,
        ds[variable],
        color=ds.attrs["model_color"],
        **kwargs_plot,
    )
    if variable == "posterior" and plot_trends:
       
        err = np.mean(
            [ds[f"{variable}_lower"].values, ds[f"{variable}_upper"].values], axis=0
        )
        opt, cov = curve_fit(
            linear, [i.year+i.month/12 for i in time_as_datetime], ds[variable], sigma=err
        )

        ax.plot(
            time_as_datetime,
            linear(np.array([i.year+i.month/12 for i in time_as_datetime]), *opt),
            color=ds.attrs["model_color"],
            linestyle=":",
            label=ds.attrs["model_label"] + " trend",
        )

        print(f"{ds.model_label} trend for {ds.attrs['country']} is: {opt[0]} {unit}")

    res = pd.DataFrame(
        {
            "time": time_as_datetime,
            "mean_val": ds[variable].values,
        }
    )
    res["type"] = variable
    res["model"] = ds.attrs["model_label"]
    for attr in ["sector", "country", "species"]:
        res[attr] = ds.attrs[attr]

    if add_unc:
        ax.fill_between(
            time_as_datetime,
            ds[f"{variable}_lower"],
            ds[f"{variable}_upper"],
            alpha=alpha_unc,
            color=ds.attrs["model_color"],
        )
        res["min_unc"] = ds[f"{variable}_lower"].values
        res["max_unc"] = ds[f"{variable}_upper"].values

    return res


def add_inventory_barplot(
    ax: Axes,
    data_dir: os.PathLike,
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
    plot_inventory_uncertainty: list[bool] | bool = False,
    plot_trends:bool=False
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
        plot_inventory_uncertainty: If True and uncertainty available, plots inventory error bars. If a list is provided, should be of same size as inventory_years.
    Returns:
        res: dataframe with one line per timestamp and 7 columns ("type", "model", "sector", "country", "species",
            "time", "mean_val")
    """

    data_dir = Path(data_dir)

    if isinstance(start_date, list):
        start_date_inv = str(min([np.datetime64(date) for date in start_date]))
    else:
        start_date_inv = start_date
    if isinstance(end_date, list):
        end_date_inv = str(max([np.datetime64(date) for date in end_date]))
    else:
        end_date_inv = end_date
    inventories_to_plot, inventories_uncert_to_plot = retrieve_inventories(
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

    (plot_inventory_uncertainty,) = update_list_params(
        [plot_inventory_uncertainty],
        ["plot_inventory_uncertainty"],
        expected_size=len(inventories_uncert_to_plot),
    )

    res = pd.DataFrame()
    for i_inv, inventory in enumerate(inventories_to_plot):
        time_as_datetime = inventory.time.values.astype("datetime64[D]").tolist()

        this_uncert_to_plot = inventories_uncert_to_plot[i_inv]

        yerr = None
        if (
            plot_inventory_uncertainty[i_inv] is True
            and this_uncert_to_plot is not None
            and np.any(this_uncert_to_plot > 0)
        ):
            yerr = this_uncert_to_plot.values

        ax.bar(
            time_as_datetime,
            inventory,
            timedelta(days=340 - i_inv * 20),
            edgecolor=inventory.plot_color,
            align="edge",
            fill=False,
            label=f"{'NID' if annex_mode else 'Inventory'} {inventory.year}",
            yerr=yerr,
            error_kw={"ecolor": inventory.plot_color, "capsize": 2},
            zorder=0,
        )
        if plot_trends:
            opt, cov = curve_fit(
                linear, [i.year+i.month/12 for i in time_as_datetime], inventory, sigma=yerr
            )
            ax.plot(
                time_as_datetime,
                linear(np.array([i.year+i.month/12 for i in time_as_datetime]), *opt),
                color="grey",
                linestyle=":",
                label="Inventory trend",
            )

            print(f"Inventory trend for country is: {opt[0]} {unit}")
        tmp = pd.DataFrame(
            {
                "time": time_as_datetime,
                "mean_val": inventory.values,
            }
        )
        
        if yerr is not None:
            tmp["min_unc"] = inventory.values-yerr
            tmp["max_unc"] = inventory.values+yerr
        
        tmp["type"] = "inventory"
        tmp["model"] = f"inventory_{inventory.year}"
        tmp["sector"] = sector
        tmp["country"] = country
        tmp["species"] = species
        res = pd.concat([res, tmp], ignore_index=True)

    return res


def add_sector_barplot(
    ax: Axes, ds_sector: xr.Dataset, variable: str, bottom_values: np.ndarray | float,
    gaps_between_bars: bool = False
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

    if variable == "inv_data" or freq in ["year", "yearly", "unknown"]:
        if gaps_between_bars:
            d = 300
        else:
            d = 365
        time_as_datetime = ds_sector.time.values.astype("datetime64[Y]").tolist()
        width = [
            timedelta(days=d+1) if isleap(date.year) else timedelta(days=d)
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
        label='LULUCF' if sector == 'land' else sector.title(),
        color=sector_colors[sector],
        bottom=bottom_values,
        alpha=0.7,
        width=width,
        align="edge",
    )

    res = pd.DataFrame(
        {
            "time": np.array(time_as_datetime) + offset,
            "mean_val": ds_sector[variable].values + bottom_values,
        }
    )

    res["type"] = variable
    res["sector"] = sector
    res["model"] = ds_sector.attrs["model_label"] if "model_label" in ds_sector.attrs else f"inventory"
    res["country"] = ds_sector.attrs["country"] if "country" in ds_sector.attrs else None
    res["species"] = ds_sector.attrs["species"] if "species" in ds_sector.attrs else None

    return res,width


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
    inventory_filename: str
) -> list[xr.Dataset]:
    """
    Prepare the inventory for the sector barplot.
    Call fluxie.operators.flux_prepare_inventory.retrieve_inventories and change the outputed dataarrays in to dataset (with variable name "inv_data").

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

    inventories, inventories_stdev = retrieve_inventories(
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
    inventories_stdev = [
        inv_std.to_dataset(name="inv_data_stdev") if inv_std else None
        for inv_std in inventories_stdev
    ]

    return inventories, inventories_stdev


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
            whose values are the output of add_inventory_barplot, add_line_plot). The data stored in them is used to infer the ylims.
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

    for max_i, ax in zip(max_cf, axes):
        if fix_y_axes:
            ax.set_ylim(0, np.nanmax(max_cf) * fac)
        else:
            ax.set_ylim(0, max_i * fac)


def add_ylabel(
    ax: Axes,
    s_data: dict[str, dict],
    species: str,
    unit: str,
    plot_type: str,
    annex_mode: bool = False,
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

        print_country_species = f"{country_equivalent.get(kwargs['region'], kwargs['region'])} {s_data.get(species, {}).get('species_print', species)}"
        print_units = f"({unit.replace('2','$_{{2}}$').replace('-1','$^{{-1}}$')})"
        
        if plot_type.split("-")[1] == "posterior":
            if annex_mode:
                ax.set_ylabel(f"{print_country_species} {print_units}"
            )
            else:
             ax.set_ylabel(f"{kwargs['label']}\n{print_country_species} {print_units}"
            )   
            
        elif plot_type.split("-")[1] == "prior":
            if annex_mode:
                ax.set_ylabel(f"Prior\n {print_country_species} {print_units}"
            )
            else:
                ax.set_ylabel(f"Prior\n {print_country_species} {print_units}"
            )
        elif plot_type.split("-")[1] == "inventory":
            if annex_mode:
                ax.set_ylabel(f"{print_country_species} {print_units}")
            else:
                ax.set_ylabel(f"{kwargs['inventory_filename'].replace('_',' ').replace('inventory','Inventory')} {print_country_species} {print_units}")
                
def add_xlims_and_ticks(
    ax: Axes,
    yearly_freq: bool,
    plotted_data_df: dict[str, dict],
    aggreg_month: bool,
    xticks_at_centre: bool = False,
):
    """
    Add x limits, ticks and ticks labels to matplotlib axes. Optimize them by looking at if they are monthly, yearly, or monthly aggregated, and covered time range.
    Args:
        ax: axis to add xlim and xticks to.
        yearly_freq: set to True if the data plotted have a yearly frequency.
        res_dict: dictionnary containing the data plotted. Should have one key per regions plotted, the values being dictionnaries with 3 keys: "inventory", "posterior" and "prior";
            whose values are the output of add_inventory_barplot, add_line_plot). The time data stored in them is used to infer the xlims.
        aggreg_month: if True, the data plotted are supposed to be a monthly aggregated so 12 stciks are created, whose labels are the 3 first letters of each month.
        xticks_at_centre: if True, set the x ticks at the centre of each time period rather than at the beginning.
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
        if xticks_at_centre:
            xticks = xticks.astype("datetime64[D]") + np.timedelta64(182, "D")
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks.astype("datetime64[Y]"))
        else:
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
    fig: Figure,
    set_global_leg: bool,
    annex_mode: bool,
    plot_inventory: bool,
    inventory_years: int | list[int] | None = None,
):
    """
    Add legend.
    Args:
        fig: figure object for which we want to make the legend
        set_global_legend: if True, set one legend object for all subplots, else plot one legend per subplot
        annex_mode: if in annex_mode,
        plot_inventory: used only if set_global_legend is False. When True, avoid enhancing the width of the last object. Not sure that if it is really
            used and if we shouldn't suppress this.
        inventory_years: used only if set_global_legend is False to avoid enhancing the width of the last objects, when plotting mulitple inventories.
    """

    if set_global_leg:

        if isinstance(fig.axes[0], list):
            legend_loc = (0.5, 1.1)
        else:
            legend_loc = (0.5, 1.15)
        handles, labels = fig.axes[0].get_legend_handles_labels()
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
        if type(inventory_years) == list:
            n_inv = len(inventory_years)
        elif type(inventory_years) == int:
            n_inv = 1
        else:
            n_inv = 0
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
                : (-n_inv if plot_inventory else None)
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
    elif country_codes_as_titles == False:
        ax.set_title(f"{print_country}")


def add_vlines(ax: Axes, vline_dates: list[str]):
    """
    Add vertical lines to matplotlib axes at specified dates.
    Args:
        ax: axis to add vertical lines to.
        vline_dates: list of dates (str) at which to add vertical lines.
    """

    for vline_date in vline_dates:
        ax.axvline(
            x=np.datetime64(vline_date),
            color="grey",
            linestyle="dotted",
            linewidth=2.5,
        )


def add_secondary_yaxis(
    ax: Axes,
    s_data: dict[str, dict],
    species: str,
    sector: str,
    unit: str,
    secondary_unit: str,
):
    """
    Add a secondary y-axis to the plot, converting from `unit`
    to `secondary_unit`, including CO2-eq aware conversions.

    Args:
        ax: axis to add secondary y-axis to.
        s_data: dictionnary containing species data. See configs/species_infos.json.
        species: name of the species.
        sector: name of sector plotted.
        unit: original unit of data plotted.
        secondary_unit: unit of the secondary y-axis.
    """

    # Get conversion factor between the two units
    conversion_factor = convert_units_co2eq(
        from_unit=unit, to_unit=secondary_unit, species_info=s_data.get(species, {})
    )

    # Define foward and backward conversion functions
    def unit_to_secondary_unit(y):
        return y * conversion_factor

    def secondary_unit_to_unit(y):
        return y / conversion_factor

    # Create secondary y-axis
    secax = ax.secondary_yaxis(
        "right", functions=(unit_to_secondary_unit, secondary_unit_to_unit)
    )

    # Styling
    sec_color = "darkred"
    secax.tick_params(axis="y", colors=sec_color)
    secax.spines["right"].set_color(sec_color)
    secax.spines["right"].set_linewidth(1.5)

    # Labeling
    add_ylabel(
        secax, s_data, species, secondary_unit, plot_type="country_plot", sector=sector
    )
    secax.yaxis.label.set_color("darkred")
    secax.yaxis.label.set_rotation(270)
    secax.yaxis.labelpad = 20


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
    plot_inventory: bool = False,
    plot_inventory_uncertainty: bool | list[bool] | None = None,
    inventory_years: list[str] | None = None,
    inventory_filename: str = "UNFCCC_inventory",
    data_dir: os.PathLike | None = None,
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
    resample_uncert_correlation: bool = True,
    plot_resample_and_original: bool = False,
    return_res: bool = False,
    rolling_mean: bool | list[bool] = False,
    aggreg_month: bool = False,
    sector: str = "total",
    xticks_at_centre: bool = False,
    plot_grid: bool = True,
    add_vline: list[str] | None = None,
    secondary_units: str | None = None,
    only_overlapping: bool = True,
    plot_trends: bool = False,
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
        plot_inventory_uncertainty: If True and uncertainty available, plots inventory error bars. If a list is provided, should be of same size as inventory_years.
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
        sector: Sector to plot. Can be 'total' or any of the sectors defined in the datasets.
        xticks_at_centre: if True, set the x ticks at the centre of each time period (year or month) rather than at the beginning.
        plot_grid: if True, add a faint grid to each subplot.
        add_vline: list of dates (str) where to add vertical lines on each plot. format 'YYYY-MM-DD'.
        secondary_units: If provided, add a secondary y-axis with these units.
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

    if data_dir is None and plot_inventory:
        raise ValueError("data_dir must be provided to plot inventory data.")

    s_data = config_data.get("species_info", {})
    r_data = config_data.get("regions_info", {})

    plot_regions = format_plot_regions(plot_regions, ds_all)
    unit = get_unit(ds_all)

    plotted_data_df = pd.DataFrame()

    # Compute default for plot_separate_unc and plot_combined_unc if not given
    plot_separate_unc = (
        np.any(plot_separate) if plot_separate_unc is None else plot_separate_unc
    )
    plot_combined_unc = (
        np.any(plot_combined) if plot_combined_unc is None else plot_combined_unc
    )

    # Sel data
    if type(start_date) != list: start_date = [start_date]
    if type(end_date) != list: end_date = [end_date]
    ds_all = {k: ds.sel(time=slice(min(start_date), max(end_date))) for k, ds in ds_all.items()}
    
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
            only_overlapping=only_overlapping,
        )

        # plot posterior and prior (if requested)
        for m, ds_region in ds_to_plot.items():
            highlighted_post = ("combined" in m) & annex_mode
            add_post_unc = (("combined" in m) & plot_combined_unc) | (
                ("combined" not in m) & plot_separate_unc
            )
            if "posterior" in ds_region.data_vars:
                posterior_df = add_line_plot(
                    ax,
                    ds_region,
                    variable="posterior",
                    highlighted_line=highlighted_post,
                    add_unc=add_post_unc,
                    unit=unit,
                    plot_trends=plot_trends,
                )
                plotted_data_df = pd.concat(
                    [plotted_data_df, posterior_df], ignore_index=True
                )

            if add_prior and "prior" in ds_region:
                prior_df = add_line_plot(
                    ax,
                    ds_region,
                    variable="prior",
                    highlighted_line=not annex_mode,
                    add_unc=add_prior_unc,
                )
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
                plot_inventory_uncertainty,
                plot_trends=plot_trends
            )
            plotted_data_df = pd.concat(
                [plotted_data_df, inventory_df], ignore_index=True
            )

        # add vertical lines
        if add_vline is not None:
            add_vlines(ax, vline_dates=add_vline)

        # set y label
        add_ylabel(ax, s_data, species, unit, plot_type="country_plot", sector=sector)

        # set grid
        if plot_grid:
            ax.grid(visible=True, which="major", alpha=0.4)

        # set ax title
        add_title(ax, country, r_data, country_codes_as_titles)

        # add secondary axis
        if secondary_units is not None:
            add_secondary_yaxis(
                ax=ax,
                s_data=s_data,
                species=species,
                sector=sector,
                unit=unit,
                secondary_unit=secondary_units,
            )

    add_ylim(axes, "country", plot_regions, plotted_data_df, fix_y_axes, set_global_leg)
    for ds in ds_to_plot.values():
        if "frequency" in ds.attrs:
            if "yearly" in ds.attrs["frequency"]:
                yearly_freq = True
            else:
                yearly_freq = False
        elif 'year' in resample:
            yearly_freq = True
        else:
            yearly_freq = False
    
    add_xlims_and_ticks(
        axes[-1], yearly_freq, plotted_data_df, aggreg_month, xticks_at_centre
    )

    add_legend(fig, set_global_leg, annex_mode, plot_inventory, inventory_years)

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
    xticks_at_centre: bool = False,
    plot_grid: bool = True,
    gaps_between_bars: bool = False,
    plot_separate: bool = False,
    plot_combined: bool = True,
    only_overlapping: bool = False,
    annex_mode: bool = True,
    vertical_line: str | None = None
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
        sectors: List of emissions sectors to plot.
        xticks_at_centre: If True, moves x ticks to centre of time period.
        plot_grid: If True, plots light grey grid in background.
        gaps_between_bars: If True, reduces bar width to produce gaps, improving readability.
        plot_separate: If True, includes separate models.
        plot_combined: If True, only includes combined model mean.
        only_overlapping: If True, only calculates combinbed model mean for overlapping time periods.
        annex_mode: If True, simplifies some plot labels.
        vertical line: Area to the left of this line is shaded light grey.
    Returns:
        fig: A plot per country/region.
        res_dict : If return_res, return also a dictionnary containing the plotted results
    """
    plot_type = "sector_barplot"
    s_data = config_data.get("species_info", {})
    r_data = config_data.get("regions_info", {})
    unit = get_unit(ds_all)

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
        plot_separate=plot_separate,
        plot_combined=plot_combined,
        only_overlapping=only_overlapping
    )

    freqs = [ds.attrs["frequency"]  if "frequency" in ds.attrs else "unknown" for ds in ds_to_plot.values()]

    if plot_inventory_or_prior == "inventory":
        start_date = str(min([ds.time.values.min() for ds in ds_to_plot.values()]))[:10]
        end_date = str(max([ds.time.values.max() for ds in ds_to_plot.values()]))[:10]
        inv_plot_data = prepare_inventory_sector_barplot(
            sectors,
            np.datetime64(start_date,'Y'),
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
            ax_comp = axes[-1]

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
                    else np.zeros(ds.time.values.shape[0])
                )
                res,width = add_sector_barplot(ax, ds.sel(sector=sector), var, 
                                         bottom_values, gaps_between_bars)
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
            annex_mode=annex_mode
        )
        if plot_inventory_or_prior == "prior":
            ax_comp.legend(ncol=1, borderpad=0.4, columnspacing=1.0)
            add_ylabel(
                ax_comp,
                s_data,
                species,
                unit,
                plot_type=f"{plot_type}-prior",
                region=plot_region,
                annex_mode=annex_mode
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
                        & (plotted_data_df.type == "inv_data")
                        & (plotted_data_df.model == f"inventory")
                    ].mean_val
                    if former_sector
                    else np.zeros(inv[0].time.values.shape[0])
                )
                res,width = add_sector_barplot(ax, inv[0].sel(sector=sector), 'inv_data', 
                                         bottom_values, gaps_between_bars)
                plotted_data_df = pd.concat([plotted_data_df, res], ignore_index=True)
                former_sector = sector
            ax.legend(ncol=1, borderpad=0.4, columnspacing=1.0,loc='upper right')
            add_ylabel(
                ax,
                s_data,
                species,
                unit,
                plot_type=f"{plot_type}-inventory",
                region=plot_region,
                inventory_filename=inventory_filename,
                year=year,
                annex_mode=annex_mode
            )

    # plot grid and legend
    for ax in axes:
        if plot_grid:
            ax.grid(visible=True, which="major", alpha=0.4)
        ax_data.legend(ncol=1, borderpad=0.4, columnspacing=1.0,loc='upper right')

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

    # add shading based on vertical line
    if vertical_line:
        ax_data.fill_between(np.arange(np.datetime64(ds.time.values[0])-np.timedelta64(width[0]),
                                       np.datetime64(vertical_line),np.timedelta64(10,'D')),
                             0,ax_data.get_ylim()[1],
                             color='dimgrey',alpha=0.2,zorder=0,label='2 sites')

    # set xlim and xticks
    yearly_freq = ("year" in freqs) or ("yearly" in freqs)

    add_xlims_and_ticks(axes[-1], yearly_freq, plotted_data_df, aggreg_month=False,
                        xticks_at_centre=xticks_at_centre)

    return fig, plotted_data_df


def plot_all_species_stacked_bar(
    all_species: list[str],
    ds_all_flux_scaled: dict[str, xr.Dataset],
    regions: list[str],
    config_data: dict[str, dict] = {},
    model_colors: dict[str, str] = {},
    model_labels: dict[str, str] = {},
    start_date: str | None = None,
    end_date: str | None = None,
    inventory_years: list[str] | None = None,
    inventory_filename: str = "UNFCCC_inventory",
    plot_inventory_uncertainty: bool = True,
    data_dir: str | None = None,
    sector: str = "total",
    y_lim: list[float] | None = None,
) -> Figure:
    """
    Stacked bar plot of posterior fluxes, summed over all species, for a single region, for a range of models.
    Args:
        all_species: List of gas species, e.g. ['ch4','n2o'].
        ds_all_flux_scaled: xarray datasets of fluxes, scaled and sliced between
            chosen dates.
        regions: Country or regions to plot, e.g. ['GBR']
        config_data: Dictionary with settings read from json file. Use json filenames as keys.
        model_colors: Models and corresponding colours used to plot the model.
        start_date: Start dates of the data to plot (used to slice inventory data).
        end_date: Start dates of the data to plot (used to slice inventory data).
        inventory_years: List of inventory data from different years to include. If None, only plots the most recent inventory data.
        inventory_filename: Name of inventory file: {inventory_filename}_{species}_{inventory_year}
        data_dir: Path to top data directory, used to read inventory data files.
        sector: Sector to plot.
        country_flux_units_print: Units for fluxes to be printed on inventory bars.
        y_lim: Y-axis limits for the plot.
    Returns:
        fig: A stacked bar plot of all species for the region.
    """

    s_data = config_data.get("species_info", {})
    r_data = config_data.get("regions_info", {})
    species_colors = config.species_color_palette

    unit = []

    for s in all_species:
        unit.append(get_unit(ds_all_flux_scaled[s]))
    if all(x == unit[0] for x in unit):
        country_flux_units_print = unit[0]
    else:
        raise ValueError("Units for all species' datasets are not equal.")
    ds_to_plot = {}
    inventories_to_plot = {}

    models = []

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for s, species in enumerate(all_species):

        ds_all_region = extract_region_flux(
            ds_all_flux_scaled[species], regions, r_data, sectors=sector
        )
        ds_to_plot[species] = prepare_data_to_plot(
            ds_all_region=ds_all_region,
            model_labels=model_labels,
            model_colors=model_colors,
            plot_separate=True,
            plot_combined=False,
            resample="year",
            rolling_mean=False,
            plot_resample_and_original=False,
            resample_uncert_correlation=False,
            aggreg_month=False,
        )

        inventories_to_plot[species], inventories_uncert_to_plot = retrieve_inventories(
            data_dir,
            regions,
            species,
            start_date,
            end_date,
            country_flux_units_print,
            s_data,
            r_data,
            inventory_years,
            inventory_filename,
            sectors=sector,
        )

        models.append(list(ds_to_plot[species].keys())[0])

        if len(list(ds_to_plot[species].keys())) > 1:
            logger.warning(
                f"Only the first model {list(ds_to_plot[species].keys())} will be plotted because this function is "
                + "currently only set up to plot inventory data and model output from one model."
            )

        this_uncert = inventories_uncert_to_plot[0]

        posterior_diff = (
            ds_to_plot[species][models[s]]["posterior_upper"].values
            - ds_to_plot[species][models[s]]["posterior_lower"].values
        )

        if this_uncert is None:
            this_uncert = np.zeros_like(inventories_to_plot[species][0].values)

        if s == 0:
            inv_plot_times = inventories_to_plot[species][0].time.values
            plot_times = ds_to_plot[species][models[s]].time.values.astype(
                "datetime64[Y]"
            )
            uncert_combined = posterior_diff
            if plot_inventory_uncertainty:
                inventories_uncert_combined = this_uncert
        else:
            uncert_combined = np.sqrt(uncert_combined**2 + posterior_diff**2)
            if plot_inventory_uncertainty:
                inventories_uncert_combined = np.sqrt(
                    inventories_uncert_combined**2 + this_uncert**2
                )

    width = np.timedelta64(150, "D")

    for s, species in enumerate(all_species):

        if s == 0:
            bottom = None
            inv_bottom = None
            inventory_label = f"{inventory_years[0]} Inventory"
        else:
            bottom = flux_sum
            inv_bottom = inventory_sum
            inventory_label = None

        if s == (len(all_species) - 1):
            uncert = uncert_combined / 2.0
            if plot_inventory_uncertainty:
                inventories_uncert = inventories_uncert_combined #/ 2.0
            else:
                inventories_uncert = None
        else:
            uncert = None
            inventories_uncert = None

        ax.bar(
            inv_plot_times + width,
            inventories_to_plot[species][0].values,
            width=width,
            bottom=inv_bottom,
            color="lightgrey",
            edgecolor="grey",
            label=inventory_label,
            yerr=inventories_uncert,
            error_kw={"capsize": 2},
        )

        ax.bar(
            plot_times,
            ds_to_plot[species][models[s]]["posterior"].values,
            width=width,
            bottom=bottom,
            color=species_colors[species],
            label=s_data.get(species, {}).get("species_print", species),
            yerr=uncert,
            error_kw={"capsize": 2},
            alpha=0.8,
        )

        if s == 0:
            flux_sum = ds_to_plot[species][models[s]]["posterior"].values
            inventory_sum = inventories_to_plot[species][0].values
        else:
            flux_sum += ds_to_plot[species][models[s]]["posterior"].values
            inventory_sum += inventories_to_plot[species][0].values

    ax.set_xticks(plot_times + (width / 2))
    ax.set_xticklabels((plot_times + (width / 2)).astype("datetime64[Y]"))

    # ax.set_ylabel(country_flux_units_print)
    add_ylabel(
        ax,
        s_data,
        species=f"{regions} total",
        unit=country_flux_units_print,
        plot_type="country_plot",
        sector=sector,
    )
    if y_lim:
        ax.set_ylim(y_lim)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        ncol=4,
        loc="upper right",
        borderpad=0.4,
        columnspacing=1.0,
        fontsize=12,
    )

    return fig,flux_sum,uncert,inventory_sum,inventories_uncert,plot_times
