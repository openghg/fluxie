import logging
from typing import Literal

import numpy as np
import xarray as xr
from datetime import date, timedelta
from calendar import month_abbr

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.dates import MonthLocator, YearLocator
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter

from fluxy import config
from fluxy.plots.utils import set_min_decimal_points
from fluxy.types import VariableType
from fluxy.operators.select import (
    FrequencyType,
    clean_timeseries_missing_data,
    get_site_index,
    get_unique_site_height_pairs,
)
from fluxy.plots.utils import set_min_decimal_points

logger = logging.getLogger(__name__)


def plot_mf_timeseries(*args, **kwargs) -> plt.Figure:
    # Solve the legacy position of the include argument
    LEGACY_INCLUDE_POSITION = 9
    NEW_INCLUDE_POSITION = 1
    default_include = {
        "mf_observed": None,
        "mf_posterior": "percentile_mf_posterior",
    }
    args = list(args)
    if len(args) > LEGACY_INCLUDE_POSITION:
        # Move the include to the correct position
        include_arg = args.pop(LEGACY_INCLUDE_POSITION)
        args.insert(NEW_INCLUDE_POSITION, include_arg)
    elif len(args) > NEW_INCLUDE_POSITION:
        # Need to add the include back to args
        args.insert(
            NEW_INCLUDE_POSITION,
            kwargs["include"] if "include" in kwargs else default_include,
        )
        if "include" in kwargs:
            kwargs.pop("include")
    else:
        # Only kwargs
        if "include" not in kwargs:
            kwargs["include"] = default_include

    return plot_timeseries(*args, **kwargs)


def _prepare_aggreg_month_var(da_var: xr.DataArray | xr.Dataset) -> xr.Dataset:
    """
    Aggregate by month the dataset/array variable(s). The outputed dataset has two dimensions: "time" (array from 1 to 12)
    and "percentile". The percentile coordinate has 3 values: "mean" corresonding to the mean value for the month and
    "lower"/"upper" corresponding to the 0.159/0.841 percentile of the variable for the month.
    Args:
        da_var: datarray/dataset to aggregate by month
    Returns:
        dataset of variable(s) aggregate by month, including their mean and spread.
    """
    if isinstance(da_var, xr.DataArray):
        da_var = da_var.to_dataset()
    mean = da_var.groupby("time.month").mean().rename({"month": "time"})
    mean = mean.expand_dims(
        {
            "percentile": [
                "mean",
            ]
        }
    )

    unc = (
        da_var.groupby("time.month")
        .quantile([0.159, 0.841])
        .rename({"month": "time", "quantile": "percentile"})
    )
    unc["percentile"] = ["lower", "upper"]

    return xr.merge([mean, unc], compat="no_conflicts", join="outer")


def _prepare_var(
    ds: xr.Dataset, var: str, unc_var: str | None, model: str = None
) -> xr.Dataset:
    """
    Format the required variable <var> and its uncertainty <unc_var> in one dataset containing one variable (named <var>).
    The variable thus created has two dimensions: "index" (corresponding to time) and "percentile". "percentile" can take 1 to 4
    values: "mean" (always present) being the main value, "lower" and "upper" (optionnals) which are the "upper" and "lower"
    boundaries of the associated uncertainty, and "std" (optionnal) which is the std / one side associated uncertainty.
    Args:
        ds: dataset containing <var> and <unc_var>
        var: main variable
        unc_var: uncertainty corresponding to <var> (optionnal)
        model: model name, just used to help if there is an error.
    Return:
        dataset containing one variable and 2 dimensions: index and percentile.
    """
    mean = ds[[var]]
    mean = mean.expand_dims(
        {
            "percentile": [
                "mean",
            ]
        }
    )

    if not unc_var:
        return mean

    # Determine uncertainty
    if unc_var not in ds:
        unc_var_in = unc_var
        if "percentile" in unc_var:
            unc_var = unc_var.replace("percentile", "stdev")

        elif "stdev" in unc_var:
            unc_var = unc_var.replace("stdev", "percentile")

        if unc_var not in ds:
            raise KeyError(
                f"Variables {unc_var_in} and {unc_var} not found for model {model}."
            )
        logger.warning(
            f"Variable {unc_var_in} not found {ds.attrs.get('model','')} so reading uncert from {unc_var}."
        )

    # Creating variable
    if unc_var.split("_")[0] == "percentile":
        unc = ds[unc_var]
        unc["percentile"] = ["lower", "upper"]
    elif unc_var.split("_")[-1] in ["prior", "posterior"]:
        unc_lower = (ds[var] - ds[unc_var]).expand_dims({"percentile": ["lower"]})
        unc_upper = (ds[var] + ds[unc_var]).expand_dims({"percentile": ["upper"]})
        unc = xr.concat([unc_lower, unc_upper],dim="percentile")
        unc.name = unc_var
    else:
        unc = ds[unc_var].expand_dims({"percentile": ["std"]})

    unc = unc.to_dataset()

    unc = unc.rename({unc_var: var})

    return xr.merge([mean, unc], compat="no_conflicts", join="outer")


def _retrieve_variable(ds, var, unc_var):
    """
    Infer variable from dataset. Possibility are to substract the boundary conditions to a variable (var should then finis with "_above_BC")
    or substract a variable from one another (var should then finis with "_diff").
    NOTE: the attribute of the new variable will be copied from the first of the original variable of the dataset used to infer it.
    Args:
        ds: dataset to infer the new variable from
        var: new variable to infer. Options are '<var1>_above_BC' and '<var1>_<var2>_diff' with <var1> and <var2> being one of
            'prior' (for 'mf_prior'), 'posterior' (for 'mf_posterior'), 'observed' (for 'mf_observed').
        unc_var: uncertainty to associated with the new variable. Currently not implemented. Will raise an error if not None.
    Returns:
        ds: dataset with new variable inside.
    """
    if unc_var:
        raise NotImplementedError(
            "No uncertainty can be plotted for `{var}` as this variable is inferred from others. Please set `include` to `{'{var}': None}`."
        )

    acceptable_var = ["prior", "posterior", "observed"]

    if var.endswith("_above_BC"):
        if "posterior" in var:
            bc = ds["mf_bc_posterior"]
        else:
            bc = ds["mf_bc_prior"]

        var0 = var.split("_")[0]
        if var0 not in acceptable_var:
            raise NotImplementedError(
                "The variable names currently available when plotting a variable value above BC are of the form '<var>_above_BC', "
                "with var being one of 'prior' (for 'mf_prior'), 'posterior' (for 'mf_posterior'), 'observed' (for 'mf_observed')."
            )
        ds[var] = ds[f"mf_{var0}"] - bc
        ds[var].attrs = ds[f"mf_{var0}"].attrs

    elif var.endswith("_diff"):
        var1, var2 = var.split("_")[:2]
        if var1 not in acceptable_var or var2 not in acceptable_var:
            raise NotImplementedError(
                "The variable names currently available when plotting a difference of two variables are of the form '<var1>_<var2>_diff', "
                "with var1 and var2 being one of 'prior' (for 'mf_prior'), 'posterior' (for 'mf_posterior'), 'observed' (for 'mf_observed')."
            )
        ds[var] = ds[f"mf_{var1}"] - ds[f"mf_{var2}"]
        ds[var].attrs = ds[f"mf_{var1}"].attrs

    else:
        raise NotImplementedError(
            "Currently, the variables accepted are either (1) those present in the dataset, "
            "or (2) a difference of some of the dataset variables, in which case the variable in `include` must finished by '_diff'; "
            "or (3) a difference of one of the dataset variables and the boundary conditions, in which case the variable in `include` must finished by '_above_BC'."
        )

    return ds


def _prepare_data_to_plot(
    ds_all: dict[str, xr.Dataset],
    include: str | dict[str, str | None] | list | tuple,
    diff_include: None | list,
    aggreg_month: bool,
    time_freq_min: FrequencyType,
    plot_type: Literal["separate", "together", "diff"],
) -> dict[str, xr.Dataset]:
    """
    Create dictionnary of datasets containing all the data that will be plotted.
    Args:
        ds_all: dictionnary of dataset from which the variable are taken
        include: variables to plot in the main panel. If is a dictionnary : the keys are the variables to plot and the
            values the uncertainty that will be shaded around them.
        diff_include: variables that will be plot in the secondary (histogram) panel.
            In this function, they are treated as the ones passed with `include` parameter.
        time_freq_min: Time frequency minimum of the timeserie that should be shown as continous line.
            If the frequency is lower than this, the line will be discontinous. For more information,
            see :py:func:`fluxy.operators.select.clean_timeseries_missing_data`
        plot_type: type of plot. Just used to check that no uncertainty plotting is asked for in `include` when `plot_type="diff"`.
    Returns:
        data_to_plot: dictionnary of dataset containing the variables to plot. Each variable as two dimensions: "index" (corresponding to time, sometimes named "time")
            and "percentile". "percentile" can take 4 values: "mean" (always present) being the main value, "lower" and "upper" (optionnals) which are the
            "upper" and "lower" boundaries of the associated uncertainty, and "std" (optionnal) which is the std / one side associated uncertainty.
    """
    if not include:
        raise ValueError(
            "The include dictionary is empty. Please provide variables to include in the plot."
        )
    if isinstance(include, str):
        all_var = {include: None}
    elif isinstance(include, (list, tuple)):
        all_var = {var: None for var in include}
    else:
        all_var = include.copy()

    data_to_plot = {m: xr.Dataset() for m in ds_all.keys()}
    if isinstance(diff_include, list):
        all_var.update({var: None for var in diff_include if var not in all_var.keys()})
        if "mf_observed" not in all_var.keys():
            all_var["mf_observed"] = None

    for m, ds in ds_all.items():

        # Check there is only one site in the dataset
        if len(np.unique(ds.get("number_of_identifier", 0))) > 1:
            raise ValueError(
                f"Dataset {m} contains more than one site. "
                "Use slice_site to select a single site."
            )

        for var, unc_var in all_var.items():
            if var not in ds.keys():
                ds = _retrieve_variable(ds, var, unc_var)

        # Clean the time dimension
        ds_clean = clean_timeseries_missing_data(
            ds, variables_nans=all_var.keys(), min_freq=time_freq_min
        )

        for var, unc_var in all_var.items():

            if aggreg_month:
                if (
                    var.split("_")[0] != "mf"
                    and not var.endswith("_above_BC")
                    and not var.endswith("_diff")
                ):
                    raise NotImplementedError(
                        "`aggreg_month` disabled this for variable that are not mole fractions (i.e. names not starting with `mf`)."
                    )
                ds_var = _prepare_aggreg_month_var(ds_clean[var])
                if all_var[var] is not None:
                    logger.warning(
                        f"`{all_var[var]}` present as value of include dict for {var} is overwritten as you put `aggreg_month=True`."
                        + " The uncertainty plotted is the 0.159 and 0.851 percentile of the variable for the corresponding month."
                    )
            else:
                ds_var = _prepare_var(ds_clean, var, unc_var, model=m)

            if unc_var:
                if plot_type == "diff":
                    raise ValueError(
                        f"Option plot_type='diff' does not accept uncertainties. Replace '{unc_var}' by None."
                    )
            data_to_plot[m] = xr.merge(
                [data_to_plot[m], ds_var], compat="no_conflicts", join="outer"
            )

    return data_to_plot


def _set_labels_and_colors(
    ds_dict: dict[str, xr.Dataset],
    model_labels: dict[str, str],
    model_colors: dict[str, list],
    plot_type: Literal["separate", "together", "diff"],
) -> dict[str, xr.Dataset]:
    """
    Set labels and colors that will be used by plot_timeseries and plot_histogram as attributes of the variables dataset.
    For variables "mf_observed" and "observed_above_BC", the color will be black (and not one of model_colors) if more than one variable is plotted.
    Args:
        ds_dict: dictionnary containing the dataset with the variables to be plotted (and only them).
        model_labels: dictionnary with same keys as ds_dict (unless plot_type="diff") that contains corresponding label
        model_colors: dictionnary with same keys as ds_dict (unless plot_type="diff") that contains list of colors to be used with each model
        plot_type: type of plot. If diff, look in model_labels for the labels of the two models used for the diff to construct the new label.
            Otherwise use directly the model_labels value corresponding to the dataset from ds_dict
    Return:
        ds_dict: dictionnary of datasets where the label and color have been added as attributes of the dataset / dataset variables.
    """
    if model_colors is None:
        model_colors = config.set_model_colors(ds_dict.keys())

    for m in ds_dict.keys():
        vars_to_plot = ds_dict[m].data_vars.keys()
        if plot_type == "diff":
            mdiff0, mdiff1 = m.split("--")
            model_label = f"{model_labels[mdiff0]} - {model_labels[mdiff1]}"
            model_color = model_colors[mdiff0]
        else:
            model_label = model_labels.get(m, m)
            model_color = model_colors[m]

        for var in vars_to_plot:
            default_index = 1 if "posterior" in var else 0
            plot_color = model_color[config.mf_color_index.get(var, default_index)]
            if var in ["mf_observed", "observed_above_BC"] and len(vars_to_plot) > 1:
                plot_color = "black"

            plot_label = f"{model_label} {config.mf_labels.get(var, var)}"

            ds_dict[m][var].attrs.update(
                {"plot_label": plot_label, "plot_color": plot_color}
            )

        ds_dict[m].attrs["label"] = model_label

    return ds_dict


def _create_figure(
    models: list[str],
    plot_type: Literal["separate", "together", "diff"],
    histogram_type: str | None,
    aggreg_month: bool,
) -> tuple[Figure, Axes]:
    """
    Create figure and axes with size, number of columns and rows depending of the arguments.
    Args:
        models: list of models to use. Is the number of rows if plot_type=="separate". Otherwise unused.
        plot_type: Options are "separate" (correspond to a plot where all models are separated), "together (all model on the same plot), "diff" (diff of the two models)
            If plot_type is one of "together" or "diff", the figure will have one row, otherwise (plot_type="separate"), the number of rows will be the number of models.
        histogram_type: type of histogramm that will be plotted. If not none, 2 columns will be created, unless aggreg_month is True.
        aggreg_month: Wheter or not the data will be aggregated by month. If True, the plot size will be smaller than if False, and the histogram_type parameter will be ignored.
    Return:
        fig, ax: matplotlib figure and axes object with appropriate sizes.
    """
    if plot_type == "separate":
        nrows = len(models)
    elif plot_type in ["together", "diff"]:
        nrows = 1
    else:
        raise ValueError(
            f"Option {plot_type} not implemented. Set plot_type to 'separate', 'together' or 'diff'."
        )

    ncols = (
        2 if (histogram_type and histogram_type != "none") and not aggreg_month else 1
    )
    lenght = 8 if aggreg_month else 15

    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(lenght, nrows * 3),
        gridspec_kw={"width_ratios": [0.8, 0.2]} if ncols == 2 else {},
        constrained_layout=True,
        sharey="row" if histogram_type == "violin" else False,
        sharex="col",
        squeeze=False,
    )

    return fig, ax


def _get_unit(ds_dict: dict[str, xr.Dataset]) -> str:
    """
    Check if all variables in the datasets from the dictionnary have the same unit. If so returns it else raise an error.
    Args:
        ds_dict: dictionnary of dataset containing the variables to check (and only those).
    Return:
        the unit of the variables
    """

    plot_units = list()

    for m in ds_dict:
        for var in ds_dict[m].data_vars:
            plot_units.append(ds_dict[m][var].attrs["units"])

    plot_units = list(set(plot_units))
    if len(plot_units) != 1:
        raise ValueError(
            f"{ds_dict[m].data_vars.keys()} in {ds_dict.keys()} do not have the same units. So far, the following were found: {plot_units}."
            + "Select only one model to plot, or run 'slice_mf' with 'mf_units_print' equal to a valid mole fraction unit before running 'plot_timeseries'."
        )

    return plot_units[0]


def add_xlims_and_ticks(
    ax: Axes,
    yearly_freq: bool,
    res_dict: dict[str, dict[str, xr.Dataset] | xr.Dataset],
    aggreg_month: bool,
    rotate_xticks: bool = False,
):
    """
    Add x limits, ticks and ticks labels to matplotlib axes. Optimize them by looking at if they are monthly, yearly, or monthly aggregated, and covered time range.
    Args:
        ax: axis to add xlim and xticks to.
        yearly_freq: set to True if the data plotted have a yearly frequency.
        res_dict: dictionnary containing the data plotted.
        aggreg_month: if True, the data plotted are supposed to be a monthly aggregated so 12 ticks are created, whose labels are the 3 first letters of each month.
        rotate_xticks: if True, rotate xticks with an angle of 70 degree.
    """
    if aggreg_month:
        ax.set_xticks(np.arange(1, 13))
        ax.set_xticklabels(list(month_abbr)[1:])
        return

    min_x, max_x = date(2100, 1, 1), date(1900, 1, 1)
    for key_1 in res_dict.keys():
        if isinstance(res_dict[key_1], dict):
            for key_2 in res_dict[key_1].keys():
                for key_3 in res_dict[key_1][key_2].keys():
                    time = res_dict[key_1][key_2][key_3]["time"]
                    min_x = np.nanmin([*time, min_x])
                    max_x = np.nanmax([*time, max_x])
        else:
            time = res_dict[key_1].time.values.astype("datetime64[D]").tolist()
            min_x = np.nanmin([*time, min_x])
            max_x = np.nanmax([*time, max_x])

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
        xticks = [date(year, 1, 1) for year in range(min_x.year, max_x.year, step)]
        if (max_x.year - min_x.year) % step == 0:
            xticks = np.append(xticks, max_x)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks.astype("datetime64[Y]"))
    else:
        ax.xaxis.set_minor_locator(MonthLocator())
    ax.xaxis.set_major_locator(YearLocator())

    if rotate_xticks:
        ax.tick_params(axis="x", rotation=70)

    ax.set_xlim(xlim)


def plot_timeseries(
    ds_all: dict[str, xr.Dataset],
    include: VariableType,
    species: str | None = None,
    site: str | None = None,
    model_colors: dict[str, str] | None = None,
    model_labels: dict[str, dict] = {},
    config_data: dict[str, dict] = {},
    annotate_coords: dict[int, list] = {},
    presentation_mode: bool = False,
    plot_type: Literal["separate", "together", "diff"] = "separate",
    diff_include: list[str] | None = None,
    y_lim: None | tuple[float | None, float | None] = None,
    n_bins: int = 30,
    time_freq_min: FrequencyType = None,
    intake_height: int | None = None,
    histogram_type: Literal["hist", "violin", "none"] | None = "hist",
    hist_kwargs: dict[str, any] = {},
    aggreg_month: bool = False,
):
    """
    Timeseries plots of observations, modelled mole fractions, baseline mf and/or
    uncertainties from each model.
    Includes a histogram of the variables plotted in the timeseries plot or
    the difference between specified variables and observations.

    Args:
        ds_all (dictionary of datasets):
            xarray datasets, scaled and sliced between chosen dates and for
            chosen site.
        include (dict of str):
            Dictionary keys are variables to include in the plot.
            The respective values are the uncertainty variables to plot as error bar/uncertainty band.
        species (str):
            Gas species, e.g. 'ch4'.
        site (str):
            Obs site, e.g. 'MHD'.
        model_colors (dict of str):
            Models and corresponding colours used to plot the model.
        config_data (dict of dict):
            Dictionary with settings read from json file.
            Use json filenames as keys.
        annotate_coords (dict of lists):
            Coordinates to annotate histogram.
        presentation_mode (logical) (optional):
            If True, adjust annotation position and xlabel rotation to accomodate bigger fonts.
        diff_include (list of str):
            Variables included in the 'obs - variable' difference histogram.
            If None, plots the histogram of the variables specified in include.
        y_lim (list of float, optional):
            Mix/max y axis limits to apply to all plots.
        n_bins (int):
            Number of bins to use in the histogram.
        time_freq_min (FrequencyType, optional):
            Time frequency minimum of the timeserie that should be shown as continous
            line. If the frequency is lower than this, the line will be discontinous.
            see :py:func:`fluxy.operators.select.clean_timeseries_missing_data`
            for more information.
    Returns:
        fig (figure):
            A timeseries and histogram plot for each model included.
    """

    models = ds_all.keys()

    species_info = config_data.get("species_info", {}).get(species, {})

    # Check the include dictionary
    data_to_plot = _prepare_data_to_plot(
        ds_all, include, diff_include, aggreg_month, time_freq_min, plot_type
    )
    data_to_plot = _set_labels_and_colors(
        data_to_plot, model_labels, model_colors, plot_type
    )
    unit = _get_unit(data_to_plot)

    min_mf = np.inf
    max_mf = -np.inf

    # Create figure
    fig, ax = _create_figure(ds_all.keys(), plot_type, histogram_type, aggreg_month)

    logger.info(
        f"Plotting {len(models)} models with {len(include.keys())} variables in {plot_type} mode."
    )

    # Loop over all models
    for i, m in enumerate(models):

        # Define plot_type specific settings
        iax = i if plot_type == "separate" else 0

        # Loop over all variables to plot
        for var in include.keys():

            ds_plot = data_to_plot[m][var]

            # Define plotting color
            x, y = ds_plot.time.values, ds_plot.sel(percentile="mean").values
            kwargs = {
                "alpha": 0.8,
                "color": ds_plot.attrs["plot_color"],
                "label": ds_plot.attrs["plot_label"],
            }

            if var in ["mf_observed", "observed_above_BC"] or plot_type == "diff":
                # Make scatter plot
                ax[iax, 0].scatter(
                    x,
                    y,
                    s=8,
                    marker="s",
                    **kwargs,
                )

            else:
                # Make line plot
                ax[iax, 0].plot(
                    x,
                    y,
                    linewidth=2.0,
                    marker="o",
                    markersize=1.5,
                    **kwargs,
                )

            if ds_plot.percentile.size == 3:
                ax[iax, 0].fill_between(
                    x,
                    y1=ds_plot.sel(percentile="lower"),
                    y2=ds_plot.sel(percentile="upper"),
                    alpha=0.2,
                    color=ds_plot.attrs["plot_color"],
                )
            elif ds_plot.percentile.size == 2:
                ax[iax, 0].errorbar(
                    x,
                    y=ds_plot.sel(percentile="mean"),
                    yerr=ds_plot.sel(percentile="std"),
                    alpha=0.4,
                    fmt="none",
                    color=ds_plot.attrs["plot_color"],
                )

        # Plot histogram
        if ax.shape[1] == 2:
            plot_histogram(
                ax[iax, 1],
                data_to_plot[m].sel(percentile="mean"),
                m,
                list(data_to_plot[m].data_vars),
                diff_include,
                presentation_mode,
                annotate_coords,
                annotate_index=i,
                plot_type=plot_type,
                n_bins=n_bins,
                violin=histogram_type == "violin",
                **hist_kwargs,
            )

        # Get timeseries y-axis minimum and maximum
        min_mf = min(min_mf, ax[iax, 0].get_ylim()[0])
        max_mf = max(max_mf, ax[iax, 0].get_ylim()[1])

        # Set timeseries title
        if plot_type in ["separate", "diff"]:
            plot_title = data_to_plot[m].attrs["label"]
        elif plot_type == "together":
            plot_title = "All models"
        ax[iax, 0].set_title(plot_title)

        # Set timeseries y-axis label and legend

        height_label = ""
        if intake_height is not None:
            height_label = f"-{intake_height}m"

        ax[iax, 0].set_ylabel(
            " ".join(
                [
                    species_info.get("species_print", ""),
                    (site if site else "") + height_label,
                    f"({unit})",
                ]
            )
        )

        leg = ax[iax, 0].legend(
            ncol=2, framealpha=0.75, borderpad=0.2, columnspacing=1.0
        )
        try:
            for l in leg.legend_handles:
                l.set_linewidth(5.0)
        except:
            for l in leg.legendHandles:
                l.set_linewidth(5.0)

        if len(data_to_plot[m].time) <= 1:
            continue

        add_xlims_and_ticks(
            ax[iax, 0],
            yearly_freq=False,
            res_dict=data_to_plot,
            aggreg_month=aggreg_month,
            rotate_xticks=presentation_mode,
        )

        ax[iax, 0].grid(color="lightgrey", linestyle="-", linewidth=0.7)
        ax[iax, 0].set_axisbelow(True)

    if y_lim is None:
        y_lim = [min_mf - 0.05 * (max_mf - min_mf), max_mf + 0.1 * (max_mf - min_mf)]

    # Set all the axes to the same y-axis limits
    for iax, ax0 in enumerate(ax[:, 0]):
        ax0.set_ylim(y_lim)

        if ax.shape[1] == 2 and (diff_include is None or len(diff_include) == 0):
            method = "set_ylim" if histogram_type == "violin" else "set_xlim"
            getattr(ax[iax, 1], method)(y_lim)

    logger.info(
        "If annotations in the histograms are not displaying correctly, adjust annotate_coords."
    )

    return fig


def plot_sites_timeseries(
    ds_all: dict[str, xr.Dataset],
    var: str,
    species: str,
    start_date: str,
    end_date: str,
    model_colors: dict[str, str],
    model_labels: dict[str, str],
    config_data: dict[str, dict],
    margin: float = 0.1,
    separate_by_height: bool = False,
):
    """
    Plot the timeseries of data available for each site and model.

    Args:
        ds_all (dictionary xarray Datasets):
            Dictionnary of xarray returned by read_output_model.
        var (str):
            Var for which the timeseries should be plotted
        species (str):
            Gas species, e.g. 'ch4'.
        start_date (str):
            Date to plot data from, e.g. '2021-01-01'
        end_date (str):
            Date to plot data to, e.g. '2022-01-01' would include all
            data up to 2021-12-31.
        model_colors (dict of str):
            Models and corresponding colours used to plot the model.
        model_labels (dict of dict):
            Dictionary with model lables.
        config_data (dict of dict):
            Dictionary with settings read from json file.
            Use json filenames as keys.
        margin (float):
            Horizontal space between datapoints from different models.
        separate_by_height (bool):
            If True, separates obs by intake height and by site.
    """

    models = ds_all.keys()
    dt_start_date = np.datetime64(start_date)
    dt_end_date = np.datetime64(end_date)
    model_labels_copy = model_labels.copy()

    # create list of grouped site-height pairs
    site_list = get_unique_site_height_pairs(ds_all, separate_by_height)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(0.7 * len(site_list), 8))

    assert margin < 0.5, "Margin must be smaller than 0.5"
    assert margin > 0, "Margin must be positive"

    if len(models) > 1:
        model_offset = (1 - 2 * margin) / (len(models) - 1)
    else:
        model_offset = 1

    for site_iter, (site, height) in enumerate(site_list):
        if site_iter != 0:
            # Add grey vertical line between sites
            ax.plot(
                [site_iter - 0.5, site_iter - 0.5],
                [dt_start_date, dt_end_date],
                c="gray",
                ls="-",
                lw=1,
            )

        for i, m in enumerate(models):

            site_index = get_site_index(ds_all[m], site)

            if site_index is None:
                continue
            # Scatter a vertical line at times where data is available
            mask = (ds_all[m]["number_of_identifier"] == site_index) & (
                ds_all[m][var].notnull()
            )
            if separate_by_height:
                mask &= ds_all[m]["intake_height"] == height
            data = ds_all[m]["time"].where(mask, drop=True)
            ax.scatter(
                (site_iter + model_offset * i - 0.5 + margin) * np.ones(data.size),
                data,
                c=model_colors[m][0],
                s=2,
                label=model_labels_copy[m],
            )

            # Erase label so it shows only once
            model_labels_copy[m] = None

    # Define plot settings
    ax.set_ylim(
        dt_start_date - np.timedelta64(1, "D"), dt_end_date + np.timedelta64(1, "D")
    )

    ax.set_xticks(np.arange(len(site_list)))
    xticklabels = [
        f"{s}\n{int(h)}m" if separate_by_height else s for (s, h) in site_list
    ]
    ax.set_xticklabels(xticklabels)

    if (
        int(dt_end_date.astype("datetime64[M]") - dt_start_date.astype("datetime64[M]"))
        > 12
    ):
        ax.yaxis.set_minor_locator(MonthLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(YearLocator())
    else:
        ax.yaxis.set_major_locator(MonthLocator())
    ax.yaxis.grid(True, which="major")

    ax.set_xlim(-0.5, len(site_list) - 0.5)

    plt.legend(loc="upper left", markerscale=4, bbox_to_anchor=(1, 1))

    species_info = config_data.get("species_info", {}).get(species, {})
    fig.suptitle(
        (
            f'Timestamps with {species_info.get("species_print","")} assimilated observations between'
            f"\n{start_date} and {end_date}"
        )
    )

    return fig


def plot_histogram(
    ax: Axes,
    ds: xr.Dataset,
    model: str,
    vars_to_plot: list[str],
    diff_include: list[str] | None,
    presentation_mode: bool,
    annotate_coords: dict[int, list],
    annotate_index: int,
    plot_type: Literal["separate", "together", "diff"],
    n_bins: int = 30,
    violin: bool = False,
    **kwargs,
) -> None:
    """
    Plots a histogram on a specified axis.

    Args:
        axis (matplotlib.axes.Axes):
            Matplotlib subplot axis where the histogram should be plotted.
        ds (xarray dataset):
            Dataset with results from a particular model.
        model (str):
            Model name tag to which the dataset ds refers to.
            i.e. '<inversionModel>_<optional_identifying_tags>', preceded by subdirectory if applicable
        vars_to_plot (list of str):
            Variables plotted in the timeseries plot.
            These variables are directly plotted in the histogram if diff_include is None.
        diff_include (list of str):
            Variables included in the 'obs - variable' difference histogram.
            If None, plots the histogram of the variables specified in vars_to_plot.
        presentation_mode (logical) (optional):
            If True, adjust annotation position to accomodate bigger fonts.
        annotate_coords (dict of lists):
            Coordinates to annotate histogram.
        annotate_index (int):
            Model index. Used to specify annotation location if plot_type == "together".
        plot_type (str):
            Type of timeseries plot in which the histogram will be plotted.
            Options for "separate", "together" and "diff".
    """

    if not annotate_coords:
        annotate_coords = config.set_print_settings(presentation_mode)

    # Get histogram variables and legend
    if diff_include:
        hist_to_plot = diff_include
        legend_hist = "Obs - Plotted variable"
    else:
        hist_to_plot = vars_to_plot
        legend_hist = "Plotted variable"

    # Loop over all variables to plot in histogram
    for v, var in enumerate(hist_to_plot):

        if var not in ds.keys():
            raise KeyError(f"Variable {var} not found in {model}.")

        if diff_include:
            var_to_plot = ds["mf_observed"] - ds[var]
        else:
            var_to_plot = ds[var]

        # Plot histogram
        if violin:
            # Drop the na values
            values = var_to_plot.values[~np.isnan(var_to_plot.values)]
            ax.violinplot([values], **kwargs)
        else:
            a, b, c = ax.hist(
                var_to_plot.values,
                bins=n_bins,
                color=ds[var].attrs["plot_color"],
                density=1,
                alpha=0.7,
                **kwargs,
            )

            if diff_include:
                ax.vlines(0, 0, np.max(a), color="dimgrey", linewidth=3.0)

        if plot_type in ["separate", "diff"]:
            index = v
        elif plot_type == "together":
            index = annotate_index

        # Compute and format mean and std of the histogram
        var_mean = np.nanmean(var_to_plot)
        var_std = np.nanstd(var_to_plot)
        str_mean = set_min_decimal_points(var_mean)
        str_std = set_min_decimal_points(var_std)

        # Write mean/std to histogram
        # If plot_type = togehter, print only mean/std of the first variable
        if not (plot_type == "together" and v != 0):
            xcoord = annotate_coords["x"]
            ycoord = annotate_coords["ytop"] - index * annotate_coords["dy"]
            ax.annotate(
                f"$\\mu$: {str_mean}\n$\\sigma$: {str_std}",
                xy=[xcoord, ycoord],
                xycoords="axes fraction",
                color=ds[var].attrs["plot_color"],
            )

    # Write number of obs
    if plot_type == "separate" and len(hist_to_plot) == 1:
        var = list(vars_to_plot)[0]
        values = ds[var].values
        mask_not_nan = ~np.isnan(values)
        n_obs = np.sum(mask_not_nan)
        if presentation_mode:
            pos_xy = [0.57, 1.05]
        else:
            pos_xy = [0.65, 1.05]

        ax.annotate(
            "$N$: " + str(n_obs), xy=pos_xy, xycoords="axes fraction", color="k"
        )

    # Set histogram x-axis label
    ax.set_xlabel(legend_hist)

    return None
