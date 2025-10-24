import logging
from typing import Literal

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.dates import MonthLocator, YearLocator
from matplotlib.ticker import NullFormatter

from fluxy import config
from fluxy.operators.select import get_site_index, get_unique_sites
from fluxy.plots.utils import set_min_decimal_points
from fluxy.types import VariableType
from fluxy.operators.select import (
    FrequencyType,
    clean_timeseries_missing_data,
    get_site_index,
    get_unique_sites,
    slice_site,
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
    if model_colors is None:
        model_colors = config.set_model_colors(models)

    species_info = config_data.get("species_info", {}).get(species, {})

    # Check the include dictionary
    if not include:
        raise ValueError(
            "The include dictionary is empty. Please provide variables to include in the plot."
        )
    if isinstance(include, str):
        include = {include: None}
    if isinstance(include, (list, tuple)):
        include = {var: None for var in include}

    vars_to_plot = include.keys()
    plot_units = []

    min_mf = np.inf
    max_mf = -np.inf

    # Define number of rows in figure
    if plot_type == "separate":
        nrows = len(models)
    elif plot_type in ["together", "diff"]:
        nrows = 1
    else:
        raise ValueError(
            f"Option {plot_type} not implemented. Set plot_type to 'separate', 'together' or 'diff'."
        )

    # Create figure
    ncols = 2 if histogram_type and histogram_type != "none" else 1
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(15, nrows * 3),
        gridspec_kw={"width_ratios": [0.8, 0.2]} if ncols == 2 else {},
        constrained_layout=True,
        sharey="row" if histogram_type == "violin" else False,
        sharex="col",
        squeeze=False,
    )

    logger.info(
        f"Plotting {len(models)} models with {len(vars_to_plot)} variables in {plot_type} mode."
    )

    # Loop over all models
    for i, m in enumerate(models):

        # Define plot_type specific settings
        iax = i if plot_type == "separate" else 0

        if plot_type == "diff":
            mdiff0, mdiff1 = m.split("--")
            model_label = f"{model_labels[mdiff0]} - {model_labels[mdiff1]}"
            model_color = model_colors[mdiff0]
        else:
            model_label = model_labels.get(m, m)
            model_color = model_colors[m]

        ds_plot = ds_all[m]
        # Check there is only one site in the dataset
        if len(np.unique(ds_plot["number_of_identifier"])) > 1:
            raise ValueError(
                f"Dataset {m} contains more than one site. "
                "Use slice_site to select a single site."
            )

        # Clean the time dimension
        ds_plot = clean_timeseries_missing_data(
            ds_plot, variables_nans=vars_to_plot, min_freq=time_freq_min
        )

        # Loop over all variables to plot
        for var in vars_to_plot:

            if var not in ds_plot.keys():
                raise KeyError(f"Variable {var} not found in {m}.")

            # Get var unit
            plot_units.append(ds_plot[var].attrs["units"])

            # Define plotting color
            plot_color = model_color[config.mf_color_index.get(var, 0)]
            if var == "mf_observed" and len(vars_to_plot) > 1:
                plot_color = "black"

            x, y = ds_plot["time"].values, ds_plot[var].values
            kwargs = {
                "label": f"{model_label} {config.mf_labels.get(var, var)}",
                "color": plot_color,
                "alpha": 0.8,
            }

            if var == "mf_observed" or plot_type == "diff":
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

            unc_var = include[var]

            if unc_var:
                if plot_type == "diff":
                    raise ValueError(
                        f"Option plot_type='diff' does not accept uncertainties. Replace '{unc_var}' by None."
                    )

                # Accept both percentile and stdev as uncertainty variables
                if unc_var not in ds_plot.keys():
                    unc_var_in = unc_var
                    if "percentile" in unc_var:
                        unc_var = unc_var.replace("percentile", "stdev")

                    elif "stdev" in unc_var:
                        unc_var = unc_var.replace("stdev", "percentile")

                    if unc_var not in ds_plot.keys():
                        raise KeyError(
                            f"Variables {unc_var_in} and {unc_var} not found in {m}."
                        )
                    logger.warning(
                        f"Variable {unc_var_in} not found in {m} so reading uncert from {unc_var}."
                    )

                kwargs = {
                    "color": plot_color,
                }

                # Define uncertainty band
                flag_fill_between = False
                if unc_var.split("_")[0] == "percentile":
                    y1 = ds_plot[unc_var][0, :].values
                    y2 = ds_plot[unc_var][1, :].values
                    flag_fill_between = True
                elif unc_var.split("_")[-1] in ["prior", "posterior"]:
                    y1 = ds_plot[var].values - ds_plot[unc_var].values
                    y2 = ds_plot[var].values + ds_plot[unc_var].values
                    flag_fill_between = True

                if flag_fill_between:
                    # Add uncertainty band
                    ax[iax, 0].fill_between(
                        x,
                        y1=y1,
                        y2=y2,
                        alpha=0.2,
                        **kwargs,
                    )

                else:
                    # Add error bar
                    ax[iax, 0].errorbar(
                        x,
                        y=ds_plot[var].values,
                        yerr=ds_plot[unc_var].values,
                        alpha=0.4,
                        fmt="none",
                        **kwargs,
                    )

        # Plot histogram
        if ncols == 2:
            plot_histogram(
                ax[iax, 1],
                ds_plot,
                m,
                vars_to_plot,
                diff_include,
                model_color,
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
            plot_title = model_label
        elif plot_type == "together":
            plot_title = "All models"
        ax[iax, 0].set_title(plot_title)

        # Set print units
        plot_units = list(set(plot_units))
        if len(plot_units) != 1:
            raise ValueError(
                f"{vars_to_plot} in {models} do not have the same units. So far, the following were found: {plot_units}."
            )

        # Set timeseries y-axis label and legend

        height_label = ""
        if intake_height is not None:
            height_label = f"-{intake_height}m"

        ax[iax, 0].set_ylabel(
            " ".join(
                [
                    species_info.get("species_print", ""),
                    (site if site else "") + height_label,
                    f"({plot_units[0]})",
                ]
            )
        )

        leg = ax[iax, 0].legend(ncol=2, borderpad=0.2, columnspacing=1.0)
        try:
            for l in leg.legend_handles:
                l.set_linewidth(5.0)
        except:
            for l in leg.legendHandles:
                l.set_linewidth(5.0)

        if len(ds_plot["time"]) <= 1:
            continue
        start_date = ds_plot["time"].values.min()
        end_date = ds_plot["time"].values.max()

        # Set timeseries x-axis ticks
        if (
            int(end_date.astype("datetime64[M]") - start_date.astype("datetime64[M]"))
            > 12
        ):
            ax[iax, 0].xaxis.set_minor_locator(MonthLocator())
            ax[iax, 0].xaxis.set_minor_formatter(NullFormatter())
            ax[iax, 0].xaxis.set_major_locator(YearLocator())
        else:
            ax[iax, 0].xaxis.set_major_locator(MonthLocator())
            if presentation_mode:
                ax[iax, 0].tick_params(axis="x", rotation=70)
        ax[iax, 0].grid(color = 'lightgrey', linestyle = '-', linewidth = 0.7)
        ax[iax, 0].set_axisbelow(True)

    if y_lim is None:
        y_lim = [min_mf - 0.02 * min_mf, max_mf + 0.05 * max_mf]

    # Set all the axes to the same y-axis limits
    for iax, ax0 in enumerate(ax[:, 0]):
        ax0.set_ylim(y_lim)

        if ncols == 2 and (diff_include is None or len(diff_include) == 0):
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
    site_list = get_unique_site_height_pairs(ds_all,separate_by_height)

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
    xticklabels = [f"{s}\n{int(h)}m" if separate_by_height else s for (s, h) in site_list]
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

    species_info = config_data.get("species_info",{}).get(species,{})
    fig.suptitle(
        (
            f'Timestamps with {species_info.get("species_print","")} assimilated observations between'
            f"\n{start_date} and {end_date}"
        )
    )

    return fig


def plot_histogram(
    ax: matplotlib.axes.Axes,
    ds: xr.Dataset,
    model: str,
    vars_to_plot: list[str],
    diff_include: list[str] | None,
    model_color: list[str],
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
        model_color (list of str):
            List of colors for plotting a specific model.
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
                color=model_color[config.mf_color_index.get(var, 0)],
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
                color=model_color[config.mf_color_index.get(var, 0)],
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
