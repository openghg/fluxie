from typing import Literal

import matplotlib.pyplot as plt
import xarray as xr

from fluxy import config
from fluxy.operators.flux_align_dataset import align_map_data
from fluxy.operators.flux_combine import combine_map_dataset
from fluxy.operators.flux_map_diff import define_var_plot
from fluxy.operators.flux_map_resample import average_over_period, get_flux_mean
from fluxy.plots.utils import (
    Region,
    add_colorbar,
    add_custom_markers,
    add_site_markers,
    compute_boundary_geometry,
    define_map_figsize,
    get_map_bounds,
    get_sites_coordinates,
    plot_country_borders,
    print_cbar_label,
    set_flux_limits,
)


def plot_flux_map(
    ds_all: dict[xr.Dataset],
    species: str,
    region: Region = None,
    config_data: dict[str, any] = {},
    model_labels: dict[str, str] = {},
    cmap: str = "viridis",
    cmap_diff: str = "coolwarm",
    c_border: str = "floralwhite",
    c_border_diff: str = "dimgrey",
    add_sites: bool = False,
    add_markers: list[str] | list[list[float]] = None,
    season: str = None,
    set_fluxlim: str | tuple = "auto",
    set_fluxlim_percentile: float = None,
    plot_inversion_grid_flux: bool = False,
    zoom_degree: float = 1,
    only: Literal["posterior", "prior", "diff"] | None = None,
    fallback_sites: list[str] | None = None,
) -> plt.Figure:
    """
    Plot posterior and prior fluxes and the difference between them for all models, time averaged.

    Args:
        ds_all (dictionary of datasets):
            Dictionary of fluxes xarray datasets.
        species (str):
            Gas species, e.g. 'ch4'.
        region (str or list):
            Region to plot, e.g. 'FRANCE', 'EUROPE', [lon_min, lon_max, lat_min, lat_max].
        config_data (dict of dict):
            Dictionary of models and species information (read from json file).
        model_labels (list):
            List of model_labels from fluxy.config.
        cmap (str, optional):
            Colour map for flux plots.
        cmap_diff (str, optional):
            Colour map for flux difference plots.
        c_border (str, optional):
            Colour for flux plot country borders.
        c_border_diff (str, optional):
            Colour for flux difference plot country borders.
        add_sites (bool, optional):
            If True, scatters triangles with site locations.
        add_markers (list of str or list of lat/lon, optional):
            List of names of points to plot over larger point sources or lat/lon locations.
            See point_markers_dict for a list of options.
            e.g. ['paris', 'london', [50.,5.]]
        season (string, optional):
            If specified, plot the seasonal mean (only valable for monthly data).
            Options are 'DJF', 'MAM', 'JJA', 'SON'.
        set_fluxlim (str or list/tuple, optional):
            If provided, set the colorbar limits based on the selected options.
            Options are 'auto', a list or tuple with two elements (min, max).
        set_fluxlim_percentile (float, optional):
            If provided, set the percentile to use when setting the colorbar limits with 'auto' option.
        plot_inversion_grid_flux (bool, optional):
            If True, plots fluxes at the spatial resolution of the inversion (using the
            inversion_grid variable). If False, plots fluxes at the spatial resolution
            of the prior.
        zoom_degree (float, optional):
            Value added to the latitude and longitude bounds of the plot.
            Positive values expand the plot area, while negative values zoom in by reducing the bounds.
            Example: `zoom_degree=1` adds 1 degree to the bounds, while `zoom_degree=-1` subtracts 1 degree.
        fallback_sites (list[str] | None):
            A list of site names to use as a fallback if 'sites' is not found in the datasets.
            If None, the first available 'sites' in the datasets will be used as fallback.

    Returns:
        fig (figure):
            Three maps, for each model, of the flux prior, the flux posterior and the difference between both.
    """

    # Determine geographical boundaries
    map_bounds = get_map_bounds(
        region,
        ds_all.values(),
        config_data,
        zoom_degree=zoom_degree,
    )

    # Define variables
    var_prior = "flux_total_prior"
    var_posterior = (
        "flux_total_posterior_inversion_grid"
        if plot_inversion_grid_flux
        else "flux_total_posterior"
    )
    var_diff = (
        "posterior_prior_diff_inversion_grid"
        if plot_inversion_grid_flux
        else "posterior_prior_diff"
    )
    if only == "posterior":
        vars_list = [var_posterior]
    elif only == "prior":
        vars_list = [var_prior]
    elif only == "diff":
        vars_list = [var_diff]
    else:
        vars_list = [var_prior, var_posterior, var_diff]

    # Load country lines, species and sites information
    country_lines = compute_boundary_geometry(map_bounds)
    species_info = config_data.get("species_info", {}).get(species, {})
    sites_info = (
        get_sites_coordinates(ds_all, config_data, fallback_sites) if add_sites else ""
    )  # TODO move in the for loop once the info comes from the concentration files

    # Set flux limits #TODO Based on posterior, is this the right way to do?
    fluxlim = set_flux_limits(
        ds_all,
        var_posterior,
        map_bounds,
        species_info,
        option=set_fluxlim,
        custom_percentile=set_fluxlim_percentile,
    )

    # Initialize figure
    n_rows = len(vars_list)
    n_cols = len(ds_all)
    figsize = define_map_figsize(
        map_bounds, n_rows, n_cols, fixed_value=3 * n_rows, fixed_dimension="height"
    )
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)

    for col, (model, ds) in enumerate(ds_all.items()):
        lon, lat = ds.longitude, ds.latitude

        model_axes = ax if n_cols == 1 else (ax[:, col] if n_rows > 1 else ax[col])

        for row, var in enumerate(vars_list):
            ax_i = model_axes if n_rows == 1 else model_axes[row]

            var_plot = define_var_plot(ds, var)
            var_plot = get_flux_mean(var_plot, season)

            # Determine plot settings
            is_diff = "diff" in var
            cmap_i = cmap_diff if is_diff else cmap
            border_color = c_border_diff if is_diff else c_border
            vlim_i = (-fluxlim[1], fluxlim[1]) if is_diff else fluxlim
            marker_color = "black" if is_diff else "red"
            extend_i = "both" if is_diff else "max"

            # Plot the data
            im = ax_i.pcolormesh(
                lon,
                lat,
                var_plot,
                cmap=cmap_i,
                vmin=vlim_i[0],
                vmax=vlim_i[1],
                shading="nearest",
            )
            plot_country_borders(
                ax=ax_i, lines=country_lines, border_color=border_color
            )
            ax_i.set_xlim(map_bounds[:2])  # Longitude limits
            ax_i.set_ylim(map_bounds[2:])  # Latitude limits
            ax_i.set_aspect(1)

            # Add titles
            # Column titles
            if row == 0:
                ax_i.set_title(model_labels.get(model, model))
            # Row titles
            if col == 0:
                ax_i.set_ylabel(config.flux_labels[var])

            # Add sites and markers if specified
            if add_sites and model in sites_info:
                add_site_markers(ax_i, sites_info[model], marker_color)
            if add_markers:
                add_custom_markers(
                    ax_i, add_markers, marker_color, config_data["regions_info"]
                )

            # Add colorbar (only for the last column)
            if col == n_cols - 1:
                cbar_label = print_cbar_label(
                    ds,
                    species_info,
                    var,
                    season,
                    format=["variable", "species", "units", "time"],
                )
                add_colorbar(
                    fig,
                    ax_i,
                    im,
                    extend_i,
                    cbar_label,
                    n_cbar=n_rows,
                    idx_cbar=row,
                    colorbar_type="row",
                )
    return fig


def plot_flux_map_model_comparison(
    ds_all: dict[xr.Dataset],
    var: str,
    models: list[str],
    species: str,
    region: Region = None,
    config_data: dict = {},
    model_labels: dict[str, str] = {},
    cmap: str = "viridis",
    cmap_diff: str = "coolwarm",
    c_border: str = "floralwhite",
    c_border_diff: str = "dimgrey",
    add_sites: bool = False,
    add_markers: list[str] | list[list[float]] = None,
    season: str = None,
    set_fluxlim: str | tuple = "auto",
    set_fluxlim_percentile: float = None,
    zoom_degree: float = 1,
    fallback_sites: list[str] | None = None,
) -> plt.Figure:
    """
    Plot a given flux variable for two models and the difference between them.

    Args:
        ds_all (dictionary of datasets):
            Dictionary of fluxes xarray datasets.
        var (str):
            The name of the flux variable to be plotted and compared across models.
            Example: 'flux_total_posterior'.
        models (list[str]):
            The name of the 2 models to be compared. This should correspond to 2 keys in `ds_all`.
            Example: ['intem_name_edgar', 'elris_name_edgar']
        species (str):
            Gas species, e.g. 'ch4'.
        region (str or list):
            Lat/lon region to plot, options for 'UK', 'FRANCE', 'GERMANY',
            'NWEU','CWEU','EUROPE'.
            A list with [lon_min, lon_max, lat_min, lat_max] can also be provided.
        config_data (dict of dict):
            Dictionary of models and species information (read from json file).
        model_labels (list):
            List of model_labels from fluxy.config.
        cmap (str, optional):
            Colour map for flux plots.
        cmap_diff (str, optional):
            Colour map for flux difference plots.
        c_border (str, optional):
            Colour for flux plot country borders.
        c_border_diff (str, optional):
            Colour for flux difference plot country borders.
        add_sites (bool, optional):
            If True, scatters triangles with site locations.
        add_markers (list of str or list of lat/lon, optional):
            List of names of points to plot over larger point sources or lat/lon locations.
            See point_markers_dict for a list of options.
            e.g. ['paris', 'london', [50.,5.]]
        season (string, optional):
            If specified, plot the seasonal mean (only valable for monthly data).
            Options are 'DJF', 'MAM', 'JJA', 'SON'.
        set_fluxlim (str or list/tuple, optional):
            If provided, set the colorbar limits based on the selected options.
            Options are 'auto', a list or tuple with two elements (min, max).
        set_fluxlim_percentile (float, optional):
            If provided, set the percentile to use when setting the colorbar limits with 'auto' option.
        zoom_degree (float, optional):
            Value added to the latitude and longitude bounds of the plot.
            Positive values expand the plot area, while negative values zoom in by reducing the bounds.
            Example: `zoom_degree=1` adds 1 degree to the bounds, while `zoom_degree=-1` subtracts 1 degree.
        fallback_sites (list[str] | None):
            A list of site names to use as a fallback if 'sites' is not found in the datasets.
            If None, the first available 'sites' in the datasets will be used as fallback.
    Returns:
        fig (figure):
            Three maps of a target flux variable of the first and second models and the diffence between both.
    """

    # Models check
    model_names = list(ds_all.keys())
    missing_models = [m for m in models if m not in model_names]
    if missing_models:
        raise ValueError(
            f"Model(s) {', '.join(missing_models)} not found in the dataset."
        )

    if not isinstance(models, list) or len(models) != 2:
        raise ValueError("Models must be a list of exactly two strings.")

    # Determine geographical boundaries
    map_bounds = get_map_bounds(
        region,
        ds_all.values(),
        config_data,
        zoom_degree=zoom_degree,
    )

    # Prepare datasets
    ds_dict = {k: v for k, v in ds_all.items() if k in models}
    ds_dict = align_map_data(ds_dict)
    ds_dict["diff"] = ds_dict[models[1]] - ds_dict[models[0]]
    ds_dict["diff"].attrs["frequency"] = ds_dict[models[0]].attrs[
        "frequency"
    ]  # Copy attributes from models[0]
    for v in ds_dict["diff"].data_vars:
        ds_dict["diff"][v].attrs = ds_dict[models[0]][v].attrs

    # Load country lines, species and sites information
    country_lines = compute_boundary_geometry(map_bounds)
    species_info = config_data["species_info"][species]
    sites_info = (
        get_sites_coordinates(ds_dict, config_data, fallback_sites) if add_sites else ""
    )  # TODO move in the for loop once the info comes from the concentration files

    # Set flux limits
    fluxlim = set_flux_limits(
        ds_dict,
        var,
        map_bounds,
        species_info,
        option=set_fluxlim,
        custom_percentile=set_fluxlim_percentile,
    )

    # Initialize figure
    n_rows = 1
    n_cols = 3
    fig, ax = plt.subplots(
        n_rows, n_cols, constrained_layout=True, figsize=(n_cols * 5, 9)
    )
    for col, (model, ds) in enumerate(ds_dict.items()):
        ax_i = ax[col]
        lon, lat = ds.longitude, ds.latitude

        var_plot = define_var_plot(ds, var)
        var_plot = get_flux_mean(var_plot, season)

        # Determine plot settings
        is_diff = ("diff" in var) or ("diff" in model)
        cmap_i = cmap_diff if is_diff else cmap
        border_color = c_border_diff if is_diff else c_border
        vlim_i = (-fluxlim[1], fluxlim[1]) if is_diff else fluxlim
        marker_color = "black" if is_diff else "red"
        extend_i = "both" if is_diff else "max"

        # Plot the data
        im = ax_i.pcolormesh(
            lon,
            lat,
            var_plot,
            cmap=cmap_i,
            vmin=vlim_i[0],
            vmax=vlim_i[1],
            shading="nearest",
        )
        plot_country_borders(ax=ax_i, lines=country_lines, border_color=border_color)
        ax_i.set_xlim(map_bounds[:2])  # Longitude limits
        ax_i.set_ylim(map_bounds[2:])  # Latitude limits
        ax_i.set_aspect(1)

        # Add titles
        if model == "diff":
            ax_i.set_title(f"{model_labels[models[1]]} - {model_labels[models[0]]}")
        else:
            ax_i.set_title(model_labels[model])

        # Add sites and markers if specified
        if add_sites:
            add_site_markers(ax_i, sites_info.get(model, {}), marker_color)
        if add_markers:
            add_custom_markers(
                ax_i, add_markers, marker_color, config_data["regions_info"]
            )

        # Add colorbar
        cbar_label = print_cbar_label(
            ds,
            species_info,
            var,
            season,
            format=["variable", "species", "units", "time"],
        )
        if model == "diff":
            cbar_lines = cbar_label.split("\n")
            cbar_lines[0] += " difference"
            cbar_label = "\n".join(cbar_lines)
        add_colorbar(
            fig,
            ax_i,
            im,
            extend_i,
            cbar_label,
            n_cbar=3,
            idx_cbar=col,
            colorbar_type="column",
        )
    return fig


def plot_flux_map_over_time(
    ds_all: dict[xr.Dataset],
    var: str,
    species: str,
    region: Region = None,
    config_data: dict = {},
    model_labels: dict[str] = {},
    chop_by: str | list[str] | list[float] | list[list[float]] = "year",
    dt: float = 1,
    plot_combined: bool = False,
    cmap: str = "viridis",
    cmap_diff: str = "coolwarm",
    c_border: str = "floralwhite",
    c_border_diff: str = "dimgrey",
    add_sites: bool = False,
    add_markers: list[str] | list[list[float]] = None,
    set_fluxlim: str | tuple = "auto",
    set_fluxlim_percentile: float = None,
    zoom_degree: float = 1,
    fallback_sites: list[str] | None = None,
) -> plt.Figure:
    """
    Plot a given flux variable averaged over specific time intervals, for all models or the model mean.

    Args:
        ds_all (dictionary of datasets):
            Dictionary of fluxes xarray datasets.
        var (str):
            The name of the flux variable to be plotted and compared across models.
            Example: 'flux_total_posterior'.
        species (str):
            Gas species, e.g. 'ch4'.
        region (str or list):
            Region to plot, e.g. 'FRANCE', 'EUROPE', [lon_min, lon_max, lat_min, lat_max].
        config_data (dict of dict):
            Dictionary of models and species information (read from json file).
        model_labels (list):
            List of model_labels from fluxy.config.
        chop_by (str or list):
            Time units to perform the average, options for 'year', 'month' and 'season'.
            Alternatively, a list of starting dates or months number can be provided.
        dt (int):
            If chop_by = 'year' or 'month': dt is the number of time steps (in chop_by units) to use in the averaging.
        plot_combined (bool):
            If True, plots the mean over all models at each time step.
        cmap (str, optional):
            Colour map for flux plots.
        cmap_diff (str, optional):
            Colour map for flux difference plots.
        c_border (str, optional):
            Colour for flux plot country borders.
        c_border_diff (str, optional):
            Colour for flux difference plot country borders.
        add_sites (bool, optional):
            If True, scatters triangles with site locations.
        add_markers (list of str or list of lat/lon, optional):
            List of names of points to plot over larger point sources or lat/lon locations.
            See point_markers_dict for a list of options.
            e.g. ['paris', 'london', [50.,5.]]
        set_fluxlim (str or list/tuple, optional):
            If provided, set the colorbar limits based on the selected options.
            Options are 'auto', a list or tuple with two elements (min, max).
        set_fluxlim_percentile (float, optional):
            If provided, set the percentile to use when setting the colorbar limits with 'auto' option.
        zoom_degree (float, optional):
            Value added to the latitude and longitude bounds of the plot.
            Positive values expand the plot area, while negative values zoom in by reducing the bounds.
            Example: `zoom_degree=1` adds 1 degree to the bounds, while `zoom_degree=-1` subtracts 1 degree.
        fallback_sites (list[str] | None):
            A list of site names to use as a fallback if 'sites' is not found in the datasets.
            If None, the first available 'sites' in the datasets will be used as fallback.
    Returns:
        fig (figure):
            A plot of spatial flux of the variable specified in var
            averaged over the number of time steps specified in dt.
    """
    # Determine geographical boundaries
    map_bounds = get_map_bounds(
        region,
        ds_all.values(),
        config_data,
        zoom_degree=zoom_degree,
    )
    # Prepare datasets and average over given periods
    if plot_combined:
        ds_dict = align_map_data(ds_all)
        ds_dict = combine_map_dataset(ds_dict)
    else:
        ds_dict = ds_all

    ds_chopby = {}
    for key, ds in ds_dict.items():
        ds_chopby[key], time_labels = average_over_period(ds, dt, chop_by)

    # Load country lines, species and sites information
    country_lines = compute_boundary_geometry(map_bounds)
    species_info = config_data.get("species_info", {}).get(species, {})
    sites_info = (
        get_sites_coordinates(ds_all, config_data, fallback_sites) if add_sites else ""
    )  # TODO move in the for loop once the info comes from the concentration files

    # Set flux limits
    lim = set_flux_limits(
        ds_chopby,
        var,
        map_bounds,
        species_info,
        option=set_fluxlim,
        custom_percentile=set_fluxlim_percentile,
    )

    # Determine plot settings
    is_diff = "diff" in var
    cmap = cmap_diff if is_diff else cmap
    border_color = c_border_diff if is_diff else c_border
    marker_color = "black" if is_diff else "red"
    extend = "both" if is_diff else "max"

    # Initialise figure
    n_rows = len(ds_chopby.keys())
    n_cols = len(time_labels)

    if n_rows * n_cols == 4:
        # Re-organize the data for a nicer display
        fig, ax = plt.subplots(2, 2, figsize=(2 * 4.2, 2 * 3))
        ax = ax.flatten()
    else:
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))

    for row, (model, ds) in enumerate(ds_chopby.items()):

        lon, lat = ds.longitude, ds.latitude
        var_plot = define_var_plot(ds, var)

        for col, time_label in enumerate(time_labels):
            if n_rows == 1 and n_cols == 1:
                ax_i = ax
            elif n_rows == 1:
                ax_i = ax[col]
            elif n_cols == 1:
                ax_i = ax[row]
            else:
                ax_i = ax[row, col]

            var_i = var_plot.isel(time=col)

            # Plot the data
            im = ax_i.pcolormesh(
                lon, lat, var_i, cmap=cmap, vmin=lim[0], vmax=lim[1], shading="nearest"
            )
            plot_country_borders(
                ax=ax_i, lines=country_lines, border_color=border_color
            )
            ax_i.set_xlim(map_bounds[:2])  # Longitude limits
            ax_i.set_ylim(map_bounds[2:])  # Latitude limits
            ax_i.set_aspect(1)

            # Add titles
            if row == 0:
                # Column titles
                ax_i.set_title(time_label)
            if not plot_combined and col == 0:
                # Row titles
                ax_i.set_ylabel(model_labels.get(model, model))

            # Add sites and markers if specified
            if add_sites:
                if model in sites_info:
                    add_site_markers(
                        ax_i, sites_info[model], marker_color
                    )  # TODO Manage combined
                elif model == "combined":
                    merged_sites_info = {
                        k: v for d in sites_info.values() for k, v in d.items()
                    }
                    add_site_markers(ax_i, merged_sites_info, marker_color)

            if add_markers:
                add_custom_markers(
                    ax_i, add_markers, marker_color, config_data["regions_info"]
                )

    # Add colorbar
    cbar_label = print_cbar_label(
        ds, species_info, var, format=["variable", "species", "units"]
    )
    add_colorbar(
        fig,
        ax,
        im,
        extend=extend,
        label=cbar_label,
        n_cbar=1,
        idx_cbar=1,
        colorbar_type="figure",
    )

    return fig
