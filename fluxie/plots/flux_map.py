from typing import Literal

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import logging

from fluxie.operators.flux_align_dataset import align_map_data
from fluxie.operators.flux_combine import combine_map_dataset
from fluxie.operators.flux_map_diff import define_var_plot, define_var_plot_by_sector, make_model_diff_ds
from fluxie.operators.flux_map_resample import resample_over_period
from fluxie.plots.utils import (
    Region,
    add_colorbar,
    add_custom_markers,
    add_site_markers,
    compute_boundary_geometry,
    define_flux_label,
    define_map_figsize,
    get_map_bounds,
    get_active_sites_coordinates,
    plot_country_borders,
    print_cbar_label,
    set_flux_limits,
)

logger = logging.getLogger(__name__)


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
    resample_uncert_correlation: bool = False,
    sector: str = "total",
    include_title_and_labels: bool = True,
    site_marker: str = "o",
    city_marker: str = "^",
    marker_color: str | None = None,
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
            List of model_labels from fluxie.config.
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
        only (str, optional):
            Option to plot only "posterior" or "prior" or "diff"
        fallback_sites (list[str] | None):
            A list of site names to use as a fallback if 'sites' is not found in the datasets.
            If None, the first available 'sites' in the datasets will be used as fallback.
        resample_uncert_correlation (bool, optional):
            If True, uncertainties are averaged directly .
            If False, uncertainties are calculated as RMSE-like aggregation.
        sector (str):
            Emissions sector to plot. Default 'total'.
        include_title_and_labels (bool):
            If False, removes titles, axis labels and extra info from the colour bar.
        site_marker (str):
            Marker for site locations.
        city_marker (str):
            Marker for city locations.
        marker_color (str):
            Marker color.
    Returns:
        fig (figure):
            Three maps, for each model, of the flux prior, the flux posterior and the difference between both.
    """

    # Check for inversion_grid and sector option
    if plot_inversion_grid_flux == True and sector != "total":
        raise ValueError(
            f"Currently, you cannot plot sectors other than 'total' using the inversion_grid variable. "
            + "Set plot_inversion_grid_flux to False to plot other sectors."
        )

    # Determine geographical boundaries
    map_bounds = get_map_bounds(
        region,
        ds_all.values(),
        config_data,
        zoom_degree=zoom_degree,
    )

    # Define variables
    var_prior = f"flux_{sector}_prior"
    var_posterior = f"flux_{sector}_posterior"
    var_diff = "posterior_prior_diff"

    if plot_inversion_grid_flux:
        var_prior += "_inversion_grid"
        var_posterior += "_inversion_grid"
        var_diff += "_inversion_grid"

    if only == "posterior":
        vars_list = [var_posterior]
        var_fluxlim = var_posterior
    elif only == "prior":
        vars_list = [var_prior]
        var_fluxlim = var_prior
    elif only == "diff":
        vars_list = [var_diff]
        var_fluxlim = var_diff
    else:
        vars_list = [var_prior, var_posterior, var_diff]
        var_fluxlim = var_posterior  # TODO Flux limits based on posterior, is this the right way to do?

    # Prepare datasets and resample over the whole time period (season=None) or a given season
    ds_dict = {
        m: resample_over_period(
            define_var_plot(ds, vars_list, sector),
            chop_by=season,
            resample_uncert_correlation=resample_uncert_correlation,
        )[0]
        for m, ds in ds_all.items()
    }

    # Load country lines and species information
    country_lines = compute_boundary_geometry(map_bounds)
    species_info = config_data.get("species_info", {}).get(species, {})

    # Set flux limits
    fluxlim = set_flux_limits(
        ds_dict,
        var_fluxlim,
        map_bounds,
        option=set_fluxlim,
        custom_percentile=set_fluxlim_percentile,
    )

    # Initialize figure
    n_rows = len(vars_list)
    n_cols = len(ds_dict)
    figsize = define_map_figsize(
        map_bounds, n_rows, n_cols, fixed_value=3 * n_rows, fixed_dimension="height"
    )
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, layout="compressed")

    for col, (model, ds) in enumerate(ds_dict.items()):
        lon, lat = ds.longitude, ds.latitude
        sites_info = (
            get_active_sites_coordinates(ds, config_data, fallback_sites)
            if add_sites
            else ""
        )

        model_axes = ax if n_cols == 1 else (ax[:, col] if n_rows > 1 else ax[col])

        for row, var in enumerate(vars_list):
            ax_i = model_axes if n_rows == 1 else model_axes[row]

            # Determine plot settings
            is_diff = "diff" in var
            cmap_i = cmap_diff if is_diff else cmap
            border_color = c_border_diff if is_diff else c_border
            vlim_i = (-fluxlim[1], fluxlim[1]) if is_diff else fluxlim
            if marker_color is None:
                marker_color = "black" if is_diff else "red"
            extend_i = "both" if is_diff else "max"

            # Plot the data
            im = ax_i.pcolormesh(
                lon,
                lat,
                ds[var],
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

            # Adjust ticks layout
            if row < n_rows - 1:
                ax_i.set_xticklabels([])
            if col > 0:
                ax_i.set_yticklabels([])

            if not include_title_and_labels:
                ax_i.set_xticks([])
                ax_i.set_yticks([])

            # Add titles
            # Column titles
            if row == 0 and include_title_and_labels:
                ax_i.set_title(model_labels.get(model, model))
            # Row titles
            if col == 0 and include_title_and_labels:
                ax_i.set_ylabel(define_flux_label(var))

            # Add sites and markers if specified
            if add_sites and sites_info:
                add_site_markers(ax_i, sites_info, marker_color, site_marker)
            if add_markers:
                add_custom_markers(
                    ax_i,
                    add_markers,
                    marker_color,
                    config_data["regions_info"],
                    city_marker,
                )

            if include_title_and_labels:
                cbar_label_format = ["variable", "species", "units", "time"]
            else:
                cbar_label_format = ["species", "units", "time"]

            # Add colorbar (only for the last column)
            if col == n_cols - 1:
                cbar_label = print_cbar_label(
                    ds,
                    species_info,
                    var,
                    format=cbar_label_format,
                )  # TODO Here, based on the last iteration. Check if consistent for all models?
                add_colorbar(
                    fig,
                    ax_i,
                    im,
                    extend_i,
                    label=cbar_label,
                    n_cbar=n_rows,
                    idx_cbar=row,
                    colorbar_type="row",
                )
    return fig


def plot_flux_map_by_sector(
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
    resample_uncert_correlation: bool = False,
    sectors: dict[list[str]] | list[str] = [
            "agriculture",
            "waste",
            "energy",
            "industry",
        ],
    columns: str = "model",
    mean_over_models: bool = False,
    include_title_and_labels: bool = True,
    site_marker: str = "o",
    city_marker: str = "^",
    marker_color: str | None = None,
) -> plt.Figure:
    """
    Plot prior, posterior and prior-posterior difference for multiple emission sectors.

    Returns a list of figures. The layout depends on the `columns` argument:
    - columns='model': one figure per sector, columns are models, colour scale
      is computed independently per sector.
    - columns='sector': one figure per model, columns are sectors, colour scale
      is shared across all sectors within a figure.

    Args:
        ds_all (dict of xarray.Dataset):
            Dictionary of flux datasets, keyed by model name.
        species (str):
            Gas species, e.g. 'ch4'.
        region (str or list):
            Region to plot, e.g. 'FRANCE', 'EUROPE', [lon_min, lon_max, lat_min, lat_max].
        config_data (dict of dict):
            Dictionary of models and species information (read from json file).
        model_labels (dict):
            Display labels for model names, from fluxie.config.
        cmap (str, optional):
            Colour map for flux plots.
        cmap_diff (str, optional):
            Colour map for flux difference plots.
        c_border (str, optional):
            Colour for flux plot country borders.
        c_border_diff (str, optional):
            Colour for flux difference plot country borders.
        add_sites (bool, optional):
            If True, scatters site location markers on each panel.
        add_markers (list of str or list of lat/lon, optional):
            Named point sources or [lat, lon] locations to mark.
        season (str, optional):
            If specified, plot the seasonal mean ('DJF', 'MAM', 'JJA', 'SON').
        set_fluxlim (str or tuple, optional):
            Colorbar limit strategy: 'auto' or a (min, max) tuple.
        set_fluxlim_percentile (float, optional):
            Percentile used for the 'auto' limit strategy.
        plot_inversion_grid_flux (bool, optional):
            If True, plot fluxes at inversion grid resolution.
        zoom_degree (float, optional):
            Degrees added to the map bounds (negative values zoom in).
        only (str, optional):
            Restrict to 'posterior', 'prior', or 'diff' only.
        fallback_sites (list[str] | None):
            Fallback site names if 'sites' is not in the dataset.
        resample_uncert_correlation (bool, optional):
            If True, uncertainties are averaged directly; otherwise aggregated as RMSE.
        sectors (list[str] or dict, optional):
            Emission sectors to plot, e.g. ['agriculture', 'waste', 'energy'].
            A dict with a 'model' key can be used to specify different sectors
            per data source.
        columns (str, optional):
            Layout of columns in each figure. Options:
            - 'model' (default): columns are models, one figure per sector.
              Colour scale is computed independently per sector.
            - 'sector': columns are sectors, one figure per model.
              Colour scale is shared across sectors within a figure.
        mean_over_models (bool, optional):
            Only used when columns='sector'. If True, averages across all models
            and returns a single figure instead of one per model. Default False.
        include_title_and_labels (bool, optional):
            If False, removes titles, axis labels and colorbar annotations.
        site_marker (str, optional):
            Marker style for site locations.
        city_marker (str, optional):
            Marker style for city locations.
        marker_color (str, optional):
            Marker colour.

    Returns:
        figs (list of Figure):
            List of matplotlib figures, one per sector (columns='model') or
            one per model (columns='sector').
    """

    # Determine geographical boundaries
    map_bounds = get_map_bounds(
        region,
        ds_all.values(),
        config_data,
        zoom_degree=zoom_degree,
    )

    # Define variables
    var_prior = [f"flux_{sector}_prior" for sector in sectors]
    var_posterior = [f"flux_{sector}_posterior" for sector in sectors]
    var_diff = "posterior_prior_diff"

    if plot_inversion_grid_flux:
        var_prior = [v + "_inversion_grid" for v in var_prior]
        var_posterior = [v + "_inversion_grid" for v in var_posterior]
        var_diff += "_inversion_grid"

    if only == "posterior":
        vars_list = var_posterior
        var_fluxlim = var_posterior
    elif only == "prior":
        vars_list = var_prior
        var_fluxlim = var_prior
    elif only == "diff":
        vars_list = [var_diff]
        var_fluxlim = var_diff
    else:
        vars_list = [var_prior, var_posterior, var_diff]
        var_fluxlim = var_posterior  # TODO Flux limits based on posterior, is this the right way to do?

    # Prepare datasets and resample over the whole time period (season=None) or a given season
    ds_dict = {
        m: resample_over_period(
            define_var_plot_by_sector(ds, vars_list, sectors),
            chop_by=season,
            resample_uncert_correlation=resample_uncert_correlation,
        )[0]
        for m, ds in ds_all.items()
    }

    # Load country lines and species information
    country_lines = compute_boundary_geometry(map_bounds)
    species_info = config_data.get("species_info", {}).get(species, {})

    vars_list = [v for item in vars_list for v in (item if isinstance(item, list) else [item])]

    multi_sector = len(sectors) > 1

    if columns == "sector":
        n_cols = len(sectors)
        if mean_over_models:
            ds_mean = xr.concat(list(ds_dict.values()), dim="model").mean(dim="model")
            ds_mean.attrs = next(iter(ds_dict.values())).attrs
            n_figs = 1
        else:
            n_figs = len(ds_dict)
    elif columns == "model":
        n_cols = len(ds_dict)
        n_figs = len(sectors)

    figs = []
    for ifig in range(n_figs):

        if columns == "model":
            sec_fig = sectors[ifig]
            vars_for_fig = [
                v for v in vars_list
                if sec_fig in v or not any(s in v for s in sectors)
            ]
            ref_sector = None
            var_fluxlim_fig = f"flux_{sec_fig}_posterior"

        elif columns == "sector":
            sec_fig = None
            # Use first sector as row template; actual_var swaps it per column
            ref_sector = sectors[0]
            vars_for_fig = [
                v for v in vars_list
                if ref_sector in v or not any(s in v for s in sectors)
            ]
            if mean_over_models:
                model = "Model mean"
                ds = ds_mean
            else:
                model, ds = list(ds_dict.items())[ifig]
            lon, lat = ds.longitude, ds.latitude
            sites_info = get_active_sites_coordinates(ds, config_data, fallback_sites) if add_sites else ""
            var_fluxlim_fig = f"flux_{ref_sector}_posterior"

        n_rows = len(vars_for_fig)
        figsize = define_map_figsize(
            map_bounds, n_rows, n_cols, fixed_value=3 * n_rows, fixed_dimension="height"
        )
        fluxlim = set_flux_limits(
            ds_dict,
            var_fluxlim_fig,
            map_bounds,
            option=set_fluxlim,
            custom_percentile=set_fluxlim_percentile,
        )
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, layout="compressed")

        if columns == "sector" and include_title_and_labels:
            fig.suptitle(model_labels.get(model, model))

        col_specs = []
        if columns == "model":
            for model, ds in ds_dict.items():
                col_specs.append({
                    "label": model,
                    "ds": ds,
                    "lon": ds.longitude,
                    "lat": ds.latitude,
                    "sector": sec_fig,  # ← Sektor dieser Figure
                    "sites_info": get_active_sites_coordinates(ds, config_data, fallback_sites) if add_sites else "",
                })
        elif columns == "sector":
            for sector in sectors:
                col_specs.append({
                    "label": sector,
                    "ds": ds,
                    "lon": ds.longitude,
                    "lat": ds.latitude,
                    "sector": sector if multi_sector else None,
                     "ref_sector": ref_sector,
                    "sites_info": get_active_sites_coordinates(ds, config_data, fallback_sites) if add_sites else "",
                })

        for col, spec in enumerate(col_specs):
            ds_col = spec["ds"]
            lon, lat = spec["lon"], spec["lat"]
            sites_info = spec["sites_info"]

            model_axes = ax if n_cols == 1 else (ax[:, col] if n_rows > 1 else ax[col])

            for row, var in enumerate(vars_for_fig):  # ← vars_for_fig statt vars_list
                ax_i = model_axes if n_rows == 1 else model_axes[row]

                
                sec = spec["sector"]
                ref_sec = spec.get("ref_sector")

                if ref_sec and ref_sec in var:
                    actual_var = var.replace(ref_sec, sec)
                elif sec and sec not in var:
                    actual_var = f"{sec}_{var}"
                else:
                    actual_var = var

                is_diff = "diff" in var
                cmap_i = cmap_diff if is_diff else cmap
                border_color = c_border_diff if is_diff else c_border
                vlim_i = (-fluxlim[1], fluxlim[1]) if is_diff else fluxlim
                if marker_color is None:
                    marker_color = "black" if is_diff else "red"
                extend_i = "both" if is_diff else "max"

                im = ax_i.pcolormesh(
                    lon, lat, ds_col[actual_var],
                    cmap=cmap_i, vmin=vlim_i[0], vmax=vlim_i[1], shading="nearest",
                )
                
                
                plot_country_borders(
                    ax=ax_i, lines=country_lines, border_color=border_color
                )
                ax_i.set_xlim(map_bounds[:2])  # Longitude limits
                ax_i.set_ylim(map_bounds[2:])  # Latitude limits
                ax_i.set_aspect(1)

                # Adjust ticks layout
                if row < n_rows - 1:
                    ax_i.set_xticklabels([])
                if col > 0:
                    ax_i.set_yticklabels([])

                if not include_title_and_labels:
                    ax_i.set_xticks([])
                    ax_i.set_yticks([])

                # Add titles
                if row == 0 and include_title_and_labels:
                    if columns == "sector":
                        ax_i.set_title(spec["label"].capitalize())  # Sektorname als Spaltentitel
                    else:
                        ax_i.set_title(model_labels.get(model, model))  # Modellname als Spaltentitel

                # Add sites and markers if specified
                if add_sites and sites_info:
                    add_site_markers(ax_i, sites_info, marker_color, site_marker)
                if add_markers:
                    add_custom_markers(
                        ax_i,
                        add_markers,
                        marker_color,
                        config_data["regions_info"],
                        city_marker,
                    )

                if include_title_and_labels:
                    cbar_label_format = ["variable", "species", "units", "time"]
                else:
                    cbar_label_format = ["species", "units", "time"]

                # Add colorbar (only for the last column)
                if col == n_cols - 1:
                    cbar_label = print_cbar_label(
                        ds_col,       # ← war: ds
                        species_info,
                        actual_var,   # ← war: var
                        format=cbar_label_format,
                    )
                    add_colorbar(
                        fig,
                        ax_i,
                        im,
                        extend_i,
                        label=cbar_label,
                        n_cbar=n_rows,
                        idx_cbar=row,
                        colorbar_type="row",
                    )
                    
        figs.append(fig)
    return figs



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
    resample_uncert_correlation: bool = False,
    sector: str = "total",
    site_marker: str = "o",
    city_marker: str = "^",
    marker_color: str | None = None,
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
            List of model_labels from fluxie.config.
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
        resample_uncert_correlation (bool, optional):
            If True, uncertainties are averaged.
            If False, uncertainties are calculated as RMSE-like aggregation.
        sector (str):
            Emissions sector to plot. Default 'total'.
        site_marker (str):
            Marker for site locations.
        city_marker (str):
            Marker for city locations.
        marker_color (str):
            Marker color.
    Returns:
        fig (figure):
            Three maps of a target flux variable of the first and second models and the diffence between both.
    """

    # Check for inversion_grid and sector option
    if "inversion_grid" in var and sector != "total":
        raise ValueError(
            f"Currently, you cannot plot sectors other than 'total' using the inversion_grid variable. "
            + "Choose a non inversion_grid variable to plot other sectors."
        )

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

    # Prepare datasets and resample over the whole time period (season=None) or a given season
    ds_dict = {
        m: define_var_plot(ds, var, sector) for m, ds in ds_all.items() if m in models
    }
    ds_dict = align_map_data(ds_dict)
    ds_dict["diff"] = make_model_diff_ds(ds_dict[models[0]], ds_dict[models[1]])

    for m, ds in ds_dict.items():
        ds_dict[m] = resample_over_period(
            ds, chop_by=season, resample_uncert_correlation=resample_uncert_correlation
        )[0]

    # Load country lines and species information
    country_lines = compute_boundary_geometry(map_bounds)
    species_info = config_data["species_info"][species]

    # Set flux limits
    lim = set_flux_limits(
        ds_dict,
        var,
        map_bounds,
        option=set_fluxlim,
        custom_percentile=set_fluxlim_percentile,
    )

    # Initialize figure
    n_rows = 1
    n_cols = 3
    figsize = define_map_figsize(
        map_bounds, n_rows, n_cols, fixed_value=5 * n_cols, fixed_dimension="width"
    )
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, layout="compressed")
    for col, (model, ds) in enumerate(ds_dict.items()):
        ax_i = ax[col]
        lon, lat = ds.longitude, ds.latitude

        # Determine plot settings
        is_diff = ("diff" in var) or ("diff" in model)
        cmap_i = cmap_diff if is_diff else cmap
        border_color = c_border_diff if is_diff else c_border
        vlim_i = (-lim[1], lim[1]) if is_diff else lim
        if marker_color is None:
            marker_color = "black" if is_diff else "red"
        extend_i = "both" if is_diff else "max"

        # Plot the data
        im = ax_i.pcolormesh(
            lon,
            lat,
            ds[var],
            cmap=cmap_i,
            vmin=vlim_i[0],
            vmax=vlim_i[1],
            shading="nearest",
        )
        plot_country_borders(ax=ax_i, lines=country_lines, border_color=border_color)
        ax_i.set_xlim(map_bounds[:2])  # Longitude limits
        ax_i.set_ylim(map_bounds[2:])  # Latitude limits
        ax_i.set_aspect(1)

        # Adjust ticks layout
        if col > 0:
            ax_i.set_yticklabels([])

        # Add titles
        if model == "diff":
            ax_i.set_title(f"{model_labels[models[0]]} - {model_labels[models[1]]}")
        else:
            ax_i.set_title(model_labels[model])

        # Add sites and markers if specified
        if add_sites:
            try:
                sites_info = get_active_sites_coordinates(
                    ds, config_data, fallback_sites
                )
            except Exception as e:
                raise RuntimeError(
                    "Failed to get active sites coordinates. "
                    "Check that `add_sites_to_flux` is True in `read_model_output` "
                    "or that a `fallback_sites` list is provided in `plot_flux_map`."
                ) from e
            add_site_markers(ax_i, sites_info, marker_color, site_marker)
        if add_markers:
            add_custom_markers(
                ax_i,
                add_markers,
                marker_color,
                config_data["regions_info"],
                city_marker,
            )

        # Add colorbar
        cbar_label = print_cbar_label(
            ds,
            species_info,
            var,
            format=["variable", "species", "units", "time"],
        )  # TODO Here, based on the last iteration. Check if consistent for all models?
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
    resample_uncert_correlation: bool = False,
    sector: str = "total",
    include_title_and_labels: bool = True,
    add_gridlines: bool = False,
    site_marker: str = "o",
    city_marker: str = "^",
    marker_color: str | None = None,
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
            List of model_labels from fluxie.config.
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
        resample_uncert_correlation (bool, optional):
            If True, uncertainties are averaged directly.
            If False, uncertainties are calculated as RMSE-like aggregation.
        sector (str):
            Emissions sector to plot. Default 'total'.
        site_marker (str):
            Marker for site locations.
        city_marker (str):
            Marker for city locations.
        marker_color (str):
            Marker color.
    Returns:
        fig (figure):
            A plot of spatial flux of the variable specified in var
            averaged over the number of time steps specified in dt.
    """

    # Check for inversion_grid and sector option
    if "inversion_grid" in var and sector != "total":
        raise ValueError(
            f"Currently, you cannot plot sectors other than 'total' using the inversion_grid variable. "
            + "Choose a non inversion_grid variable to plot other sectors."
        )

    # Determine geographical boundaries
    map_bounds = get_map_bounds(
        region,
        ds_all.values(),
        config_data,
        zoom_degree=zoom_degree,
    )

    # Prepare datasets and resample over given periods
    ds_dict = {m: define_var_plot(ds, var, sector) for m, ds in ds_all.items()}

    if plot_combined:
        ds_dict = align_map_data(ds_dict)
        ds_dict = combine_map_dataset(ds_dict)

    time_labels = {}
    for key, ds in ds_dict.items():
        ds_dict[key], time_labels[key] = resample_over_period(
            ds, dt, chop_by, resample_uncert_correlation
        )

    if all([v == time_labels[key] for v in time_labels.values()]):
        time_labels = time_labels[key]
    else:
        raise ValueError(
            f"Uncoherent `time_labels` derived : {time_labels}. Most probable reason is difference between start and end dates of the datasets, slicing them to their common period should resolve the issue."
        )

    # Load country lines, species and sites information
    country_lines = compute_boundary_geometry(map_bounds)
    species_info = config_data.get("species_info", {}).get(species, {})

    # Set flux limits
    lim = set_flux_limits(
        ds_dict,
        var,
        map_bounds,
        option=set_fluxlim,
        custom_percentile=set_fluxlim_percentile,
    )

    # Determine plot settings
    is_diff = "diff" in var
    cmap = cmap_diff if is_diff else cmap
    border_color = c_border_diff if is_diff else c_border
    if marker_color is None:
        marker_color = "black" if is_diff else "darkblue"
    extend = "both" if is_diff else "max"

    # Initialise figure
    n_rows = len(ds_dict.keys())
    n_cols = len(time_labels)

    is_single_season = chop_by == "season" and n_rows == 1
    if is_single_season:
        fig_rows = 2
        fig_cols = 2
        fixed_value = 7
    else:
        fig_rows = n_rows
        fig_cols = n_cols
        fixed_value = 4 * n_rows

    figsize = define_map_figsize(
        map_bounds,
        fig_rows,
        fig_cols,
        fixed_value=fixed_value,
        fixed_dimension="height",
    )
    fig, ax = plt.subplots(fig_rows, fig_cols, figsize=figsize, layout="compressed")
    ax = ax.flatten() if is_single_season else ax

    for row, (model, ds) in enumerate(ds_dict.items()):
        lon, lat = ds.longitude, ds.latitude

        for col, time_label in enumerate(time_labels):
            if n_rows == 1 and n_cols == 1:
                ax_i = ax
            elif n_rows == 1:
                ax_i = ax[col]
            elif n_cols == 1:
                ax_i = ax[row]
            else:
                ax_i = ax[row, col]

            var_i = ds[var].isel(time=col)

            # Plot the data
            im = ax_i.pcolormesh(
                lon,
                lat,
                var_i,
                cmap=cmap,
                vmin=lim[0],
                vmax=lim[1],
                shading="nearest",
            )
            plot_country_borders(
                ax=ax_i, lines=country_lines, border_color=border_color
            )
            ax_i.set_xlim(map_bounds[:2])  # Longitude limits
            ax_i.set_ylim(map_bounds[2:])  # Latitude limits
            ax_i.set_aspect(1)

            # Adjust ticks layout
            if is_single_season:
                if col in [0, 1] or not include_title_and_labels:
                    ax_i.set_xticklabels([])
                if col in [1, 3] or not include_title_and_labels:
                    ax_i.set_yticklabels([])
            else:
                if row < n_rows - 1 or not include_title_and_labels:
                    ax_i.set_xticklabels([])
                if col > 0 or not include_title_and_labels:
                    ax_i.set_yticklabels([])

            # Add titles
            if row == 0 and include_title_and_labels:
                # Column titles
                ax_i.set_title(time_label)
            if col == 0 and not plot_combined and include_title_and_labels:
                # Row titles
                ax_i.set_ylabel(model_labels.get(model, model))

            # Add sites and markers if specified
            if add_sites:
                try:
                    sites_info = get_active_sites_coordinates(
                        ds.isel(time=[col]), config_data, fallback_sites
                    )
                except Exception as e:
                    raise RuntimeError(
                        "Failed to get active sites coordinates. "
                        "Check that `add_sites_to_flux` is True in `read_model_output` "
                        "or that a `fallback_sites` list is provided in `plot_flux_map`."
                    ) from e
                add_site_markers(ax_i, sites_info, marker_color, site_marker)

            if add_markers:
                add_custom_markers(
                    ax_i,
                    add_markers,
                    marker_color,
                    config_data["regions_info"],
                    city_marker,
                )

            if add_gridlines:
                ax_i.grid(visible=True, which="major", alpha=0.4)

    if include_title_and_labels:
        cbar_label_format = ["variable", "species", "units", "time"]
    else:
        cbar_label_format = ["species", "units", "time"]

    # Add colorbar
    cbar_label = print_cbar_label(
        ds,
        species_info,
        var,
        sector=sector if sector != "total" else "",
        format=cbar_label_format,
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


def plot_flux_map_combined_models_comparison(
    ds_all: dict[xr.Dataset],
    group_a_models: list[str],
    group_b_models: list[str],
    var: str,
    species: str,
    region: Region = None,
    config_data: dict = {},
    group_a_label: str = None,
    group_b_label: str = None,
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
    resample_uncert_correlation: bool = False,
    sector: str = "total",
    site_marker: str = "o",
    city_marker: str = "^",
    marker_color: str | None = None,
) -> plt.Figure:
    """
    Plot a given flux variable for two groups of combined models and the difference between them.

    Args:
        ds_all (dictionary of datasets):
            Dictionary of fluxes xarray datasets.
        group_a_models (list[str]):
            List of model names to be combined for the first group.
        group_b_models (list[str]):
            List of model names to be combined for the second group.
        var (str):
            The name of the flux variable to be plotted and compared across models.
            Example: 'flux_total_posterior'.
        species (str):
            Gas species, e.g. 'ch4'.
        region (str or list):
            Lat/lon region to plot, options for 'UK', 'FRANCE', 'GERMANY',
            'NWEU','CWEU','EUROPE'.
            A list with [lon_min, lon_max, lat_min, lat_max] can also be provided.
        config_data (dict of dict):
            Dictionary of models and species information (read from json file).
        group_a_label: str,
            Label for the first group of combined models.
        group_b_label: str,
            Label for the second group of combined models.
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
        resample_uncert_correlation (bool, optional):
            If True, uncertainties are averaged.
            If False, uncertainties are calculated as RMSE-like aggregation.
        sector (str):
            Emissions sector to plot. Default 'total'.
        site_marker (str):
            Marker for site locations.
        city_marker (str):
            Marker for city locations.
        marker_color (str):
            Marker color.
    Returns:
        fig (figure):
            Three maps of a target flux variable of the first and second groups of combined models and the diffence between both.
    """

    # Check for inversion_grid and sector option
    if "inversion_grid" in var and sector != "total":
        raise ValueError(
            f"Currently, you cannot plot sectors other than 'total' using the inversion_grid variable. "
            + "Choose a non inversion_grid variable to plot other sectors."
        )

    # Determine geographical boundaries
    map_bounds = get_map_bounds(
        region,
        ds_all.values(),
        config_data,
        zoom_degree=zoom_degree,
    )

    # Prepare datasets
    ds_dict = {m: define_var_plot(ds, var, sector) for m, ds in ds_all.items()}
    ds_dict = align_map_data(ds_dict)

    # Combine models from group_a and group_b and keep only 'combined' variable
    ds_group_a = combine_map_dataset({k: ds_dict[k] for k in group_a_models})
    ds_group_b = combine_map_dataset({k: ds_dict[k] for k in group_b_models})

    ds_comparison = {}
    ds_comparison["group_a"] = ds_group_a["combined"]
    ds_comparison["group_b"] = ds_group_b["combined"]
    ds_comparison["diff"] = make_model_diff_ds(
        ds_comparison["group_a"], ds_comparison["group_b"]
    )

    # Resample over the whole time period (season=None) or a given season
    for m, ds in ds_comparison.items():
        ds_comparison[m] = resample_over_period(
            ds, chop_by=season, resample_uncert_correlation=resample_uncert_correlation
        )[0]

    # Define group labels
    group_a_label = group_a_label or "\n".join(group_a_models)
    group_b_label = group_b_label or "\n".join(group_b_models)
    separator = "\n-\n" if ("\n" in group_a_label or "\n" in group_b_label) else " - "
    diff_label = f"{group_a_label}{separator}{group_b_label}"

    labels = {
        "group_a": group_a_label,
        "group_b": group_b_label,
        "diff": diff_label,
    }

    # Load country lines, species and sites information
    country_lines = compute_boundary_geometry(map_bounds)
    species_info = config_data.get("species_info", {}).get(species, {})

    # Set flux limits
    lim = set_flux_limits(
        ds_comparison,
        var,
        map_bounds,
        option=set_fluxlim,
        custom_percentile=set_fluxlim_percentile,
    )

    # Initialise figure
    n_rows = 1
    n_cols = 3
    figsize = define_map_figsize(
        map_bounds, n_rows, n_cols, fixed_value=5 * n_cols, fixed_dimension="width"
    )

    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, layout="compressed")
    for col, (model, ds) in enumerate(ds_comparison.items()):
        ax_i = ax[col]
        lon, lat = ds.longitude, ds.latitude

        # Determine plot settings
        is_diff = ("diff" in var) or ("diff" in model)
        cmap_i = cmap_diff if is_diff else cmap
        border_color = c_border_diff if is_diff else c_border
        vlim_i = (-lim[1], lim[1]) if is_diff else lim
        if marker_color is None:
            marker_color = "black" if is_diff else "red"
        extend_i = "both" if is_diff else "max"

        # Plot the data
        im = ax_i.pcolormesh(
            lon,
            lat,
            ds[var],
            cmap=cmap_i,
            vmin=vlim_i[0],
            vmax=vlim_i[1],
            shading="nearest",
        )
        plot_country_borders(ax=ax_i, lines=country_lines, border_color=border_color)
        ax_i.set_xlim(map_bounds[:2])  # Longitude limits
        ax_i.set_ylim(map_bounds[2:])  # Latitude limits
        ax_i.set_aspect(1)

        # Adjust ticks layout
        if col > 0:
            ax_i.set_yticklabels([])

        # Add titles
        ax_i.set_title(labels[model])

        # Add sites and markers if specified
        if add_sites:
            try:
                sites_info = get_active_sites_coordinates(
                    ds, config_data, fallback_sites
                )
            except Exception as e:
                raise RuntimeError(
                    "Failed to get active sites coordinates. "
                    "Check that `add_sites_to_flux` is True in `read_model_output` "
                    "or that a `fallback_sites` list is provided in `plot_flux_map`."
                ) from e
            add_site_markers(ax_i, sites_info, marker_color, site_marker)
        if add_markers:
            add_custom_markers(
                ax_i,
                add_markers,
                marker_color,
                config_data["regions_info"],
                city_marker,
            )

        # Add colorbar
        cbar_label = print_cbar_label(
            ds,
            species_info,
            var,
            format=["variable", "species", "units", "time"],
        )  # TODO Here, based on the last iteration. Check if consistent for all models?
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


def plot_flux_map_period_comparison(
    ds_all: dict[xr.Dataset],
    var: str,
    species: str,
    start_dates: list[str],
    end_dates: list[str],
    region: Region = None,
    config_data: dict = {},
    model_labels: dict[str] = {},
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
    resample_uncert_correlation: bool = False,
    sector: str = "total",
    site_marker: str = "o",
    city_marker: str = "^",
    marker_color: str | None = None,
) -> plt.Figure:
    """
    Plot a given flux variable averaged over two time periods and the difference, for all models or the model mean.
    Args:
        ds_all (dictionary of datasets):
            Dictionary of fluxes xarray datasets.
        var (str):
            The name of the flux variable to be plotted and compared across models.
            Example: 'flux_total_posterior'.
        species (str):
            Gas species, e.g. 'ch4'.
        start_dates (list[str]):
            List of starting dates for the two periods to compare (format: 'YYYY-MM-DD').
        end_dates (list[str]):
            List of ending dates for the two periods to compare (format: 'YYYY-MM-DD').
        region (str or list):
            Region to plot, e.g. 'FRANCE', 'EUROPE', [lon_min, lon_max, lat_min, lat_max].
        config_data (dict of dict):
            Dictionary of models and species information (read from json file).
        model_labels (list):
            List of model_labels from fluxie.config.
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
        resample_uncert_correlation (bool, optional):
            If True, uncertainties are averaged directly.
            If False, uncertainties are calculated as RMSE-like aggregation.
        sector (str):
            Emissions sector to plot. Default 'total'.
        site_marker (str):
            Marker used for site locations.
        city_marker (str):
            Marker used for city locations.
        marker_color (str):
            Marker color.
    Returns:
        fig (figure):
            A plot of spatial flux of the variable specified in var
            averaged over the number of time steps specified in dt.
    """

    # Check for inversion_grid and sector option
    if "inversion_grid" in var and sector != "total":
        raise ValueError(
            f"Currently, you cannot plot sectors other than 'total' using the inversion_grid variable. "
            + "Choose a non inversion_grid variable to plot other sectors."
        )

    # Determine geographical boundaries
    map_bounds = get_map_bounds(
        region,
        ds_all.values(),
        config_data,
        zoom_degree=zoom_degree,
    )

    # Prepare datasets and resample over given periods
    ds_dict = {m: define_var_plot(ds, var, sector) for m, ds in ds_all.items()}

    if plot_combined:
        ds_dict = align_map_data(ds_dict)
        ds_dict = combine_map_dataset(ds_dict)

    time_labels = {}
    for key, ds in ds_dict.items():
        ds_dict[key], time_labels[key] = resample_over_period(
            ds,
            chop_by=(start_dates, end_dates),
            resample_uncert_correlation=resample_uncert_correlation,
        )

    if all([v == time_labels[key] for v in time_labels.values()]):
        time_labels = time_labels[key]
    else:
        raise ValueError(
            f"Uncoherent `time_labels` derived : {time_labels}. Most probable reason is difference between start and end dates of the datasets, slicing them to their common period should resolve the issue."
        )

    # Load country lines, species and sites information
    country_lines = compute_boundary_geometry(map_bounds)
    species_info = config_data.get("species_info", {}).get(species, {})

    # Set flux limits
    lim = set_flux_limits(
        ds_dict,
        var,
        map_bounds,
        option=set_fluxlim,
        custom_percentile=set_fluxlim_percentile,
    )

    # Initialise figure
    n_rows = len(ds_dict.keys())
    n_cols = 3  # Two periods + difference

    figsize = define_map_figsize(
        map_bounds, n_rows, n_cols, fixed_value=5 * n_cols, fixed_dimension="width"
    )

    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, layout="compressed")

    for row, (model, ds) in enumerate(ds_dict.items()):
        lon, lat = ds.longitude, ds.latitude

        for col in range(n_cols):
            if n_rows == 1 and n_cols == 1:
                ax_i = ax
            elif n_rows == 1:
                ax_i = ax[col]
            elif n_cols == 1:
                ax_i = ax[row]
            else:
                ax_i = ax[row, col]

            if col < 2:
                var_i = ds[var].isel(time=col)
            else:
                var_i = ds[var].isel(time=1) - ds[var].isel(time=0)

            # Determine plot settings
            is_diff = ("diff" in var) or (col == 2)
            cmap_i = cmap_diff if is_diff else cmap
            border_color = c_border_diff if is_diff else c_border
            vlim_i = (-lim[1], lim[1]) if is_diff else lim
            if marker_color is None:
                marker_color = "black" if is_diff else "magenta"
            extend_i = "both" if is_diff else "max"

            # Plot the data
            im = ax_i.pcolormesh(
                lon,
                lat,
                var_i,
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

            # Adjust ticks layout
            if row < n_rows - 1:
                ax_i.set_xticklabels([])
            if col > 0:
                ax_i.set_yticklabels([])

            # Add titles
            if row == 0:
                # Column titles
                if col < 2:
                    ax_i.set_title(time_labels[col])

            if col == 0 and not plot_combined:
                # Row titles
                ax_i.set_ylabel(model_labels.get(model, model))

            # Add sites and markers if specified
            if add_sites:
                try:
                    if col < 2:
                        sites_info = get_active_sites_coordinates(
                            ds.isel(time=[col]), config_data, fallback_sites
                        )
                    else:
                        sites_info = get_active_sites_coordinates(
                            ds.isel(time=[0, 1]), config_data, fallback_sites
                        )
                except Exception as e:
                    raise RuntimeError(
                        "Failed to get active sites coordinates. "
                        "Check that `add_sites_to_flux` is True in `read_model_output` "
                        "or that a `fallback_sites` list is provided in `plot_flux_map`."
                    ) from e
                add_site_markers(ax_i, sites_info, marker_color, site_marker)

            if add_markers:
                add_custom_markers(
                    ax_i,
                    add_markers,
                    marker_color,
                    config_data["regions_info"],
                    city_marker,
                )

            # Add colorbar
            cbar_label = print_cbar_label(
                ds,
                species_info,
                var,
                format=["species", "units"],
            )
            add_colorbar(
                fig,
                ax_i,
                im,
                extend=extend_i,
                label=cbar_label,
                n_cbar=3,
                idx_cbar=col,
                colorbar_type="column",
            )

    return fig
