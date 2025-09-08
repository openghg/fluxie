import itertools
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import logging
import re
import warnings

from shapely.geometry import MultiPolygon, Polygon
from typing import Literal
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from copy import deepcopy

from fluxy import config
from fluxy.io import load_countries_shape

logger = logging.getLogger(__name__)


def update_list_params(params_to_check: list, expected_size: int) -> list:
    """
    Check if parameters are list of the expected lenght. If they are not list, convert them to list (except is it is None), raise an erro if it is a list but not of the expected size.
    Args:
        params_to_check : parameters to be checked
        expected_size : expected size for the list (should be the number of models used in the plots)
    Returns
        updated_params: the updated list of lists
    """
    updated_params = list()
    for param in params_to_check:
        if param is None:
            updated_params.append([False] * expected_size)
        elif type(param) is list:
            if len(param) == expected_size:
                updated_params.append(param)
            else:
                raise ValueError(
                    f"{param} must be a boolean or a list of booleans of the same length as models."
                )
        else:
            updated_params.append([param] * expected_size)
    return updated_params


def add_colorbar(fig, ax, im, extend, label, n_cbar, idx_cbar, colorbar_type="row"):
    """Add a colorbar to the plot."""

    if colorbar_type == "row":
        cax = inset_axes(
            ax,
            width="5%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(1.05, 0.0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = fig.colorbar(im, cax=cax, orientation="vertical", extend=extend)

    elif colorbar_type == "column":
        cax = inset_axes(
            ax,
            width="100%",
            height="5%",
            loc="lower left",
            bbox_to_anchor=(0, -0.17, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal", extend=extend)

    elif colorbar_type == "figure":
        nrows = fig.axes[0].get_subplotspec().get_gridspec().nrows
        ncols = fig.axes[0].get_subplotspec().get_gridspec().ncols
        ax_dim = np.array(ax).ndim

        if ax_dim == 1:
            if nrows ==2 and ncols ==2: # single_season case
                target_ax = [ax[1], ax[3]]
            else:
                target_ax = ax[:]
        elif ax_dim == 2:
            target_ax = ax[:, -1]
        else:
            target_ax = ax

        cbar = fig.colorbar(
            im, ax=target_ax, orientation="vertical", extend=extend
        )

    else:
        raise ValueError(
            f"The colorbar option {colorbar_type} is not available. Options are column, row, or figure."
        )

    cbar.set_label(label)


def print_cbar_label(
    ds: xr.Dataset,
    species_info: dict,
    var: str = None,
    sector: str = "total",
    format: list[str] = ["variable", "sector", "species", "units", "time"],
) -> str:
    """
    Generate a colorbar label for a dataset variable.

    Args:
        ds (xr.DataArray):
            The DataArray containing the variable. The DataArray should be named with the name of the variable
            (e.g. "posterior_prior_diff", "flux_total_prior", ...)
        species_info (dict):
            A dictionary with metadata for species, including display names.
        var (str, optional):
            The variable name in the dataset.
        format (list[str], optional):
            Specifies the components to include in the label.
            Options: ['variable', 'species', 'units', 'time']. Default includes all.

    Returns:
        cbar_label(str):
            A formatted colorbar label including variable, species, units, and period.
    """

    var_label = f"{config.flux_labels[var]}" if "variable" in format else ""

    species_label = (
        f"{species_info.get('species_print')}" if "species" in format else ""
    )

    units_label = f"({get_units(ds[var])})" if "units" in format else ""

    sector_label = f"{sector}" if "sector" in format else ""

    middle_label = " ".join(filter(None, [species_label, sector_label, units_label]))

    time_label = ""
    if "time" in format:
        time_label = ds.attrs["time_label"]

    # Construct the final label with proper line breaks
    label_parts = [var_label, middle_label, time_label]
    cbar_label = "\n".join(filter(None, label_parts))  # Remove empty lines
    return cbar_label


def get_units(
    data: xr.DataArray,
) -> str:
    """
    Retrieve and format the units of an xarray DataArray.

    Args:
        data (xr.DataArray):
            The xarray DataArray from which to retrieve the units.

    Returns:
        period (str):
            A formatted string representing the units.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError("Input must be an xarray DataArray.")

    units = data.attrs.get("units")
    if units is None:
        raise AttributeError(
            "The 'units' attribute is missing in the provided DataArray."
        )

    formatted_units = format_units(units)
    return formatted_units


def format_units(units: str) -> str:
    """
    Format units string to replace negative exponents with LaTeX-compatible superscripts.

    Args:
        units (str):
            The units string to format.

    Returns:
        formatted_units (str):
            The formatted units string with LaTeX-compatible superscripts.
    """
    if not isinstance(units, str):
        raise TypeError("Input must be a string.")

    formatted_units = re.sub(r"-\d+", lambda m: f"$^{{{m.group()}}}$", units)
    return formatted_units


def get_frequency(
    ds: xr.Dataset,
) -> Literal["Y", "M"]:
    """
    Determine the temporal frequency of a dataset for a given species ('Y' for yearly, 'M' for monthly).

    Args:
        ds (xr.Dataset):
            The dataset containing metadata attributes related to frequency.

    Returns:
        Literal['Y', 'M']:
            A shorthand representation of the temporal frequency ('Y' for yearly, 'M' for monthly).
    """

    frequency_map = {
        "yearly": "Y",
        "monthly": "M",
        # Add more mappings as needed
    }

    frequency = ds.attrs.get("frequency")
    frequency = frequency_map.get(frequency, None)

    if frequency is None:
        raise ValueError(
            "The only valid inversion frequencies are yearly and monthly. If a new frequency is needed, update get_frequency."
        )

    return frequency


def print_period(
    ds: xr.Dataset,
    freq: str,
    season: str = None,
) -> str:
    """
    Generate a formatted string representing the time period covered by a dataset.

    Args:
        ds (xr.Dataset):
            The dataset containing a `time` dimension to determine the start and end dates.
        freq (str):
            The temporal frequency (e.g., 'Y' for yearly, 'M' for monthly) to format the dates.
        season (str, optional):
            A seasonal label (e.g., 'DJF', 'MAM') to include in the output. Default is None.

    Returns:
        period (str):
            A formatted string representing the dataset's time period, optionally including a season.
    """

    datetime_format = f"datetime64[{freq}]" if season is None else "datetime64[Y]"

    if "time" in ds.dims:
        start_date = ds.time.values.min()
        end_date = ds.time.values.max()
    elif "start_date" in ds.attrs and "end_date" in ds.attrs:
        start_date, end_date = ds.attrs["start_date"], ds.attrs["end_date"]
    else:
        raise ValueError("Cannot infer start and end dates from dataset")

    start_date = start_date.astype(datetime_format)
    end_date = end_date.astype(datetime_format)

    if start_date == end_date:
        period = f"{start_date}"
    else:
        period = f"{start_date}—{end_date}"

    if season:
        period = f"{season} of {period}"

    return period


def add_custom_markers(ax, markers, color, regions_info):
    """Add custom markers to the plot."""
    for marker in markers:
        lon, lat = get_marker_coordinates(marker, regions_info)
        ax.scatter(
            lon, lat, facecolor="none", edgecolor=color, marker="^", s=30, zorder=2
        )


def get_marker_coordinates(
    marker: str | tuple[float, float],
    regions_info: dict[str, str],
) -> tuple[float, float]:
    """
    Retrieve latitude and longitude coordinates for a specified marker.

    Args:
        marker (str | tuple):
            The location identifier.
            - If a string, it must match a key in `point_source` in regions_info.json.
            - If a tuple, it must contain exactly two numeric values representing (latitude, longitude).
        regions_info (dict of str):
            Dictionary with country and region names (read from json file).

    Returns:
        tuple:
            A tuple containing the latitude and longitude as `(lat_marker, lon_marker)`.
    """

    if isinstance(marker, str):
        if marker in regions_info["point_source"]:
            lon_marker, lat_marker = regions_info["point_source"][marker]
        else:
            raise ValueError(
                f"Location '{marker}' not found in point source dictionary."
            )

    elif isinstance(marker, tuple) and len(marker) == 2:
        lon_marker, lat_marker = marker

    else:
        raise TypeError(
            "Marker must be a string (location name) or a tuple of (lat, lon)."
        )

    return lon_marker, lat_marker


def add_site_markers(ax, site_info, color):
    """Add site markers to the plot."""
    for site, site_data in site_info.items():
        ax.scatter(
            site_data["longitude"],
            site_data["latitude"],
            facecolor="none",
            edgecolor=color,
            marker="o",
            s=30,
            zorder=2,
        )


def get_active_sites_coordinates(
    ds: xr.Dataset,
    config_data: dict,
    fallback_sites: list[str] | None = None,
) -> dict:
    """
    Retrieve coordinates for active platforms/sites from an xarray Dataset.

    Args:
        ds (xr.Dataset):
           xarray flux dataset.
        config_data (dict):
            Dictionary of sites with information for plotting (read from json file).
        fallback_sites (list[str] | None):
            A list of site/platform names to use if no active sites are found in `ds`.

    Returns:
        dict:
            A dictionary of site coordinates for either the active sites or the fallback sites.
            Returns an empty dict if no sites are found and no fallback_sites are provided.
    """

    sites = ds["sites"] if "sites" in ds else None

    if sites is None:
        if fallback_sites:
            logger.warning(
                "No active 'sites' found in dataset, using fallback sites from the list provided."
            )
            return extract_site_info(fallback_sites, config_data)

        else:
            logger.warning(
                "No 'sites' found in dataset. "
                "Please ensure 'add_sites_to_flux' is True in 'read_model_output' "
                "or that a 'fallback_sites' list is provided in plot_flux_map."
            )
            return {}

    active_sites = sites.platform.values[sites.any(dim="time").values].tolist()
    return extract_site_info(active_sites, config_data)


def get_sites_coordinates(
    ds_all: dict[xr.Dataset],
    config_data: dict,
    fallback_sites: list[str] | None = None,
) -> dict:
    """
    DEPRECATED: Use `get_active_sites_coordinates` instead.
    Collect the 'sites' attribute from a dictionary of xarray datasets.
    If 'sites' is missing, use it from another dataset where it's available.

    Args:
        ds_all (dict):
            Dictionary of xarray datasets.
        config_data (dict):
            Dictionary of sites with information for plotting (read from json file).
        fallback_sites (list[str] | None):
            A list of site names to use as a fallback if 'sites' is not found in the datasets.
            If None, the first available 'sites' in the datasets will be used as fallback.

    Returns:
        dict:
            A mapping of dataset keys to their respective 'sites' attribute.
    """
    warnings.warn(
        "'get_sites_coordinates' is deprecated and will be removed in a future release. "
        "Please use 'get_active_sites_coordinates' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return {
        key: get_active_sites_coordinates(ds, config_data, fallback_sites)
        for key, ds in ds_all.items()
    }


def extract_site_info(
    sites: list[str], config_data: dict[str, dict]
) -> dict[str, dict]:
    """
    Extract latitude and longitude for each site from site_info in the config data.

    Args:
        sites (list[str]): A list of site names to extract information for.
        config_data (dict[str, dict]): A dictionary containing configuration data,
                                       where 'site_info' holds the latitude and longitude info.

    Returns:
        sites_coordinates (dict[str, dict]): A dictionary mapping site names to their respective latitude and longitude.
    """

    site_info = config_data.get("site_info", {})
    sites_coordinates = {}

    for site in sites:
        if site not in site_info:
            logger.warning(
                f"Site '{site}' not found in config_data['site_info']. It will be skipped."
            )
            continue

        # Take the first entry in site_info[site]
        first_key = next(iter(site_info[site]))

        sites_coordinates[site] = {
            "latitude": site_info[site][first_key]["latitude"],
            "longitude": site_info[site][first_key]["longitude"],
        }

    return sites_coordinates


def get_bounds_from_datasets(
    ds_list: list[xr.Dataset],
) -> tuple[float, float, float, float]:
    """
    Get the bounds of a list of xarray datasets.

    Args:
        ds_list: list of xarray datasets to be bounded
    Returns:
        bounds: tuple of min and max values for latitude and longitude
    """
    lat_min = min([ds.latitude.min() for ds in ds_list])
    lat_max = max([ds.latitude.max() for ds in ds_list])
    lon_min = min([ds.longitude.min() for ds in ds_list])
    lon_max = max([ds.longitude.max() for ds in ds_list])

    return lon_min, lon_max, lat_min, lat_max


# Region type
Region = str | list[float] | tuple[float] | None


def get_map_bounds(
    region: Region = None,
    ds_all: list[xr.Dataset] = [],
    config_data: dict[str, any] = {},
    zoom_degree: float = 1,
) -> tuple[float, float, float, float]:
    """
    Get the bounding coordinates for a specified region or dataset.

    Three options for specifying the bounds based on the 'region' argument:
    1. A string representing a country, continent, or region name.
    2. A list of four floats representing the bounding box coordinates (lon_min, lon_max, lat_min, lat_max).
    3. None, in which case the bounds are read from the datasets.

    Args:
        ds_all (list[xr.Dataset]):
            A list of xarray datasets to get the bounds from.
        region (str | list[float] | None):
            The region name or bounding box coordinates. See above for details.
        config_data (dict[str, any]):
            Configuration data containing regions information.

    Returns:
        map_bounds (tuple[float, float, float, float]):
            The bounding coordinates of the region or dataset (lon_min, lon_max, lat_min, lat_max).


    """
    ds_all = list(ds_all)
    if isinstance(region, str):
        # Use the non-zero country_fraction to define the clipping region, for coherence in the country definition
        if len(ds_all) > 0 and "country_fraction" in ds_all[0]:
            da_mask = ds_all[0].country_fraction.sum(dim="country")
            clipped = (
                da_mask.where(da_mask != 0)
                .dropna(dim="longitude", how="all")
                .dropna(dim="latitude", how="all")
            )
            clip_region = [
                clipped.longitude.values.min(),
                clipped.latitude.values.min(),
                clipped.longitude.values.max(),
                clipped.latitude.values.max(),
            ]
        else:
            clip_region = None

        map_bounds = get_region_coordinates(
            region,
            config_data.get("regions_info", {}),
            zoom_degree=zoom_degree,
            clip_region=clip_region,
        )
    elif isinstance(region, (list, tuple)) and all(
        isinstance(coord, (int, float)) for coord in region
    ):
        map_bounds = tuple(region)
    elif region is None:
        if len(ds_all) == 0:
            raise ValueError("No datasets provided to determine bounds.")
        # Read the bounds from the dataset
        map_bounds = get_bounds_from_datasets(ds_all)
    else:
        if not isinstance(region, str):
            raise ValueError(
                "Invalid input: 'region' must be a string or a list of numbers."
            )

    return map_bounds


def get_region_coordinates(
    region_name: str,
    regions_info: dict[str, str],
    zoom_degree: float = 1,
    clip_region: list[float] = None,
) -> tuple[float, float, float, float]:
    """
    Get the bounding coordinates of a specified region with an option to zoom in/out.

    Args:
        region_name (str):
            The name of the country or continent or region to get the coordinates for.
        regions_info (dict of str):
            Dictionary with country and region names (read from json file).
        zoom_degree (float):
            The number of degrees to zoom in/out from the bounding box. Default is 1.
        clip_region (list[float]):
            Coordinates ([min_lon, min_lat, max_lon, max_lat]) use to restrict the boundaries of the region.
            For example, if the focus is on France and clip_region is the extent of continental Europe, overseas territory
            (Reunion, Mayotte,...) won't be included; if no clip region is provided, they will be included.

    Returns:
        region_coordinates (tuple):
            The bounding coordinates of the region (lon_min, lon_max, lat_min, lat_max), after zooming.
    """
    world = load_countries_shape()
    region_code = regions_info.get("country_codes", {})

    region_name_title = region_name.title()

    if region_name_title in world["NAME"].values:
        region = world[world["NAME"] == region_name_title].copy()
    elif region_name_title in world["CONTINENT"].values:
        region = world[world["CONTINENT"] == region_name_title].copy()
        # Exclude Russia from Europe
        if region_name_title == "Europe":
            region = region[region["NAME"] != "Russia"]
    elif region_name_title in world["SUBREGION"].values:
        region = world[world["SUBREGION"] == region_name_title].copy()
    else:
        region_name_upper = region_name.upper()

        if region_name_upper in world["ISO_A3"].values:
            region = world[world["ISO_A3"] == region_name_upper].copy()
        # If region_name_upper is in the region_code dictionary, get its corresponding ISO_A3 code(s)
        elif region_name_upper in region_code:
            iso_codes = region_code[
                region_name_upper
            ]  # Could be a single ISO_A3 or a list
            # If it's a group (contains '-'), split into a list
            if "-" in iso_codes:
                iso_codes = iso_codes.split("-")
                region = world[world["ISO_A3"].isin(iso_codes)].copy()
            else:
                region = world[world["ISO_A3"] == iso_codes].copy()
        else:
            raise ValueError(
                f"Region '{region_name}' not found. Please ensure the region is correctly specified as a country, continent, subregion, or ISO_A3 code.\n"
                f"You can also try using a predefined region from regions dictionary in regions_info.json.\n"
                "If you're still having trouble, check and update the country_codes in regions_info.json if necessary."
            )

    # Handle empty region case
    if region.empty:
        raise ValueError(f"No coordinates found for region '{region_name}'.")

    # Restrict to clip_region
    if clip_region:
        region = gpd.clip(region, clip_region)

    # Get the bounding box of the region of interest
    region_boundaries = region.total_bounds  # [minx, miny, maxx, maxy]

    # Apply zoom adjustment
    lon_min = region_boundaries[0] - zoom_degree
    lat_min = region_boundaries[1] - zoom_degree
    lon_max = region_boundaries[2] + zoom_degree
    lat_max = region_boundaries[3] + zoom_degree

    region_coordinates = (lon_min, lon_max, lat_min, lat_max)
    return region_coordinates


def compute_boundary_geometry(map_bounds):
    """
    Compute the boundary geometry (lines) from a shapefile based on the given map bounds.

    This step improves performance by precomputing the geometry only once,
    compared to directly plotting the boundaries inside the loop in flux_map.py.

    Args:
        map_bounds (tuple):
            Bounding box (minx, miny, maxx, maxy) to filter countries.

    Returns:
        list: A list of coordinate sequences representing country boundaries.
    """
    gdf = load_countries_shape(map_bounds)
    lines = []
    for geom in gdf.boundary.geometry:
        if geom.geom_type == "LineString":
            lines.append(list(geom.coords))  # Extract coordinates
        elif geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                lines.append(list(line.coords))  # Extract all LineString coords
    return lines


def plot_country_borders(ax, lines, border_color):
    """
    Plot country borders on a given matplotlib axis.

    Args:
        ax (matplotlib.axes.Axes):
            The axis to add the country borders to.
        lines (list):
            A list of coordinate sequences representing country boundaries.
        border_color (str):
            Color for the country borders.
    """
    boundary_collection = deepcopy(
        LineCollection(lines, color=border_color, linewidth=1)
    )
    ax.add_collection(boundary_collection)


def set_flux_limits(
    ds_all: dict[xr.Dataset],
    var: str,
    region_plot: tuple[float, float, float, float],
    option: Literal["auto"] | list[float] | tuple[float, float] = "auto",
    custom_percentile: float = None,
) -> tuple[float, float]:
    """
    Set flux limits based on the option provided:

    1. List or tuple with two values (e.g., [0, 1]) - use specified limits.
    2. 'auto' - auto-calculate limits based on data percentiles.

    Args:
        ds_all (dict[xr.Dataset]):
            A dictionary of Datasets containing the flux variables
        var (str):
            Variable use to define the flux limits
        region_plot (tuple[float, float, float, float]):
            Coordinates [lon_min, lon_max, lat_min, lat_max].
        option ('auto', [lower_lim, upper_lim]):
            The option for setting limits.
                - A list or tuple with two elements (lower_lim, upper_lim) for specified limits.
                - 'auto' for auto-calculated values using the 99th percentile (if custom_percentile is None).
        custom_percentile (float, optional):
            The percentile to use as the upper limit if option is 'auto'.

    Returns:
        flux_lim (tuple):
            A tuple containing the flux limits (lower_lim, upper_lim).
    """

    # Case 1: Use specified values [lower_lim, upper_lim]
    if isinstance(option, (list, tuple)) and len(option) == 2:
        flux_lim = tuple(option)

    # Case 2: Auto-calculate limits based on percentiles
    elif option == "auto":
        models_var = []
        for model, ds in ds_all.items():
            var_plot = ds[var]
            # Filter based on longitude and latitude of region_plot
            mask_region = (
                (var_plot.longitude > region_plot[0])
                & (var_plot.longitude < region_plot[1])
                & (var_plot.latitude > region_plot[2])
                & (var_plot.latitude < region_plot[3])
            )
            var_plot = var_plot.where(mask_region, drop=True)
            models_var.append(var_plot)

        models_var = xr.concat(models_var, dim="time")

        # Calculate upper limit based on custom_percentile or default to 99th percentile
        upper_lim = models_var.quantile(
            custom_percentile if custom_percentile else 0.99
        ).item()

        if "diff" in models_var.name:
            flux_lim = (-upper_lim, upper_lim)
        else:
            flux_lim = (0, upper_lim)

    else:
        raise ValueError(
            f"Invalid option '{option}'. Use a (lower_lim, upper_lim) tuple, or 'auto'."
        )

    # Validation: Check that flux_lim[0] is smaller than flux_lim[1]
    if flux_lim[0] >= flux_lim[1]:
        raise ValueError(
            f"The lower flux limit {flux_lim[0]} must be less than the upper flux limit {flux_lim[1]}."
        )

    return flux_lim


def set_min_decimal_points(value: float, sig_fig: int = 2, dec_points: int = 2) -> str:
    """
    Converts float to string with a specified number of significant digits and
    decimal points.

    Args:
        value (float):
            Floating point number to be converted to string.
        sig_fig (int):
            Number of significant figures to specify in the output if 'value'
            is lowet than 1.
        dec_points (int):
            Number of decimal points to specify in the output if 'value'
            is greater or equal than 1.
    Returns:
        formatted_str (str):
            Floating point 'value' converted to string.
            E.g. if value=0.00123 and sig_fig=2 -> formatted_str=0.0012
                 if value=10.456 and dec_points=2 -> formatted_str=10.45
    """

    formatted_str = f"{value:.{dec_points}f}" if value >= 1 else f"{value:.{sig_fig}g}"

    return formatted_str


def define_map_figsize(
    map_bounds: tuple[float, float, float, float],
    n_rows: int,
    n_cols: int,
    fixed_value: float = 5,
    fixed_dimension: str = "width",
) -> tuple[float, float]:
    """
    Dynamically computes figsize for map subplots, allowing to fix either 'height' or 'width'.

    Args:
    - map_bounds: (lon_min, lon_max, lat_min, lat_max)
    - n_rows: Number of subplot rows
    - n_cols: Number of subplot columns
    - fixed_value: Fixed height (if fixed_dimension="height") or fixed width (if fixed_dimension="width")
    - fixed_dimension: "height" to fix height and adjust width, or "width" to fix width and adjust height,
    or None to adjust height and width.

    Returns:
    - figsize tuple (width, height)
    """
    if fixed_dimension not in ["height", "width", None]:
        raise ValueError("fixed_dimension must be either 'height' or 'width' or None")

    lon_min, lon_max, lat_min, lat_max = map_bounds
    aspect_ratio = (lat_max - lat_min) / (lon_max - lon_min)

    if fixed_dimension == "height":
        subplot_height = fixed_value / n_rows
        subplot_width = subplot_height / aspect_ratio
        fig_width = n_cols * subplot_width
        fig_height = fixed_value
    elif fixed_dimension == "width":
        subplot_width = fixed_value / n_cols
        subplot_height = subplot_width * aspect_ratio
        fig_width = fixed_value
        fig_height = n_rows * subplot_height
    else:
        subplot_height = fixed_value
        fig_height = subplot_height * n_rows
        fig_width = fig_height * aspect_ratio * n_cols

    # Limit maximum figure size to avoid too large figures
    fig_height = min(fig_height, 20)
    fig_width = min(fig_width, 20)
    return (fig_width, fig_height)


def stack_plot(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
    area: bool = False,
    colors_of_category: dict[str, str] = {},
):
    """Function to plot stacked bar plots for the emissions data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the emissions data.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on, by default None
    area : bool, optional
        If True, use area plot instead of bar plot, by default False
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    default_colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    colors = {
        cat: colors_of_category.get(cat, next(default_colors)) for cat in df.columns
    }
    if area:
        # Use the same labels and colors for the positive and negative values
        total_pos = np.zeros(df.shape[0])
        total_neg = np.zeros(df.shape[0])
        for i, column in enumerate(df.columns):
            values = df[column].values
            ax.fill_between(
                df.index,
                y1=np.where(values >= 0, total_pos, total_neg),
                y2=np.where(values >= 0, total_pos + values, total_neg + values),
                color=colors.get(column, None),
                label=column,
            )
            total_pos += np.clip(values, 0, None)
            total_neg += np.clip(values, None, 0)

        # ax.set_ylim(df_neg.sum(axis=1).min() * 1.1, df_pos.sum(axis=1).max() * 1.1)
    else:
        ax = df.plot.bar(stacked=True, ax=ax, color=colors)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        list(reversed(handles)),
        list(reversed(labels)),
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    return ax
