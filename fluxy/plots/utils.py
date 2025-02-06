import numpy as np
import xarray as xr
import geopandas as gpd
import logging
import re

from shapely.geometry import MultiPolygon, Polygon
from typing import Literal

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
                raise ValueError(f'{param} must be a boolean or a list of booleans of the same length as models.')        
        else:
            updated_params.append([param] * expected_size)
    return updated_params

def add_colorbar(fig, ax, mappable, cmap, extend, label, presentation_mode=False, orientation="vertical"):
    """Add a colorbar to the plot."""
    labelpad_v = 20 if presentation_mode else 5

    color_bar = fig.colorbar(
        mappable, orientation=orientation, extend=extend, ax=ax, shrink=1, pad=0.01
    )
    color_bar.set_label(label, labelpad=labelpad_v)

def print_cbar_label(
    ds: xr.Dataset, 
    species_info: dict, 
    var: str, 
    season: str, 
    period_override: str = None,
    ) -> str:
    """
    Generate a colorbar label for a dataset variable.

    Args:
        ds (xr.Dataset): 
            The dataset containing the variable.
        species_info (dict): 
            A dictionary with metadata for species, including display names.
        var (str): 
            The variable name in the dataset.
        season (str): 
            The season to include in the label (e.g., 'DJF', 'MAM').
        period_override (str, optional): 
            Overrides the default period calculation. 

    Returns:
        cbar_label(str): 
            A formatted colorbar label including species, units, and period.
    """
    species_print = species_info.get("species_print")

    var_units = get_units(ds[var])

    freq = get_frequency(ds, species_info, period_override) #TODO  Remove 'period_override' once 'coarser_frequency' attribute is implemented
    period = print_period(ds, freq, season) #TODO Here, based on the last iteration. Check if consistent for all models?
    
    cbar_label = f"{species_print} ({var_units})\n{period}"

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

    units = data.attrs.get('units')
    if units is None:
        raise AttributeError("The 'units' attribute is missing in the provided DataArray.")

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
    species_info: dict, 
    period_override: str = None,
    ) -> Literal['Y', 'M']:
    """
    Determine the temporal frequency of a dataset for a given species ('Y' for yearly, 'M' for monthly).

    Args:
        ds (xr.Dataset): 
            The dataset containing metadata attributes related to frequency.
        species_info (dict): 
            A dictionary containing additional information about species and their associated periods.
        period_override (str, optional): 
            A user-provided override for the frequency. Default is None.

    Returns:
        Literal['Y', 'M']: 
            A shorthand representation of the temporal frequency ('Y' for yearly, 'M' for monthly). 
            Returns None if no valid frequency is found.
    """

    frequency_map = {
        'yearly': 'Y',
        'monthly': 'M',
        # Add more mappings as needed
    }

    frequency_sources = [
        # First, check for coarser frequency
        ds.get('coarser_frequency', None),  # TODO: Add 'coarser_frequency' attribute in slice_flux function
        period_override,                    # TODO: Remove once 'coarser_frequency' attribute is implemented
        # Then, check for the original frequency
        ds.get('frequency', None),          # TODO: Add 'frequency' attribute in flux netCDF file
        species_info.get('period', {})  # TODO: Remove from species_info.json when no longer needed
    ]

    # Get the first valid frequency from the sources list
    frequency = next((freq for freq in frequency_sources if freq is not None), None)

    frequency = frequency_map.get(frequency, None)

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

    time = ds.time
    datetime_format = f'datetime64[{freq}]' if season is None else 'datetime64[Y]'

    start_date = time.values[0].astype(datetime_format)
    end_date = time.values[-1].astype(datetime_format)

    if start_date == end_date:
        period = f'{start_date}'
    else:
        period = f'{start_date}—{end_date}'

    if season:
        period = f'{season} of {period}'

    return period

def add_custom_markers(ax, markers, color):
    """Add custom markers to the plot."""
    for marker in markers:
        lon, lat = get_marker_coordinates(marker)
        ax.scatter(lon, lat, facecolor="none", edgecolor=color, marker="^", s=30, zorder=2)

def get_marker_coordinates(
    marker: str | tuple[float, float], 
    ) -> tuple[float, float]:
    """
    Retrieve latitude and longitude coordinates for a specified marker.

    Args:
        marker (str | tuple): 
            The location identifier. 
            - If a string, it must match a key in `config.point_source_dict`.
            - If a tuple, it must contain exactly two numeric values representing (latitude, longitude).

    Returns:
        tuple: 
            A tuple containing the latitude and longitude as `(lat_marker, lon_marker)`.
    """

    if isinstance(marker, str):
        if marker in config.point_source_dict:
            lon_marker, lat_marker = config.point_source_dict[marker]
        else:
            raise ValueError(f"Location '{marker}' not found in point source dictionary.")

    elif isinstance(marker, tuple) and len(marker) == 2:
        lon_marker, lat_marker = marker
        
    else:
        raise TypeError("Marker must be a string (location name) or a tuple of (lat, lon).")

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

def get_sites_coordinates(
    ds_all: dict[xr.Dataset],
    config_data: dict,
    ) -> dict:
    # TODO DODGY FUNCTION!!! Modify this function once 'sites' is included in all the attributes.
    """
    Collect the 'sites' attribute from a dictionary of xarray datasets. 
    If 'sites' is missing, use it from another dataset where it's available.

    Args:
        ds_all (dict): 
            Dictionary of xarray datasets.
        config_data (dict):
            Dictionary of sites with information for plotting (read from json file).

    Returns:
        dict: 
            A mapping of dataset keys to their respective 'sites' attribute.
    """

    sites_list = {}
    fallback_sites = None

    # First pass: Extract 'sites' where available
    for key, ds in ds_all.items():
        try:
            # Parse the 'sites' attribute if it exists
            if hasattr(ds, 'sites'):
                sites = eval(ds.sites)
                sites_list[key] = sites
                # Store fallback sites if not already set
                if fallback_sites is None:
                    fallback_sites = sites
            else:
                sites_list[key] = None
        except (ValueError, SyntaxError) as e:
            logger.warning(f"Could not parse 'sites' for {key}: {e}")
            sites_list[key] = None

    # Second pass: Fill in missing 'sites' using the fallback
    for key in sites_list.keys():
        if sites_list[key] is None and fallback_sites is not None:
            logger.warning(f"No 'sites' attribute in {key}, using fallback from another dataset.")
            sites_list[key] = fallback_sites

    sites_coordinates = {}
    for key in sites_list.keys():
        sites_coordinates[key] = extract_site_info(sites_list[key], config_data)

    return sites_coordinates

def extract_site_info(
    sites: list[str], 
    config_data: dict[str, dict]
    ) -> dict[str, dict]:
    """
    Extract latitude and longitude for each site from site_info in the config data.

    Args:
        sites (list[str]): A list of site names to extract information for.
        config_data (dict[str, dict]): A dictionary containing configuration data, 
                                       where 'site_info' holds the latitude and longitude info.
    
    Returns:
        site_data (dict[str, dict]): A dictionary mapping site names to their respective latitude and longitude.
    """

    site_info = config_data['site_info']

    site_data = {
        site: {
            'latitude': site_data_info[next(iter(site_data_info))]['latitude'],
            'longitude': site_data_info[next(iter(site_data_info))]['longitude']
        }
        for site, site_data_info in site_info.items() if site in sites
    }

    return site_data

def get_region_coordinates(
    region_name: str, 
    zoom_degree: float = 1,
    ) -> tuple[float, float, float, float]:
    """
    Get the bounding coordinates of a specified region with an option to zoom in/out.

    Args:
        region_name (str): 
            The name of the country or continent or region to get the coordinates for.
        zoom_degree (float): 
            The number of degrees to zoom in/out from the bounding box. Default is 1.

    Returns:
        region_coordinates (tuple): 
            The bounding coordinates of the region (min_lon, max_lon, min_lat, and max_lat), after zooming.
    """
    world = load_countries_shape()
    region_code = config.countrycodes_dict

    region_name_title = region_name.title()

    if region_name_title in world['NAME'].values:
        region = world[world['NAME'] == region_name_title].copy()
    elif region_name_title in world['CONTINENT'].values:
        region = world[world['CONTINENT'] == region_name_title].copy()
        # Exclude Russia from Europe
        if region_name_title == "Europe":
            region = region[region['NAME'] != 'Russia']
    elif region_name_title in world['SUBREGION'].values:
        region = world[world['SUBREGION'] == region_name_title].copy()
    else:
        region_name_upper = region_name.upper()

        if region_name_upper in world['ISO_A3'].values:
            region = world[world['ISO_A3'] == region_name_upper].copy()
        # If region_name_upper is in the region_code dictionary, get its corresponding ISO_A3 code(s)
        elif region_name_upper in region_code:
            iso_codes = region_code[region_name_upper]  # Could be a single ISO_A3 or a list
            # If it's a group (contains '-'), split into a list
            if '-' in iso_codes:
                iso_codes = iso_codes.split('-')
                region = world[world['ISO_A3'].isin(iso_codes)].copy()
            else:
                region = world[world['ISO_A3'] == iso_codes].copy()
        else:
            raise ValueError(f"Region '{region_name}' not found. Please ensure the region is correctly specified as a country, continent, subregion, or ISO_A3 code.\n"
                             f"You can also try using a predefined region from the list: {list(config.regions_dict.keys())}.\n"
                             "If you're still having trouble, check and update the config.countrycodes_dict if necessary.")

    # Handle empty region case
    if region.empty:
        raise ValueError(f"No coordinates found for region '{region_name}'.")

    # Remove overseas territories by keeping only the largest landmass for each country
    region['geometry'] = region['geometry'].apply(lambda geom: extract_largest_polygon(geom))

    # Get the bounding box of the region of interest
    region_boundaries = region.total_bounds  # [minx, miny, maxx, maxy]

    # Apply zoom adjustment
    min_lon = region_boundaries[0] - zoom_degree
    min_lat = region_boundaries[1] - zoom_degree
    max_lon = region_boundaries[2] + zoom_degree
    max_lat = region_boundaries[3] + zoom_degree

    region_coordinates = (min_lon, max_lon, min_lat, max_lat)
    return region_coordinates

def extract_largest_polygon(
    geometry: MultiPolygon | Polygon,
    ) -> Polygon:
    """
    Extract the largest polygon from a geometry object (e.g., MultiPolygon or Polygon).

    Args:
        geometry (MultiPolygon, Polygon): 
            Geometry object to process.

    Returns:
        Polygon: 
            Largest polygon from the geometry.
    """
    if isinstance(geometry, MultiPolygon):
        # Use the .geoms attribute to access the Polygons in the MultiPolygon
        return max(geometry.geoms, key=lambda p: p.area)
    elif isinstance(geometry, Polygon):
        # Single Polygon, return as is
        return geometry
    else:
        # Handle invalid or unexpected geometries
        return None

def set_flux_limits(
    ds_all: dict[xr.Dataset], 
    var: str, 
    region_plot: tuple[float, float, float, float], 
    species_info: dict, 
    option: Literal['default', 'auto'] | list[float] | tuple[float, float] = 'default', 
    custom_percentile: float = None,
) -> tuple[float, float]:
    """
    Set flux limits based on the option provided:
    
    1. `default` - use default values from species_info. #TODO Remove this option when restructuring the json file.
    2. List or tuple with two values (e.g., [0, 1]) - use specified limits.
    3. 'auto' - auto-calculate limits based on data percentiles.
    
    Args:
        ds_all (dict[xr.Dataset]): 
            A dictionary of datasets containing the flux variables.
        var (str): 
            The variable name to compute limits for.
        region_plot (tuple[float, float, float, float]): 
            Coordinates [lon_min, lon_max, lat_min, lat_max].
        species_info (dict): 
            Contains default limit values.
        option ('default', 'auto', [lower_lim, upper_lim]): 
            The option for setting limits.
                - `default` for default values.
                - A list or tuple with two elements (lower_lim, upper_lim) for specified limits.
                - 'auto' for auto-calculated values using the 99th percentile (if custom_percentile is None).
        custom_percentile (float, optional): 
            The percentile to use as the upper limit if option is 'auto'.
    
    Returns:
        flux_lim (tuple): 
            A tuple containing the flux limits (lower_lim, upper_lim).
    """
    
    # Case 1: Use default values from species_info
    if option == 'default':
        if var in ['posterior_prior_diff', 'posterior_mean_diff']:
            if 'difflim' in species_info:
                flux_lim = tuple(species_info['difflim'])
            else:
                raise KeyError("Key 'difflim' not found in species_info.")
        else:
            if 'fluxlim' in species_info:
                flux_lim = tuple(species_info['fluxlim'])
            else:
                raise KeyError("Key 'fluxlim' not found in species_info.")    

    # Case 2: Use specified values [lower_lim, upper_lim]          
    elif isinstance(option, (list, tuple)) and len(option) == 2:
        flux_lim = tuple(option)

    # Case 3: Auto-calculate limits based on percentiles    
    elif option == 'auto':
        models_var = []
        for model, ds in ds_all.items():
            if var == 'posterior_prior_diff':
                var_i = ds['flux_total_posterior'] - ds['flux_total_prior']
            elif var == 'posterior_mean_diff':
                var_i = ds['flux_total_posterior'] - ds['flux_total_posterior'].mean(dim='time')            
            else:
                var_i = ds[var]
                
            # Filter based on longitude and latitude of region_plot
            mask_region = ((ds.longitude > region_plot[0]) &
                           (ds.longitude < region_plot[1]) &
                           (ds.latitude > region_plot[2]) &
                           (ds.latitude < region_plot[3]))
            var_i = var_i.where(mask_region, drop=True)
            models_var.append(var_i)

        models_var = xr.concat(models_var, dim='time')   

        # Calculate upper limit based on custom_percentile or default to 99th percentile
        upper_lim = models_var.quantile(custom_percentile if custom_percentile else 0.99).item()
        
        if var in ['posterior_prior_diff', 'posterior_mean_diff']:
            flux_lim = (-upper_lim, upper_lim)
        else:
            flux_lim = (0, upper_lim)
            
    else:
        raise ValueError(f"Invalid option '{option}'. Use 'default', a (lower_lim, upper_lim) tuple, or 'auto'.")
        
    # Validation: Check that flux_lim[0] is smaller than flux_lim[1]
    if flux_lim[0] >= flux_lim[1]:
        raise ValueError(f"The lower flux limit {flux_lim[0]} must be less than the upper flux limit {flux_lim[1]}.") 
        
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
