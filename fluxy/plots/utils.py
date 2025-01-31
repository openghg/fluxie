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

def update_list_params(params_to_check: list | None,
                       expected_size: int
                       )-> list:
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
            updated_params.append([False]*expected_size)
        elif type(param) is list :
            if len(param) == expected_size:
                updated_params.append(param)
            else:
                raise ValueError(f'{param} must be a boolean or a list of booleans of the same length as models.')        
        else:
            updated_params.append([param]*expected_size)
    return updated_params


def set_flux_limits(ds_all, var, region_plot, species_info, option='default', custom_percentile=None):
    """
    Set flux limits based on the option provided:
    
    1. `default` - use default values from species_info.
    2. List or tuple with two values (e.g., [0, 1]) - use specified limits.
    3. 'auto' - auto-calculate limits based on data percentiles.
    
    Args:
        ds_all (dict): A dictionary of datasets containing the flux variables.
        var (str): The variable name to compute limits for.
        region_plot (list/tuple): Coordinates [lon_min, lon_max, lat_min, lat_max].
        species_info (dict): Contains default limit values.
        option (None, str or list/tuple): The option for setting limits.
            - `default` for default values.
            - A list or tuple with two elements (min, max) for specified limits.
            - 'auto' for auto-calculated values using the 99th percentile (if custom_percentile is None).
        custom_percentile (float, optional): The percentile to use as the upper limit if option is 'auto'.
    
    Returns:
        tuple: A tuple containing the flux limits (min, max).
        
    Raises:
        ValueError: If `fluxlim[0] >= fluxlim[1]` or an invalid option is provided.
    """
   
    if option == 'default':  # Case 1: Use default values from species_info
        if var in ['posterior_prior_diff', 'posterior_mean_diff']:
            if 'difflim' in species_info:
                fluxlim = tuple(species_info['difflim'])
            else:
                raise KeyError("Key 'difflim' not found in species_info.")
        else:
            if 'fluxlim' in species_info:
                fluxlim = tuple(species_info['fluxlim'])
            else:
                raise KeyError("Key 'fluxlim' not found in species_info.")    
                
    elif isinstance(option, (list, tuple)) and len(option) == 2:  # Case 2: Use specified values [min, max]
        fluxlim = (option[0], option[1])
        
    elif option == 'auto':  # Case 3: Auto-calculate limits based on percentiles
        all_var = []
        for j, m in enumerate(ds_all.keys()):
            if var == 'posterior_prior_diff':
                var_j = ds_all[m]['flux_total_posterior'] - ds_all[m]['flux_total_prior']
            elif var == 'posterior_mean_diff':
                var_j = ds_all[m]['flux_total_posterior'] - np.mean(ds_all[m]['flux_total_posterior'], axis=0)            
            else:
                var_j = ds_all[m][var]
                
            # Filter based on longitude and latitude of region_plot
            mask_region = ((ds_all[m].longitude > region_plot[0]) &
                           (ds_all[m].longitude < region_plot[1]) &
                           (ds_all[m].latitude > region_plot[2]) &
                           (ds_all[m].latitude < region_plot[3]))
            var_j = var_j.where(mask_region).dropna(dim='longitude', how='all').dropna(dim='latitude', how='all')
            all_var.append(var_j.values)
            
        all_var = np.concatenate(all_var, axis=0)  # Concatenate along the time axis (axis=0)
        
        if custom_percentile is None:
            upper_lim = np.quantile(all_var, .99) # Calculate the 99th percentile
        else:
            upper_lim = np.quantile(all_var, custom_percentile)
        
        if var in ['posterior_prior_diff', 'posterior_mean_diff']:
            fluxlim = (-upper_lim, upper_lim)
        else:
            fluxlim = (0, upper_lim)
            
    else:
        # Raise an error if the option is invalid
        raise ValueError(f"Invalid option '{option}'. Use 'default', a [min, max] list/tuple, or 'auto'.")
        
    # Validation: Check that fluxlim[0] is smaller than fluxlim[1]
    if fluxlim[0] >= fluxlim[1]:
        raise ValueError(f"The lower flux limit {fluxlim[0]} must be less than the upper flux limit {fluxlim[1]}.") 
        
    return fluxlim
