import xarray as xr
import numpy as np
import os
from pathlib import Path
import logging
from typing import Literal
import re
from collections import Counter
from fluxy import config

logger = logging.getLogger(__name__)

def scale_mf(
        model: str,
        ds_model: xr.Dataset,
        mf_target_units: str,
) -> xr.Dataset:
    """
    Scales mole fractions according to print units defined in species_info.json.

    Args:
        model (str):
            Name tag of the model being scaled.
        ds_model (xarray dataset):
            Sliced dataset with mf data from model. 
        mf_target_units (dictionary of str):
            Units to which mole fractions should be converted to.
    Returns:
        ds_model (xarray dataset):
            Sliced and scaled dataset.
    """

    # Get list of variables with mf units
    var_names, var_unit = get_variables(ds_model,unit_type='mf')

    if not var_names:
        raise ValueError(f'There are no variables in {model} with mole fraction units in the attributes. Scaling to {mf_target_units} cannot be applied.')

    if var_unit is None:
        raise ValueError(f'{model} dataset considers different mole fraction units. Uniform scaling to {mf_target_units} cannot be applied.')

    # Get scaling factor
    scaling_factor = get_units_conversion_factor(var_unit,mf_target_units)

    # Scale mole fractions
    for v in var_names:
        ds_model[v] = ds_model[v]*scaling_factor
        ds_model[v].attrs['units'] = mf_target_units
    
    logger.info(f'Scaling {model} mole fractions by {scaling_factor}.')

    return ds_model

def scale_country_flux(
        model: str,
        ds_model: xr.Dataset,
        specie_info: dict[str, str|float],
        country_flux_units_print: str,
) -> xr.Dataset:
    """
    Scales country fluxes according to print units defined in species_info.json.

    Args:
        model (str):
            Name tag of the model being scaled.
        ds_model (xarray dataset):
            Sliced dataset with country fluxes data from model.
        specie_info (dictionary of str):
            Dictionary with species-specific settings.
        country_flux_units_print (str):
            Units to which country fluxes should be converted to.
    Returns:
        ds_model (xarray dataset):
            Sliced and scaled dataset.
    """

    # Get list of variables with country flux unit
    #NOTE: country fluxes must be in mass units so that conversion to CO2-eq works
    var_names, var_unit = get_variables(ds_model,unit_type='mass1 time-1')

    if not var_names:
        raise ValueError(f'There are no variables in {model} with country flux units in the attributes. Scaling to {country_flux_units_print} cannot be applied.')

    if var_unit is None:
        raise ValueError(f'{model} dataset considers different country flux units. Uniform scaling to {country_flux_units_print} cannot be applied.')
    
    # Get scaling factor
    gwp = 1
    target_unit = country_flux_units_print
    if "CO2-eq" in target_unit:
        gwp = specie_info["gwp"]
        target_unit = country_flux_units_print.replace("CO2-eq","")
        logger.info(f'Converting to mass of CO2-eq using GWP = {gwp}.')

    scaling_factor = get_units_conversion_factor(var_unit,target_unit)

    # Scale country flux
    for v in var_names:
        ds_model[v] = ds_model[v] * scaling_factor * gwp
        ds_model[v].attrs['units'] = country_flux_units_print

    logger.info(f'Scaling {model} country fluxes by {scaling_factor*gwp}.')

    # Scale covariance matrix
    cov_var = 'covariance_country_flux_total_posterior'
    if cov_var in ds_model.keys():
        ds_model[cov_var] = ds_model[cov_var] * scaling_factor**2 * gwp**2
        logger.info(f'Scaling covariance in {model} by {scaling_factor**2 * gwp**2}')

    return ds_model

def scale_flux(
        model: str,
        ds_model: xr.Dataset,
        specie_info: dict[str, str|float],
        flux_units_print: str,
) -> xr.Dataset:
    """
    Scales fluxes according to print units defined in species_info.json.

    Args:
        model (str):
            Name tag of the model being scaled.
        ds_model (xarray dataset):
            Sliced dataset with flux data from model. 
        specie_info (dictionary of str):
            Dictionary with species-specific settings.
        flux_units_print (str):
            Units to which fluxes should be converted to.
    Returns:
        ds_model (xarray dataset):
            Sliced and scaled dataset.
    """

    # Get list of variables with flux units
    unit_type = 'amount1 length-2 time-1'
    var_names, var_unit = get_variables(ds_model,unit_type)

    if not var_names:
        # Alternatively look for flux variables with mass units
        unit_type = 'mass1 length-2 time-1'
        var_names, var_unit = get_variables(ds_model,unit_type)

    if not var_names:
        raise ValueError(f'There are no variables in {model} with flux units in the attributes. Scaling to {flux_units_print} cannot be applied.')

    if var_unit is None:
        raise ValueError(f'{model} dataset considers different flux units. Uniform scaling to {flux_units_print} cannot be applied.')
    
    # Get scaling factor
    scaling_factor = get_units_conversion_factor(var_unit, flux_units_print, specie_info["molar_mass"])

    # Scale mole fractions
    for v in var_names:
        ds_model[v] = ds_model[v]*scaling_factor
        ds_model[v].attrs['units'] = flux_units_print

    logger.info(f'Scaling {model} fluxes by {scaling_factor}.')

    return ds_model

def slice_flux(
        ds_all: dict[str, xr.Dataset],
        config_data: dict[str, str | float],
        start_date: str = None,
        end_date: str = None,
        specie: str = None,
        flux_units_print: str = None,
        country_flux_units_print: str = None,
) -> dict[str, xr.Dataset]:
    """
    Slices the flux datasets to within given time limits and 
    calls scaling functions.
    
    Args:
        ds_all (dictionary of datasets): 
            xarray datasets read directly from each model's flux netCDF.
        start_date (str): 
            Date to slice data from, e.g. '2021-01-01'
        end_date (str): 
            Date to slice data to, e.g. '2022-01-01' would include all
            data up to 2021-12-31.
        config_data (dict of dict):
            Dictionary with settings read from json file.
            Use json filename as keys.
        specie (str):
            Gas species, used to choose scaling units, e.g. 'ch4'.
            Set to None to prevent scaling.
        flux_units_print (str):
            Units to which fluxes should be converted to.
            Expected format: "<letters><(-)integer>" separated by spaces (e.g. "mol m-2 s-1").
        country_flux_units_print (str):
            Units to which country fluxes should be converted to.
            Expected format: "<letters><(-)integer>" separated by spaces (e.g. "Tg yr-1").
            Conversion to CO2 equivalent can be specified with tag "CO2-eq" (e.g. "Tg CO2-eq yr-1").
    Returns:
        ds_all (dictionary of datasets):
            xarray datasets, scaled, converted, and sliced between chosen dates.
    
    """
    
    if specie is not None:
        specie_info = config_data['species_info'][specie]   

    for m in ds_all.keys():
        logger.info(f'Masking data from {m}.')

        # Slice data according to time window
        ds_all[m] = ds_all[m].sel(time=slice(start_date,end_date))

        if len(ds_all[m]['time']) == 0:
            logger.warning(f'No {m} fluxes found between {start_date} and {end_date}.')
            ds_all.pop(m)
            continue

        # Scale fluxes
        if specie is not None:
            if flux_units_print is not None:
                ds_all[m] = scale_flux(m,ds_all[m],specie_info,flux_units_print)
        
            if country_flux_units_print is not None:
                ds_all[m] = scale_country_flux(m,ds_all[m],specie_info,country_flux_units_print)
        
    return ds_all

def slice_mf(
        ds_all: dict[str, xr.Dataset],
        start_date: str = None,
        end_date: str = None,
        site: str = None,
        baseline_site: str = None,
        data_dir: os.PathLike = None,
        mf_units_print: str = None,
) -> dict[str, xr.Dataset]:
    """
    Slices down the mole fraction timeseries data, to within the
    given time limits, and/or for the chosen site.
    
    Args:
        ds_all (dictionary of datasets): 
            xarray datasets read directly from each model's flux netCDF.
        start_date (str): 
            Date to slice data from, e.g. '2021-01-01'
        end_date (str): 
            Date to slice data to, e.g. '2022-01-01' would include all
            data up to 2021-12-31.
        site (str):
            Obs site to select data from, e.g. 'MHD'.
        baseline_site (str):
            Site used to define baseline at, options for 'MHD', 'JFJ', or 'CMN'.
            If None, does not mask timeseries by baseline times.
        data_dir (str): 
            Path to top data directory, used to read baseline info files.
        mf_units_print (str):
            Units to which mole fractions should be converted to.
            Expected format: "<letters><(-)integer>" separated by spaces (e.g. "mol mol-1")
    Returns:
        ds_all (dictionary of datasets):
            xarray datasets, scaled and sliced between chosen dates and for 
            chosen site.
    """

    data_dir = Path(data_dir)
    models = list(ds_all.keys())

    # Get logical array with baseline timestamps 
    if baseline_site is not None:
        baseline_file = data_dir / 'intem_baseline_timestamps' / f'{baseline_site}_InTEM_baseline_timestamps.nc'

        # Check if files exists
        if not baseline_file.is_file():
            raise FileNotFoundError(f'Cannot find baseline file for masking: {baseline_file}.')

        # Read baseline file
        with xr.open_dataset(baseline_file) as f:
            baseline = f.sel(time=slice(start_date,end_date))
    
    for m in models:
        logger.info(f'Masking data from {m}.')
        
        # Compute offset
        if 'Yav' in ds_all[m].keys():
            offset = int(np.mean(ds_all[m]['Yav']))
        else:
            offset = (ds_all[m].time.values[1].astype('datetime64[h]') - ds_all[m].time.values[0].astype('datetime64[h]')).astype(int)

        # Round time to seconds (for consistency between models)
        ds_all[m]['time'] = ds_all[m]['time'].dt.round('s')

        # Slice data according to site and time window
        if site is not None:
            site_index = get_site_index(ds_all[m], site)

            if site_index is not None:
                ds_all[m] = ds_all[m].sel(time=slice(start_date,end_date),
                                          nsite=site_index)

                if len(ds_all[m]['time']) == 0:
                    logger.warning(f'No {m} obs found for {site} between {start_date} and {end_date}.')
                    ds_all.pop(m)
                    continue

            else:
                logger.warning(f'No {m} obs found for {site}.')
                ds_all.pop(m)
                continue
        else:
            ds_all[m] = ds_all[m].sel(time=slice(start_date,end_date))

            if len(ds_all[m]['time']) == 0:
                logger.warning(f'No {m} obs found between {start_date} and {end_date}.')
                ds_all.pop(m)
                continue

        # Scale mole fractions
        if mf_units_print is not None:
            ds_all[m] = scale_mf(m, ds_all[m], mf_units_print)

        # Mask mole fractions according to baseline timestamps
        if baseline_site is not None:
            logger.info('Masking timeseries to only include baseline times.')
                                
            #average baseline mask over obs averaging period
            b = baseline.resample(time=f'{offset}h').mean()
            #adjust baseline mask time back to centre of av period (resample removes this)
            b['time'] = b['time'] + np.timedelta64(offset,'h')/2
                                
            #mask baseline mask again, to only include timestamps where every period in the averaging period is classified as baseline
            b_masked = b.sel(time=b['time'][np.where(b['baseline'] == 1.)])
                            
            #mask dataset using only baseline times
            both_times = np.isin(ds_all[m].time,b_masked.time)          
            ds_all[m] = ds_all[m].sel(time=both_times)
                   
    return ds_all

def get_site_index(ds: xr.Dataset, site: str) -> int | None:
    """
    Gets the index of a given site in a dataset.

    Args:
        ds (xarray dataset):
            Dataset with mf data of a given model.
        site (str):
            Site of interest.
    Returns:
        index (int):
            Index of site of interest in the dataset.
            Returns None if site does not exist.
    """

    # Get all sites
    sites = ds['sitenames'].astype(str)

    # Get site index
    if site in sites:
        index = np.where(ds['sitenames'].astype(str) == site)[0][0]
        return index
    
    return None

def get_unique_sites(ds_all: dict[str, xr.Dataset]) -> list[str]:
    """
    Gets list of all sites present in all datasets.

    Args:
        ds (xarray dataset):
            Dictionary of datasets with mf data from all models.
    Returns:
        sites (list of str):
            List of unique and sorted sites from all datasets.
    """

    sites = []
    for ds in ds_all.values():
        sites.append(ds['sitenames'].astype(str))

    sites = np.sort(np.unique(sites))

    return sites

def get_variables(ds_model: xr.Dataset, unit_type: str) -> tuple[list[str], str | None]:
    """
    Finds variables of a given type in a dataset.

    Args:
        ds_model (xarray dataset):
            Dataset with data from a model.
        unit_type (str):
            Unit type of interest (e.g. "mf", "mass1 length-2 time-1")
    Returns:
        var_names (list of str):
            Names of variables in the dataset with unit_type units.
        unique_units (str | None):
            unit_type explicit units used in the dataset (e.g. "mol mol-1", "kg m-2 s-1")
            Set to None if different units are present.
    """

    var_names = []
    var_units = []

    # Find variables of type var_type in dataset
    for var, var_data in ds_model.items():
        if 'units' in var_data.attrs.keys():
            unit = var_data.attrs['units']

            # Particular case of mole fractions:
            if unit_type == "mf":
                if unit in config.units_scale["mf"].keys():
                    if var == 'sitenames':
                        # Correction for InTEM (units are wrongly set to mol mol-1)
                        continue

                    var_names.append(var)
                    var_units.append(unit)

            else:
                # Get unit type
                x, unit_dims = get_unit_type_and_conversion_to_base(unit)

                # Append var if unit is of the desiared var_type
                if Counter(unit_dims) == Counter(unit_type.split()):
                    var_names.append(var)
                    var_units.append(unit)

    # Get var units if unique
    unique_units = list(set(var_units))
    if len(unique_units) == 1:
        unique_units = unique_units[0]
    else:
        unique_units = None

    return var_names, unique_units

def get_unit_type_and_conversion_to_base(input_unit: str) -> tuple[float,list[str]]:
    """
    Computes conversion factor to base unit.
    Units are expected to have the following format:
        "<letters><(-)integer>" separated by spaces (e.g. "kg m-2 s-1")

    Args:
        input_unit (str):
            Units of a given variable.
    Returns:
        conversion_factor (float):
            Scaling factor to base unit.
        unit_dim_type (list of str):
            Unit type of input_unit (e.g. ["mass1", "length-2", "time-1"])
    """

    # Get individual units
    units_list = input_unit.split()

    # Initialize list of dimensions and exponents
    unit_dim = []
    unit_exponent = [1]*len(units_list)

    # Get unit dimension and exponent
    for i,unit_exp in enumerate(units_list):
        unit_elements = list(re.findall(r'[a-zA-Z]+|[-+]?\d+', unit_exp))
        unit_dim.append(unit_elements[0])
        if len(unit_elements) > 1:
            unit_exponent[i] = float(unit_elements[1])

    conversion_factor = 1
    unit_dim_type = []

    # Get unit family and conversion factor to base units
    for i,unit in enumerate(unit_dim):
        factor_to_base = None
        for unit_family, units in config.units_scale.items():
            if unit in units:
                factor_to_base = units[unit]**unit_exponent[i]
                unit_dim_type.append(unit_family+f"{unit_exponent[i]:.0f}")
                break
        
        if factor_to_base is None:
            raise KeyError(f'Unit {unit} does not exist in units_scale dictionary.')
        
        conversion_factor = conversion_factor * factor_to_base

    return conversion_factor, unit_dim_type

def get_units_conversion_factor(from_unit: str, to_unit: str, molar_mass: float | None = None) -> float:
    """
    Computes conversion factor between two units.
    Units are expected to have the following format:
        "<letters><(-)integer>" separated by spaces (e.g. "kg m-2 s-1")

    Consistency between base and target units are verified.
    Non simplified units (e.g. "kg m-1 m-1 s-1") are assumed different from their simplified form
    and should be avoided.

    Conversion from mass units (g) to amount of substance (mol) is done via molar_mass.
    The current implementation only applies the conversion if the units exponent is equal to 1.

    Args:
        from_unit (str):
            Units of the variable to be converted.
        to_unit (str):
            Target units.
        molar_mass (float):
            Molar mass (g mol-1) to be used in g<->mol conversion.
    Returns:
        conversion_factor (float):
            Scaling factor that guarantees the requested units conversion.
    """

    # Deal with particular case of mol mol-1
    if from_unit == "mol mol-1" or to_unit == "mol mol-1":

        unit_to_base = config.units_scale['mf'].get(from_unit, None)
        if unit_to_base is None:
            raise KeyError(f'Conversion factor to/from {from_unit} does not exist in units_scale dictionary.')
        
        target_to_base = config.units_scale['mf'].get(to_unit, None)
        if target_to_base is None:
            raise KeyError(f'Conversion factor to/from {to_unit} does not exist in units_scale dictionary.')
        
        return unit_to_base / target_to_base

    # General case
    unit_to_base, unit_dim_type = get_unit_type_and_conversion_to_base(from_unit)
    target_to_base, target_dim_type = get_unit_type_and_conversion_to_base(to_unit)

    # Get g-mol conversion factor if needed
    M_scaling = 1
    if molar_mass is not None:
        non_common_units = set(unit_dim_type) ^ set(target_dim_type)

        if non_common_units == {'mass1','amount1'}:
            # NOTE: conversion is only applied if exponent == 1
            if 'amount1' in unit_dim_type:
                M_scaling = molar_mass # g mol-1
                # Update unit type so that it passes consistency check
                index = unit_dim_type.index('amount1')
                unit_dim_type[index] = 'mass1'
            else:
                M_scaling = 1/molar_mass # mol g-1
                # Update unit type so that it passes consistency check
                index = unit_dim_type.index('mass1')
                unit_dim_type[index] = 'amount1'
            #NOTE: only one unit element of type amount/mass is expected and replaced

            logger.info(f"Converting g<->mol using M = {molar_mass} g mol-1")

    # Check for consistency
    if Counter(unit_dim_type) != Counter(target_dim_type):
        raise ValueError(f'Units {from_unit} ({unit_dim_type}) and {to_unit} ({target_dim_type}) are not consistent.')

    return unit_to_base / target_to_base * M_scaling
