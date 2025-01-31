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

def convert_molar_to_mass_flux(ds, M):
    """
    Converts spatial flux variables in the flux dataset from mol/m²/s to kg/km²/year.
    
    Args:
        ds (xarray dataset): The xarray dataset containing flux variables.
        M (float): The molar mass in g/mol.
    """
    
    # Convert the molar mass to kg/mol
    M_kg = M * 0.001  # Convert grams to kilograms
    
    # List to store names of converted variables
    converted_vars = []
    
    for var_name, variable in ds.items():
        if 'units' in variable.attrs and variable.attrs['units'] == 'mol m-2 s-1':
            target_units = "kg km-2 yr-1" # TODO: create variable flux_units_print in json file
            
            # Apply scaling and adjust units
            ds[var_name] = variable * M_kg
            ds[var_name].attrs['units'] = 'kg m-2 s-1' # Update units

            scaling_factor = get_unit_conversion_factor(ds[var_name].attrs['units'], target_units)
            ds[var_name] = ds[var_name] * scaling_factor 
            ds[var_name].attrs['units'] = target_units # Update units

            # Add variable name to the list of converted variables
            converted_vars.append(var_name)

    if converted_vars:
        logger.info(f"Converting molar flux variables to mass flux using M = {M_kg} kg mol-1")
        logger.info(f"Scaling mass flux variables by {scaling_factor}.")
    
    return ds

def scale_mf(
        model: str,
        ds_model: xr.Dataset,
        specie_info: dict[str]
) -> xr.Dataset:
    """
    Scales mole fractions according to pre-defined scaling factor.

    Args:
        model (str):
            Name tag of the model being scaled.
        ds_model (xarray dataset):
            Sliced dataset with mf data from model. 
        specie_info (dictionary of str):
            Dictionary with species-specific settings.
    Returns:
        ds_model (xarray dataset):
            Sliced and scaled dataset.
    """

    # Get list of variables with mf units
    var_names, var_unit = get_mf_variables(ds_model)

    if len(var_names) == 0:
        raise ValueError(f'There are no variables in {model} with mole fraction units in the attributes. Scaling to {specie_info["mf_units_print"]} cannot be applied.')

    if var_unit is None:
        raise ValueError(f'{model} dataset considers different mole fraction units. Uniform scaling to {specie_info["mf_units_print"]} cannot be applied.')

    # Get scaling factor
    scaling_factor = get_unit_conversion_factor(var_unit,specie_info["mf_units_print"])

    # Scale mole fractions
    for v in var_names:
        ds_model[v] = ds_model[v]*scaling_factor
        ds_model[v].attrs['units'] = specie_info["mf_units_print"]
    
    logger.info(f'Scaling {model} mole fractions by {scaling_factor}.')

    return ds_model

def slice_flux(ds_all,start_date,end_date,config_data,
               scale_units=True,scale_co2eq=False,convert_flux_units=False,specie=None):
    """
    Slices the flux datasets to within given time limits and 
    scales fluxes into Tg/Gg based on the species.
    
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
        scale_units (bool): 
            If True, scales country fluxes to Tg or Gy per year.
        scale_co2eq (bool):
            If True, converts country fluxes to CO2-eq in Tg per year.
        convert_flux_units (bool): 
            If True, performs the conversion of molar flux to mass flux (default is False).
        specie (str):
            Gas species, used to choose scaling units, e.g. 'ch4'.
    Returns:
        ds_all (dictionary of datasets):
            xarray datasets, scaled, converted, and sliced between chosen dates.
    
    """
    
    if specie is not None:
        specie_info = config_data['species_info'][specie] 

    #variables that aren't scaled by units
    skip_var = ['flux_total_prior','flux_total_posterior','percentile_flux_total_prior',
                'percentile_flux_total_posterior','countryname','country',
                'country_fraction','outer_region_fraction',
                'covariance_country_flux_total_posterior','flux_total_posterior_inversion_grid']

    for m in ds_all.keys():
        
        m0 = m.split('_')[0]
        
        print(f'\nMasking data from {m}')
        try:
            ds_all[m] = ds_all[m].sel(time=slice(start_date,end_date))
        except:
            ds_all[m] = None
            print(f'No {m} fluxes found between {start_date} and {end_date}')
            print(f'Skipping {m}')
            
        if scale_units == True:
            gwp = 1
            scale_factor = specie_info["units_scaling"][m0]

            # Update scaling factors
            if (scale_co2eq):
                gwp = specie_info["gwp"]
                if (specie_info["units_print"] == "G"): #units_print is expected to be either G or T
                    scale_factor = scale_factor * 1e3 #Convert to Tg
                    # Note: units_print is not re-written because it would go back to
                    #       its original value if initialize_settings is re-run.

            print(f'Scaling {m} country fluxes by {scale_factor*gwp}')
            if ds_all[m] is not None:
                var_names = [k for k in ds_all[m].keys() if k not in skip_var]
                for v in var_names:
                    ds_all[m][v].values = ds_all[m][v].values/scale_factor * gwp

                cov_var = 'covariance_country_flux_total_posterior'
                if cov_var in ds_all[m].keys():
                    ds_all[m][cov_var].values = ds_all[m][cov_var].values/scale_factor**2 * gwp**2
                    print(f'Scaling covariance in {m} by {scale_factor**2 * gwp**2}')
                    
        if convert_flux_units:
            M = specie_info["molar_mass"]
            ds_all[m] = convert_molar_to_mass_flux(ds_all[m], M)
        
    return ds_all


def slice_mf(
        ds_all: dict[str, xr.Dataset],
        config_data: dict[str, dict],
        start_date: str = None,
        end_date: str = None,
        site: str = None,
        baseline_site: str = None,
        data_dir: os.PathLike = None,
        scale_units: bool = False,
        specie: str = None
) -> dict[str, xr.Dataset]:
    """
    Slices down the mole fraction timeseries data, to within the
    given time limits, and/or for the chosen site.
    
    Args:
        ds_all (dictionary of datasets): 
            xarray datasets read directly from each model's flux netCDF.
        config_data (dict of dict):
            Dictionary with settings read from json file.
            Use json filenames as keys.
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
        scale_units (bool): 
            If True, scales country fluxes to Tg or Gy per year.
        specie (str):
            Gas specie, used to choose scaling units, e.g. 'ch4'.
    Returns:
        ds_all (dictionary of datasets):
            xarray datasets, scaled and sliced between chosen dates and for 
            chosen site.
    """

    data_dir = Path(data_dir)
    models = list(ds_all.keys())
    specie_info = config_data['species_info'][specie]   

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
        # Correction for Flexinvert (no units attribute)
        if scale_units == True and 'flexinvert' not in m:
            ds_all[m] = scale_mf(m, ds_all[m], specie_info)

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

def get_mf_variables(ds_model: xr.Dataset) -> tuple[list[str], str | None]:
    """
    Finds mole fraction variables in a dataset.

    Args:
        ds_model (xarray dataset):
            Dataset with mf data from a model.
    Returns:
        var_names (list of str):
            Names of variables in the dataset with mole fraction units.
        unique_units (str | None):
            Mole fraction units used in the dataset.
            Set to None if different units are present.
    """

    var_names = []
    var_units = []

    # Find mole fraction variables in dataset
    for var in ds_model.keys():
        if 'units' in ds_model[var].attrs.keys():
            unit = ds_model[var].attrs['units']
            if unit in config.units_scale['mf'].keys():
                if var == 'sitenames':
                    # Correction for InTEM (units are wrongly set to mol mol-1)
                    continue

                var_names.append(var)
                var_units.append(unit)

    # Get mf units if unique
    unique_units = list(set(var_units))
    if len(unique_units) == 1:
        unique_units = unique_units[0]
    else:
        unique_units = None

    return var_names, unique_units

def get_unit_conversion_factor(from_unit: str, to_unit: str) -> float:
    """
    Computes conversion factors of compound units.
    Units are expected to have the following format:
        "<letters><(-)numbers>" separated by spaces (e.g. "kg m-2 s-1")

    Consistency between base and target units are verified.
    Consistency between the exponents of the base and target units are verified until the 2nd decimal point.

    Args:
        from_unit (str):
            Units of the variable to be converted.
        to_unit (str):
            Target units.
    Returns:
        conversion_factor (float):
            Scaling factor that guarantees the requested units convertion.
    """

    # Deal with particular case of mol mol-1
    if from_unit == "mol mol-1" or to_unit == "mol mol-1":

        factor_to_base = config.units_scale['mf'].get(from_unit, None)
        if factor_to_base is None:
            raise KeyError(f'Conversion factor to/from {from_unit} does not exist in units_scale dictionary.')
        
        factor_to_target = config.units_scale['mf'].get(to_unit, None)
        if factor_to_target is None:
            raise KeyError(f'Conversion factor to/from {to_unit} does not exist in units_scale dictionary.')
        
        return factor_to_base / factor_to_target

    # General case
    # Get individual units
    units_list = from_unit.split(' ')
    target_list = to_unit.split(' ')

    # Initialize list of dimensions and exponents
    unit_dim = []
    unit_exponent = [1]*len(units_list)
    target_dim = []
    target_exponent = [1]*len(target_list)

    # Get unit dimension and exponent (original units)
    for i,unit_exp in enumerate(units_list):
        unit_elements = list(re.findall(r'[a-zA-Z]+|[-+]?\d*\.?\d+', unit_exp))
        unit_dim.append(unit_elements[0])
        if len(unit_elements) > 1:
            unit_exponent[i] = float(unit_elements[1])

    # Get unit dimension and exponent (target units)
    for i,unit_exp in enumerate(target_list):
        target_elements = list(re.findall(r'[a-zA-Z]+|[-+]?\d*\.?\d+', unit_exp))
        target_dim.append(target_elements[0])
        if len(target_elements) > 1:
            target_exponent[i] = float(target_elements[1])

    conversion_factor = 1
    unit_dim_type = []
    target_dim_type = []

    # Get conversion factor to base units
    for i,unit in enumerate(unit_dim):
        factor_to_base = None
        for unit_family, units in config.units_scale.items():
            if unit in units:
                factor_to_base = units[unit]**unit_exponent[i]
                unit_dim_type.append(unit_family+f"{unit_exponent[i]:.2f}")
                break
        
        if factor_to_base is None:
            raise KeyError(f'Unit {unit} does not exist in units_scale dictionary.')
        
        conversion_factor = conversion_factor * factor_to_base

    # Get conversion factor to target units
    for i,target in enumerate(target_dim):
        factor_to_target = None
        for target_family, units in config.units_scale.items():
            if target in units:
                factor_to_target = units[target]**target_exponent[i]
                target_dim_type.append(target_family+f"{target_exponent[i]:.2f}")
                break
        
        if factor_to_target is None:
            raise KeyError(f'Unit {target} does not exist in units_scale dictionary.')
        
        conversion_factor = conversion_factor / factor_to_target

    if Counter(unit_dim_type) != Counter(target_dim_type):
        raise ValueError(f'Units {from_unit} ({unit_dim_type}) and {to_unit} ({target_dim_type}) are not consistent.')

    return conversion_factor
