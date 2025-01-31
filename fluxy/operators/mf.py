import numpy as np
import xarray as xr
import logging
from fluxy.operators.select import get_unique_sites, get_site_index, get_mf_variables

logger = logging.getLogger(__name__)

def compute_mf_difference(ds_all: dict[str, xr.Dataset],
                         models_to_subtract: list[str]
) -> dict[str, xr.Dataset]:
    """
    Calculates the difference between two datasets with mole fraction data.
    Only the timestamps/sites common to both datasets are used.
    Only mole fraction variables are subtracted and included in the return
    dataset.

    Args:
        ds_all (dictionary of datasets):
            xarray datasets from all models, sliced between chosen dates.
        models_to_subtract (list of str):
            List with two elements which correspond to the names of the models
            to subtract.
    Returns:
        ds_diff (dictionary of dataset):
            Dictionary with one single key pointing to an xarray dataset.
            Key is given by the two elements in models_to_subtract separated
            by a minus sign (-).
    """

    models = list(ds_all.keys())
        
    if len(models_to_subtract) != 2:
        raise ValueError('List of models to subtract must be of size 2.')
    
    for m in models_to_subtract:
        if m not in models:
            raise KeyError(f'{m} not found in the dataset.')
     
    # Reduce datasets to timestamps/sites common to both models
    ds0, ds1 = xr.align(ds_all[models_to_subtract[0]], ds_all[models_to_subtract[1]], join="inner")

    ds_diff = {}
    key_name = f'{models_to_subtract[0]}-{models_to_subtract[1]}'
    ds_diff[key_name] = ds0

    # Compute difference between the two datasets (mole fraction variables only)
    var_names, x = get_mf_variables(ds0)
    for v in var_names:
        if ds0[v].attrs['units'] != ds1[v].attrs['units']:
            logger.warning(f'{v} in {models_to_subtract[0]} and {models_to_subtract[1]} have different units. {v} will not be included in the diff dataset.')
            continue
        
        ds_diff[key_name][v] = ds0[v] - ds1[v]

    return ds_diff

def stats_mf(ds_all: dict[str,dict]) -> dict[str,dict]:
    """
    Calculates multiple statistical measures of the fit between the posterior
    mean mf and the observed mole fraction.
    Implemented statistics: Pearson correlation coefficent, root mean square
    error, normalised root mean square error, standard deviation.
    
    Args:
        ds_all (dictionary of datasets):
            xarray datasets from slice_mf(), sliced between chosen dates
            but still containing all sites.
    Returns:
        stats (dictionary of dictionaries):
            Statistical measures, for each site and for each model.
    """
    
    sites_all = get_unique_sites(ds_all)
    
    # Implemented statistics
    stat_vars = ['pearson','rmse','nrmse','std']
    
    stats = {stat: {site: {} for site in sites_all} for stat in stat_vars}

    # Compute stats for all sites and all models
    for site in sites_all:
        for model, ds in ds_all.items():
            site_index = get_site_index(ds, site)
            if (site_index is not None) and (ds['Yobs'].isel(nsite=site_index).count() != 0):
                Yobs = ds['Yobs'].isel(nsite=site_index).dropna(dim='time')
                Yapost = ds['Yapost'].isel(nsite=site_index).dropna(dim='time')

                stats['pearson'][site][model] = np.corrcoef(Yobs,Yapost)[0,1]
                stats['rmse'][site][model] = np.sqrt(np.mean((Yapost-Yobs)**2))
                stats['nrmse'][site][model] = stats['rmse'][site][model]/np.mean(Yobs)
                stats['std'][site][model] = np.std(Yapost-Yobs)

            else:
                for stat in stat_vars:
                    stats[stat][site][model] = np.nan

        # Delete site from dict if first statistics is NaN        
        if all([np.isnan(v) for v in stats['pearson'][site].values()]) == True:
            for stat in stat_vars:
                del stats[stat][site]
    
    return stats
