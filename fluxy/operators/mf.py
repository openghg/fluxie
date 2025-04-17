import numpy as np
import xarray as xr
import pandas as pd
import logging
from fluxy.operators.select import get_unique_sites, get_site_index
from fluxy.operators.convert import get_variables

logger = logging.getLogger(__name__)


def compute_mf_difference(
    ds_all: dict[str, xr.Dataset], models_to_subtract: list[str]
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
        raise ValueError("List of models to subtract must be of size 2.")

    for m in models_to_subtract:
        if m not in models:
            raise KeyError(f"{m} not found in the dataset.")

    # Reduce datasets to timestamps/sites common to both models
    ds0, ds1 = xr.align(
        ds_all[models_to_subtract[0]], ds_all[models_to_subtract[1]], join="inner"
    )

    ds_diff = {}
    key_name = f"{models_to_subtract[0]}-{models_to_subtract[1]}"
    ds_diff[key_name] = xr.Dataset()

    # Compute difference between the two datasets (mole fraction variables only)
    var_names0, x = get_variables(ds0, "mf")
    var_names1, x = get_variables(ds1, "mf")
    common_mf_vars = list(set(var_names0) & set(var_names1))

    for v in common_mf_vars:
        units_0 = ds0[v].attrs["units"]
        units_1 = ds1[v].attrs["units"]
        if units_0 != units_1:
            logger.warning(
                f"{v} in {models_to_subtract[0]} and {models_to_subtract[1]} have different units. {v} will not be included in the diff dataset."
            )
            continue

        ds_diff[key_name][v] = ds0[v] - ds1[v]
        ds_diff[key_name][v].attrs["units"] = units_0

    return ds_diff


def stats_mf(ds_all: dict[str, dict], type='prior') -> pd.DataFrame:
    """
    Calculates multiple statistical measures of the fit between the posterior
    mean mf and the observed mole fraction.
    Implemented statistics: Pearson correlation coefficent, root mean square
    error, normalised root mean square error, standard deviation.

    Args:
        ds_all (dictionary of datasets):
            xarray datasets from slice_mf(), sliced between chosen dates
            but still containing all sites.
        type : 
            type of statistics to be computed. One of 'prior', 'posterior' for 
            statistics on the absolute mole fractions and 'prior_above_BC', 
            'posterior_above_BC' for regional part of mole fraction, i.e. with 
            BC contribution subtracted from both observation and simulation. 
    Returns:
        stats (pandas.DataFrame):
            Statistical measures, for each site and for each model.
    """
    type_options = ['prior', 'posterior', 'prior_above_BC', 'posterior_above_BC']
    assert type in type_options, f"'{type}' is not in {type_options}"
    sites_all = get_unique_sites(ds_all)

    # Implemented statistics
    stat_vars = ["pearson", "rmse", "bias", "nrmse", "std", "sd_sim", "sd_ref"]
    
    # init empty variable to hold DataFrame 
    stats = None

    # Compute stats for all sites and all models
    for site in sites_all:
        for model, ds in ds_all.items():
            site_index = get_site_index(ds, site)
            if (site_index is not None) and (
                ds["Yobs"].isel(nsite=site_index).count() != 0
            ):
                # xarray for single site
                dc = ds.isel(nsite=site_index).dropna(dim="time")
                
                # select what to compare
                if (type=="prior"):
                    obs = dc["Yobs"]
                    sim = dc["Yapriori"]
                elif (type=="posterior"):
                    obs = dc["Yobs"]
                    sim = dc["Yapost"]
                elif (type=="prior_above_BC"):
                    obs = dc["Yobs"]     - dc["YaprioriBC"]                    
                    sim = dc["Yapriori"] - dc["YaprioriBC"]                    
                elif (type=="posterior_above_BC"):
                    obs = dc["Yobs"]   - dc["YapostBC"]
                    sim = dc["Yapost"] - dc["YapostBC"]
                    
                # calculate stats
                d = {
                    'model': model, 
                    'site': site, 
                    'pearson': np.corrcoef(obs, sim)[0, 1], 
                    'rmse': np.sqrt(np.mean((sim-obs)**2)).item(), 
                    'crmse': np.sqrt(np.mean((sim-obs-np.mean(sim - obs))**2)).item(), 
                    'bias': np.mean(sim-obs).item(), 
                    'sd_sim': np.std(sim).item(), 
                    'sd_ref': np.std(obs).item(), 
                    'std': np.std(sim - obs).item(),
                    'nn': np.size(sim)
                    }            
                
                # create DataFrame
                tmp = pd.DataFrame(data=d, index = [0])                
                
                # additional derived stats
                tmp['nrmse'] = tmp['rmse'].values/np.mean(obs).item(0)                 
                
                # concatenate DataFrames after each iteration 
                if stats is None:
                    stats = tmp
                else: 
                    stats = pd.concat([stats, tmp], ignore_index=True)
    
    return stats
