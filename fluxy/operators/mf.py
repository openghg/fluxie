import numpy as np
import xarray as xr
import pandas as pd
import logging
from fluxy.operators.select import get_unique_sites, get_site_index
from fluxy.operators.convert import get_variables
from typing import Literal

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


def stats_mf(
    ds_all: dict[str, dict],
    stats_type: Literal[
        "prior", "posterior", "prior_above_BC", "posterior_above_BC"
    ] = "prior",
) -> pd.DataFrame:
    """
    Calculates multiple statistical measures of the fit between the posterior
    mean mf and the observed mole fraction.
    Implemented statistics: Pearson correlation coefficent, root mean square
    error, normalised root mean square error, standard deviation.

    Args:
        ds_all (dictionary of datasets):
            xarray datasets from slice_mf(), sliced between chosen dates
            but still containing all sites.
        stats_type :
            type of statistics to be computed. One of 'prior', 'posterior' for
            statistics on the absolute mole fractions and 'prior_above_BC',
            'posterior_above_BC' for regional part of mole fraction, i.e. with
            BC contribution subtracted from both observation and simulation.
    Returns:
        stats (pandas.DataFrame):
            Statistical measures, for each site and for each model between observations and
            simulations. Columns: 'model': model string,
            'site': observation platform ID, 'pearson': Pearson correlation coefficient,
            'rmse': root mean square error, 'crmse': centered root mean square error,
            'bias': bias, 'sd_sim': standard deviation of simulation,
            'sd_obs': standard deviation of observation (reference),
            'sd_res': standard deviation of simulation - observation (residuals) ,
            'nrmse': root mean square error normalised by observation mean,
            'nn': number of value pairs. Index: integer.
    """

    # assure that stats_type in allowed options
    type_options = ["prior", "posterior", "prior_above_BC", "posterior_above_BC"]
    assert stats_type in type_options, f"'{stats_type}' is not in {type_options}"

    # names of sites
    sites_all = get_unique_sites(ds_all)

    # init empty list to hold results for individual sites
    stats = []

    # Compute stats for all sites and all models
    for site in sites_all:
        for model, ds in ds_all.items():
            site_index = get_site_index(ds, site)
            if (site_index is not None) and (
                ds["Yobs"].isel(nsite=site_index).count() != 0
            ):
                # xarray for single site
                ds_site = ds.isel(nsite=site_index).dropna(dim="time")

                # select what to compare
                if stats_type == "prior":
                    obs = ds_site["Yobs"].values
                    sim = ds_site["Yapriori"].values
                elif stats_type == "posterior":
                    obs = ds_site["Yobs"].values
                    sim = ds_site["Yapost"].values
                elif stats_type == "prior_above_BC":
                    obs = ds_site["Yobs"].values - ds_site["YaprioriBC"].values
                    sim = ds_site["Yapriori"].values - ds_site["YaprioriBC"].values
                elif stats_type == "posterior_above_BC":
                    obs = ds_site["Yobs"].values - ds_site["YapostBC"].values
                    sim = ds_site["Yapost"].values - ds_site["YapostBC"].values
                else:
                    raise ValueError()

                # calculate stats
                stats_site = {
                    "model": model,
                    "site": site,
                    "pearson": np.corrcoef(obs, sim)[0, 1],
                    "rmse": np.sqrt(np.mean((sim - obs) ** 2)),
                    "crmse": np.sqrt(np.mean((sim - obs - np.mean(sim - obs)) ** 2)),
                    "bias": np.mean(sim - obs),
                    "sd_sim": np.std(sim),
                    "sd_obs": np.std(obs),
                    "sd_res": np.std(sim - obs),
                    "nn": np.size(sim),
                }

                # change to DataFrame
                stats_site = pd.DataFrame(data=stats_site, index=[0])

                # additional derived stats
                stats_site["nrmse"] = stats_site["rmse"].values / np.mean(obs)

                # append to list of all stats
                stats.append(stats_site)

    # fold list of DataFrames into a single DataFrame
    stats = pd.concat(stats, ignore_index=True)

    return stats
