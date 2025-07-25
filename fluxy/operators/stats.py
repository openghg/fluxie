import logging
import numpy as np
import pandas as pd
import xarray as xr
from fluxy.operators.select import get_unique_sites, get_site_index


def stats_observed_vs_simulated(
    ds_all: dict[str, dict],
    obs_var: str,
    sim_var: str,
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

    logger = logging.getLogger(__name__)

    # names of sites
    sites_all = get_unique_sites(ds_all)

    # init empty list to hold results for individual sites
    stats = []

    # Compute stats for all sites and all models
    for site in sites_all:
        for model, ds in ds_all.items():
            # Remove the NaNs
            ds = ds.dropna("index")
            site_index = get_site_index(ds, site)
            if site_index is None:
                continue
            mask_site = ds["number_of_identifier"] == site_index
            if not mask_site.any():
                continue
            ds_site = ds.where(mask_site, drop=True)

            # select what to compare
            obs = ds_site[obs_var]
            sim = ds_site[sim_var]

            # Check that they are on the same coordinates
            if not obs.coords.equals(sim.coords):
                logger.warning(f"Coordinates do not match for {site} in {model}.")
                continue

            obs, sim = obs.values, sim.values

            # calculate stats
            stats_site = {
                "model": model,
                "site": site,
                "pearson": np.corrcoef(obs, sim)[0, 1],
                "rmse": np.sqrt(np.mean((sim - obs) ** 2)),
                "crmse": np.sqrt(np.mean((sim - obs - np.mean(sim - obs)) ** 2)),
                "bias": np.mean(sim - obs),
                "mae": np.mean(np.abs(sim - obs)),
                "mre": np.mean(np.abs((sim - obs) / obs)),
                "mean_sim": np.mean(sim),
                "mean_obs": np.mean(obs),
                "sd_sim": np.std(sim),
                "sd_obs": np.std(obs),
                "sd_res": np.std(sim - obs),
                "nn": np.size(sim),
                "variable_sim": sim_var,
                "variable_obs": obs_var,
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
