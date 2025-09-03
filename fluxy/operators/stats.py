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
    Calculates multiple statistical measures of the fit between the observed
    and simulated variables.

    Statistics are computed for each site and each model.

    Implemented statistics:
        * Standard statistics
        * Pearson correlation coefficent
        * Root mean square error
        * Normalised root mean square error

    Args:
        ds_all (dictionary of datasets):
            xarray datasets
        obs_var (str):
            Name of the observed variable.
        sim_var (str):
            Name of the simulated variable.

    Returns:
        stats (pandas.DataFrame):
            Statistical measures, for each site and for each model between observations and
            simulations. Columns:
                * 'model': model string
                * 'site': observation platform ID
                * 'pearson': Pearson correlation coefficient
                * 'mae': mean absolute error
                * 'mre': mean relative error
                * 'rmse': root mean square error
                * 'nrmse': root mean square error normalised by observation mean
                * 'crmse': centered root mean square error
                * 'bias': bias
                * 'std_sim': standard deviation of simulation
                * 'std_obs': standard deviation of observation (reference)
                * 'std_res': standard deviation of simulation - observation (residuals)
                * 'nn': number of value pairs
                * 'variable_sim': name of the simulated variable
                * 'variable_obs': name of the observed variable
                * 'unit': unit of the variables
                * '{min|max}_{sim|obs}': minimum/maximum value of the simulated/observed variable
                * 'mean_{sim|obs}': mean value of the simulated/observed variable
                * 'median_{sim|obs}': median of the simulated/observed variable
                * 'std_{sim|obs}': standard deviation of the simulated/observed variable
                * 'q{05|25|75|95}_{sim|obs}': quantiles of the simulated/observed variable

    """

    logger = logging.getLogger(__name__)

    # names of sites
    sites_all = get_unique_sites(ds_all)

    # init empty list to hold results for individual sites
    stats = []

    # Compute stats for all sites and all models
    for site in sites_all:
        for model, ds in ds_all.items():
            site_index = get_site_index(ds, site)
            if site_index is None:
                logger.warning(f"Site {site} not found in model {model}.")
                continue
            mask_site = ds["number_of_identifier"] == site_index
            if not mask_site.any():
                logger.warning(
                    f"No data for site {site} with index {site_index} in model {model}."
                )
                continue
            ds_site = ds.where(mask_site, drop=True)

            # select what to compare
            obs = ds_site[obs_var]
            sim = ds_site[sim_var]

            # Remove the nans
            mask_nan = ~np.isnan(obs) & ~np.isnan(sim)
            if not mask_nan.any():
                logger.warning(f"No valid data for site {site} in model {model}.")
                continue
            obs = obs.where(mask_nan, drop=True)
            sim = sim.where(mask_nan, drop=True)

            # Check that they are on the same coordinates
            if not obs.coords.equals(sim.coords):
                logger.warning(f"Coordinates do not match for {site} in {model}.")
                continue

            unit_obs = obs.attrs.get("units", "-")
            unit_sim = sim.attrs.get("units", "-")
            if unit_obs != unit_sim:
                logger.warning(
                    f"Units do not match for {site} in {model}: {unit_obs} vs {unit_sim}."
                )
            unit = unit_obs

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
                "std_res": np.std(sim - obs),
                "nn": np.size(sim),
                "variable_sim": sim_var,
                "variable_obs": obs_var,
                "unit": unit,
            }
            stats_site |= {
                f"q{i:02d}_{simobs}": np.quantile(
                    sim if simobs == "sim" else obs, i / 100
                )
                for i in [5, 25, 75, 95]
                for simobs in ["sim", "obs"]
            }
            stats_site |= {
                f"{func}_{simobs}": getattr(np, func)(sim if simobs == "sim" else obs)
                for func in ["max", "min", "mean", "median", "std"]
                for simobs in ["sim", "obs"]
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
