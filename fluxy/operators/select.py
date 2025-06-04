import pandas as pd
import xarray as xr
import numpy as np
import os
from pathlib import Path
import logging

from fluxy.operators.convert import scale_variables

logger = logging.getLogger(__name__)


def slice_flux(
    ds_all: dict[str, xr.Dataset],
    config_data: dict[str, str | float] = {},
    start_date: str | list[str] = None,
    end_date: str | list[str] = None,
    species: str = None,
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
        species (str):
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

    if species is not None:
        species_info = config_data["species_info"][species]

    if type(start_date) is str:
        start_date = [start_date] * len(ds_all.keys())
        end_date = [end_date] * len(ds_all.keys())

    for im, m in enumerate(ds_all.keys()):
        logger.info(f"Masking data from {m}.")

        # Slice data according to time window
        ds_all[m] = ds_all[m].sel(time=slice(start_date[im], end_date[im]))

        if len(ds_all[m]["time"]) == 0:
            logger.warning(
                f"No {m} fluxes found between {start_date[im]} and {end_date[im]}."
            )
            ds_all.pop(m)
            continue

        # Scale fluxes
        if species is not None:
            ds_all[m] = scale_variables(
                m,
                ds_all[m],
                species_info,
                flux_unit=flux_units_print,
                country_flux_unit=country_flux_units_print,
            )

    return ds_all


def slice_mf(
    ds_all: dict[str, xr.Dataset],
    start_date: str = None,
    end_date: str = None,
    site: str = None,
    baseline_site: str = None,
    data_dir: os.PathLike | None = None,
    mf_units_print: str = None,
    keep_unassimilated: bool = False,
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
        keep_unassimilated (bool):
            If True, keeps unassimilated data (assimilation_flag != 1).
            If False, only keeps assimilated data (assimilation_flag == 1).
    Returns:
        ds_all (dictionary of datasets):
            xarray datasets, scaled and sliced between chosen dates and for
            chosen site.
    """

    models = list(ds_all.keys())

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Get logical array with baseline timestamps
    if baseline_site is not None:
        if data_dir is None:
            raise ValueError(
                "Baseline site is set, but no data_dir provided. "
                "Please provide a data_dir to read baseline timestamps."
            )
        data_dir = Path(data_dir)
        baseline_file = (
            data_dir
            / "intem_baseline_timestamps"
            / f"{baseline_site}_InTEM_baseline_timestamps.nc"
        )

        # Check if files exists
        if not baseline_file.is_file():
            raise FileNotFoundError(
                f"Cannot find baseline file for masking: {baseline_file}."
            )

        # Read baseline file
        with xr.open_dataset(baseline_file) as f:
            baseline = f.sel(time=slice(start_date, end_date))

    for m in models:
        logger.info(f"Masking data from {m}.")

        # Compute offset
        if "Yav" in ds_all[m].keys():
            offset = int(np.mean(ds_all[m]["Yav"]))
        else:
            offset = (
                ds_all[m]["time"].values[1].astype("datetime64[h]")
                - ds_all[m]["time"].values[0].astype("datetime64[h]")
            ).astype(int)

        # Round time to seconds (for consistency between models)
        ds_all[m]["time"] = ds_all[m]["time"].dt.round("s")

        # Slice data according to time window
        mask = (ds_all[m]["time"] >= start_date) & (ds_all[m]["time"] <= end_date)
        if not keep_unassimilated:
            # Mask assimilated data only
            mask &= ds_all[m]["assimilation_flag"] == 1
        ds_all[m] = ds_all[m].where(mask, drop=True)

        # Slice data according to site
        if site is not None:
            site_index = get_site_index(ds_all[m], site)

            if site_index is None:
                logger.warning(f"No {m} obs found for {site}.")
                ds_all.pop(m)
                continue

            mask_site = ds_all[m]["number_of_identifier"] == site_index
            ds_all[m] = ds_all[m].where(mask_site, drop=True)

        if len(ds_all[m]["time"]) == 0:
            # Remove model if no data left after time slicing
            logger.warning(
                f"No {m} obs found for {site=} between {start_date} and {end_date}."
            )
            ds_all.pop(m)
            continue

        # Scale mole fractions
        ds_all[m] = scale_variables(m, ds_all[m], mf_unit=mf_units_print)

        # Mask mole fractions according to baseline timestamps
        if baseline_site is not None:
            logger.info("Masking timeseries to only include baseline times.")

            # average baseline mask over obs averaging period
            b = baseline.resample(time=f"{offset}h").mean()
            # adjust baseline mask time back to centre of av period (resample removes this)
            b["time"] = b["time"] + np.timedelta64(offset, "h") / 2

            # mask baseline mask again, to only include timestamps where every period in the averaging period is classified as baseline
            b_masked = b.sel(time=b["time"][np.where(b["baseline"] == 1.0)])

            # mask dataset using only baseline times
            both_times = np.isin(ds_all[m].time, b_masked.time)
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

    if site in ds["platform"]:
        index = np.where(ds["platform"] == site)[0][0]
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
        sites = np.concatenate([sites, ds["platform"].values])

    sites = np.sort(np.unique(sites))

    return sites
