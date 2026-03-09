import numpy as np
import xarray as xr
import pandas as pd


def calculate_resampled_flux(
    ds_all: dict[str, xr.Dataset], rtime: list[str]
) -> dict[str, xr.Dataset]:
    """
    Resample the datasets.

     Args:
        ds_all: xarray datasets of fluxes, scaled and sliced between chosen dates.
        rtime: list of time period used for resampling, e.g. 'YS' or 'QS-DEC'.
             See http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
     Returns:
        ds_all_output: resampled datasets
    """
    ds_all_output = dict()
    for i, (m, ds) in enumerate(ds_all.items()):
        if rtime[i] is not None:
            ds_all_output[m] = ds.resample(time=rtime[i]).mean(dim="time")
        else:
            ds_all_output[m] = ds.copy()

    return ds_all_output


def calculate_resampled_uncertainty(
    ds_all_original: dict[str, xr.Dataset],
    ds_all_resampled: dict[str, xr.Dataset],
    rtime: list[str],
) -> dict[str, xr.Dataset]:
    """
    Recalculates resampled flux uncertainty, using the assumption
    that all periods in the resampled flux average are uncorrelated.
    Args:
        ds_all_original: Extracted flux datasets from flux-format netcdfs.
        ds_all_resampled: Same datasets than above that have been resampled with calculate_resampled_flux.
        rtime: list of time period used for resampling, e.g. 'YS' or 'QS-DEC'.
            See http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
    Returns:
        ds_all_resampled: Same dataset as above, but with updated '..._upper'/'..._lower' terms.
    """

    for i, m in enumerate(ds_all_original.keys()):
        if rtime[i] is None:
            continue

        for v in ["posterior", "prior"]:
            n_periods = (
                ds_all_original[m][v].resample(time=rtime[i]).count()
            )  # count the number of sample in each period

            lower = (
                np.sqrt(
                    ((ds_all_original[m][f"{v}_lower"] - ds_all_original[m][v]) ** 2)
                    .resample(time=rtime[i])
                    .sum(dim="time")
                )
                / n_periods
            )
            upper = (
                np.sqrt(
                    ((ds_all_original[m][f"{v}_upper"] - ds_all_original[m][v]) ** 2)
                    .resample(time=rtime[i])
                    .sum(dim="time")
                )
                / n_periods
            )

            ds_all_resampled[m][f"{v}_lower"] = ds_all_resampled[m][v] - lower
            ds_all_resampled[m][f"{v}_upper"] = ds_all_resampled[m][v] + upper

    return ds_all_resampled


def resample_flux(
    ds_all: dict[str, xr.Dataset],
    resample: list[str],
    resample_uncert_correlation: bool = False,
) -> dict[str, xr.Dataset]:
    """
    Resample the datasets and align their time dimension to the middle of the resampling time period.

    Args:
        ds_all: xarray datasets of fluxes, scaled and sliced between chosen dates.
        resample: Option to be passed to resample built-in function of xarray Dataset. For yearly average, 'YS' option should be used; 'QS-DEC' for seasonaly average.
            See http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
        resample_uncert_correlation: If True, calculates the resampled uncertainty as the mean from all averaged periods.
            If False, recalculates uncertainty assuming no correlation between all averaged periods, by taking the square root of the summed variances, divided by the number of averaging periods.
    Returns:
        ds_all_resampled: resampled datasets
    """

    rtime = []

    for resample_val in resample:
        if resample_val == "year":
            rtime.append("YS")

        elif resample_val == "season":
            rtime.append("QS-DEC")

        elif resample_val is None:
            rtime.append(None)

        else:
            raise ValueError(
                f"'{resample_val}' is not available for resample. Try 'year' or 'season' or None."
            )

    ds_all_original = {m: ds_all[m].copy() for m in ds_all.keys()}

    ds_all_resampled = calculate_resampled_flux(ds_all, rtime)

    if not resample_uncert_correlation:
        ds_all_resampled = calculate_resampled_uncertainty(
            ds_all_original, ds_all_resampled, rtime
        )

    # shift timestamps of averaged data forwards to centre of inversion period
    new_keys = list(ds_all.keys())
    for im, m in enumerate(ds_all.keys()):

        if (
            rtime[im] is not None
            and ds_all_original[m].time.size > ds_all_resampled[m].time.size
        ):

            # I feel like there is better ways of doing this but didn't find them -> check openGHG_inversions
            # in fact do it on the plot axis (or not)
            date_list_for_offset = pd.date_range(
                start="2018-01-01", end="2023-01-01", freq=rtime[im]
            )
            offset = (date_list_for_offset[1:] - date_list_for_offset[:-1]).mean() / 2
            ds_all_resampled[m]["time"] = ds_all_resampled[m]["time"].values + offset
            new_keys[im] = m + "_resample"

            ds_all_resampled[m].attrs["frequency"] = resample[im]

    return {
        key: ds.dropna(dim="time", how="all")
        for key, ds in zip(new_keys, ds_all_resampled.values())
    }
