import numpy as np
import xarray as xr
import pandas as pd
import logging
from typing import Literal

logger = logging.getLogger(__name__)


def check_times_within_tolerance(
    ds: xr.Dataset, target_time: np.ndarray, tolerance: np.timedelta64
):
    """
    Check if timestamps of a dataset are within a given tolerance of at least one time value in a target time array.

    Args:
        ds: xarray dataset with time dimension
        target_time: target time array
        tolerance: maximum time difference allowed between timestamps.
    """

    # Convert to datetime
    ds_time = ds.time.values.astype("datetime64[ns]")
    target_ds_time = target_time.astype("datetime64[ns]")
    max_diff = np.timedelta64(int(tolerance), "ns")

    # Get difference between ds_time and target_time
    time_diff = np.min(np.abs(ds_time[:, None] - target_ds_time[None, :]), axis=1)

    # Check if difference is within tolerance
    if np.any(time_diff > max_diff):
        raise ValueError(
            f"{ds.attrs['inversion_system']}, {ds.attrs['species']}: Timestamps are too different from target time array. Timestamps cannot be aligned."
        )


def align_time(
    ds_list: list[xr.Dataset], only_overlapping: bool = True
) -> list[xr.Dataset]:
    """
    Check the time coordinate of a list of datasets, then aligns them,
    either only between overlapping time periods, or across all available
    times (if only_overlapping is False).

    Args:
        ds_list: list of xarray datasets to be time-aligned
        only_overlapping: if True, reduce datasets to their overlapping time range before aligning
    Returns:
        aligned_ds_list: list of xarray datasets time-aligned
    """

    # Check if all timestamps are equal
    time_dim_equal = [ds_list[0].time.equals(x.time) for x in ds_list[1:]]

    if all(time_dim_equal):
        return ds_list

    # Infer period of first dataset (only if it has >1 time step)
    if ds_list[0].time.size > 1:
        dtime = ds_list[0].time.values[1:] - ds_list[0].time.values[:-1]
        if any(abs(dtime - np.median(dtime)) > 0.1 * np.median(dtime)):
            raise ValueError("Unable to infer period from dataset")
        period = np.median(dtime)
    else:
        logger.warning(
            "Datasets have only one time value — aligning them with the time "
            "coordinate of the first dataset in the list "
            "without checking time difference."
        )

        # Redefine timestamps according to first dataset
        aligned_ds_list = [ds_list[0]]
        for ds_p in ds_list[1:]:
            ds_p["time"] = ds_list[0].time
            aligned_ds_list.append(ds_p)

        return aligned_ds_list

    # Define time tolerance
    tolerance = 0.1 * period

    if only_overlapping:
        # Reduce datasets to their overlapping time range
        min_date = max([x.time.min() for x in ds_list]) - period / 2
        max_date = min([x.time.max() for x in ds_list]) + period / 2
        sliced_ds_list = [ds.sel(time=slice(min_date, max_date)) for ds in ds_list]

        # Redefine timestamps according to first dataset
        target_time = sliced_ds_list[0].time
        aligned_ds_list = [sliced_ds_list[0]]

        for ds_p in sliced_ds_list[1:]:
            if ds_p.time.equals(target_time):
                aligned_ds_list.append(ds_p)
                continue

            # Check if time difference is within reasonal bounds before rewriting timestamps
            check_times_within_tolerance(ds_p, target_time.values, tolerance)
            ds_p["time"] = target_time
            aligned_ds_list.append(ds_p)

        return aligned_ds_list

    # Define new timestamps based on min/max timestamps of all datasets
    min_date = min([x.time.min() for x in ds_list])
    max_date = max([x.time.max() for x in ds_list])
    target_time = pd.date_range(
        start=min_date.values, end=max_date.values, freq=pd.to_timedelta(period)
    ).values

    # Check if time difference is within reasonal bounds before reindexing
    [check_times_within_tolerance(ds, target_time, tolerance) for ds in ds_list]
    aligned_ds_list = [
        ds.reindex(time=target_time, method="nearest", tolerance=tolerance)
        for ds in ds_list
    ]

    # Check for added NaNs
    for ds in aligned_ds_list:
        test_var = list(ds.data_vars)[0]
        mask_no_data = []
        for i in range(ds.sizes["time"]):
            mask_no_data.append(ds[test_var].isel(time=i).isnull().all().values)

        if np.any(mask_no_data):
            times_no_data = ds.time.values[mask_no_data]
            logger.warning(
                f"NaN is being added to the timeseries of {ds.attrs.get('inversion_system','undefined model')} {ds.attrs.get('species','')} at: {times_no_data}."
            )

    return aligned_ds_list


def align_lat_lon(
    ds_list: list[xr.Dataset], coord: Literal["latitude", "longitude"]
) -> list[xr.Dataset]:
    """
    Check the latitude/longitude coordinate of a list of xarray datasets and, if they differ, align them with the latitudes/longitudes of the first dataset in the list.

    Args:
        ds_list: list of xarray datasets to be latitude/longitude-aligned
    Returns:
        aligned_ds_list: list of xarray datasets latitude/longitude-aligned
    """

    dim_equal = [ds_list[0][coord].equals(x[coord]) for x in ds_list[1:]]

    if all(dim_equal):
        return ds_list

    dim_close = [
        np.allclose(ds_list[0][coord].values, x[coord].values) for x in ds_list[1:]
    ]  # Small tolerance
    if all(dim_close):
        aligned_ds_list = [ds_list[0]]

        for ds_p in ds_list[1:]:
            if ds_list[0][coord].equals(ds_p[coord]):
                aligned_ds_list.append(ds_p)
                continue

            ds_aligned = ds_p
            ds_aligned[coord] = ds_list[0][coord]
            aligned_ds_list.append(ds_p)
    else:
        raise ValueError(
            f"{coord} dimensions seem to be too different between the datasets for them to be combined."
        )

    return aligned_ds_list


def align_map_data(
    ds_all: dict[xr.Dataset | xr.DataArray],
) -> dict[xr.Dataset | xr.DataArray]:
    """
    Prepare flux datasets for flux maps by:
      - filtering variables to only those with expected spatial or platform dimensions,
      - keeping only variables common to all datasets,
      - removing unused dimensions,
      - aligning time and spatial coordinates.

    Args:
        ds_all (dict[xr.Dataset | xr.DataArray]):
            Dictionary of model names and corresponding xarray Datasets/DataArrays.
        only_overlapping (bool):
            If True, reduces datasets to their overlapping time range before aligning.
            If False, includes all data and fills in missing time steps with NaNs.
    Returns:
        dict[xr.Dataset | xr.DataArray]:
            Aligned Datasets/DataArrays, with consistent variables and coordinates.
    """

    # Step 1: Filter variables based on dimension criteria
    for key, ds in ds_all.items():
        if isinstance(ds, xr.DataArray):
            continue
        # Applied only if Dataset and not DataArray
        ds = ds.drop_vars(
            [
                var
                for var in ds.data_vars
                if not (
                    {"time", "latitude", "longitude"}.issubset(ds[var].dims)
                    or {"time", "platform"}.issubset(ds[var].dims)
                )
            ]
        )
        # Remove unused coordinates
        unused_dims = set(ds.dims) - set(
            dim for var_i in ds.data_vars for dim in ds[var_i].dims
        )
        ds_all[key] = ds.drop_dims(unused_dims)

    # Step 2: Keep only variables common to all datasets
    var_sets = [set(ds.data_vars) for ds in ds_all.values()]
    common_vars = set.intersection(*var_sets)

    for key in ds_all:
        ds_all[key] = ds_all[key][list(common_vars)]

    # Step 3: Align dataset coordinates
    models = list(ds_all.keys())
    ds_list = list(ds_all.values())
    ds_list = align_time(ds_list)
    ds_list = align_lat_lon(ds_list, coord="latitude")
    ds_list = align_lat_lon(ds_list, coord="longitude")

    return dict(zip(models, ds_list))
