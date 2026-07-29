import numpy as np
import xarray as xr
import pandas as pd
import logging
from typing import Literal

logger = logging.getLogger(__name__)


def _timedelta_to_ns(value: np.timedelta64) -> np.timedelta64:
    """Convert a timedelta scalar to nanoseconds for stable comparisons."""
    return value.astype("timedelta64[ns]")


def _get_time_tolerance(period: np.timedelta64) -> np.timedelta64:
    return np.timedelta64(int(_timedelta_to_ns(period).astype(int) * 0.1), "ns")


def _is_regular_or_missing_periods(
    dtime: np.ndarray, period: np.timedelta64, tolerance: np.timedelta64
) -> bool:
    """
    Check that time gaps are close to the period or whole multiples of it.

    Missing complete periods create gaps like 2 years in otherwise yearly data.
    Those are acceptable for alignment, but non-periodic gaps are not.
    """

    period_ns = _timedelta_to_ns(period).astype(int)
    tolerance_ns = tolerance.astype(int)
    dtime_ns = dtime.astype("timedelta64[ns]").astype(int)

    if period_ns <= 0:
        return False

    multiples = np.maximum(np.rint(dtime_ns / period_ns).astype(int), 1)
    return np.all(np.abs(dtime_ns - multiples * period_ns) <= tolerance_ns)


def _infer_period_from_frequency_attr(
    ds_list: list[xr.Dataset],
) -> np.timedelta64 | None:
    frequency_periods = {
        "month": np.timedelta64(30, "D"),
        "monthly": np.timedelta64(30, "D"),
        "year": np.timedelta64(365, "D"),
        "yearly": np.timedelta64(365, "D"),
        "annual": np.timedelta64(365, "D"),
        "annually": np.timedelta64(365, "D"),
    }
    periods = []
    for ds in ds_list:
        frequency = str(ds.attrs.get("frequency", "")).lower()
        if frequency in frequency_periods:
            periods.append(frequency_periods[frequency])

    if not periods:
        return None

    if not all(period == periods[0] for period in periods[1:]):
        raise ValueError(
            "Unable to infer period from dataset: datasets have incompatible frequency attributes."
        )

    return periods[0]


def _infer_period(ds_list: list[xr.Dataset]) -> np.timedelta64:
    period_from_attr = _infer_period_from_frequency_attr(ds_list)
    if period_from_attr is not None:
        return period_from_attr

    dtime = []
    for ds in ds_list:
        if ds.time.size <= 1:
            continue
        time_values = np.sort(ds.time.values.astype("datetime64[ns]"))
        dtime.append(time_values[1:] - time_values[:-1])

    if not dtime:
        raise ValueError("Unable to infer period from dataset")

    dtime = np.concatenate(dtime)
    dtime = dtime[dtime > np.timedelta64(0, "ns")]
    if dtime.size == 0:
        raise ValueError("Unable to infer period from dataset")

    candidates = [np.median(dtime), np.min(dtime)]
    for period in candidates:
        tolerance = _get_time_tolerance(period)
        if _is_regular_or_missing_periods(dtime, period, tolerance):
            if np.any(np.abs(dtime - period) > tolerance):
                logger.warning(
                    "Missing time periods detected while aligning datasets. "
                    "Alignment will continue using the inferred period."
                )
            return period

    raise ValueError("Unable to infer period from dataset")


def _target_times_within_tolerance(
    target_time: np.ndarray, ds_time: np.ndarray, tolerance: np.timedelta64
) -> np.ndarray:
    target_time = target_time.astype("datetime64[ns]")
    ds_time = ds_time.astype("datetime64[ns]")

    if target_time.size == 0:
        return np.array([], dtype=bool)
    if ds_time.size == 0:
        return np.zeros(target_time.size, dtype=bool)

    time_diff = np.min(np.abs(target_time[:, None] - ds_time[None, :]), axis=1)
    return time_diff <= tolerance


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
    max_diff = _timedelta_to_ns(tolerance)

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

    # Infer period of datasets (only if one dataset has >1 time step)
    if any(ds.time.size > 1 for ds in ds_list):
        period = _infer_period(ds_list)
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
    tolerance = _get_time_tolerance(period)

    if only_overlapping:
        # Reduce datasets to their overlapping time range
        min_date = max([x.time.min() for x in ds_list]) - period / 2
        max_date = min([x.time.max() for x in ds_list]) + period / 2
        sliced_ds_list = [ds.sel(time=slice(min_date, max_date)) for ds in ds_list]

        # Redefine timestamps according to times available in all datasets.
        target_time = sliced_ds_list[0].time.values
        for ds_p in sliced_ds_list[1:]:
            target_time = target_time[
                _target_times_within_tolerance(
                    target_time, ds_p.time.values, tolerance
                )
            ]

        if target_time.size == 0:
            raise ValueError("Datasets do not have overlapping timestamps.")

        aligned_ds_list = [
            ds.reindex(time=target_time, method="nearest", tolerance=tolerance)
            for ds in sliced_ds_list
        ]

        return aligned_ds_list

    # Define new timestamps based on min/max timestamps of all datasets
    min_date = min([x.time.min() for x in ds_list])
    max_date = max([x.time.max() for x in ds_list]) + period / 2
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
    ds_list: list[xr.Dataset],
    coord: Literal["latitude", "longitude"],
    rel_tolerance: float = 0.01,
    min_rel_overlap: float = 0.05,
) -> list[xr.Dataset]:
    """
    Check the latitude/longitude coordinate of a list of xarray datasets and align them.

    If coordinates cover a different range, select the intersection.
    If the coordinates agree approximately, align them with the
    latitudes/longitudes of the first dataset in the list.
    If the coordinates differ such that an interpolation is required,
    raise a ValueError.

    Args:
        ds_list: list of xarray datasets to be latitude/longitude-aligned
        coord: coordinate name (latitude or longitude)
        rel_tolerance: float, tolerance coordinates relative to grid spacing
        min_rel_overlap: float, minimum relative overlap of coordinates.
    Returns:
        aligned_ds_list: list of xarray datasets latitude/longitude-aligned
    """

    # Check if coordinates agree exactly.
    dim_equal = all(ds_list[0][coord].equals(x[coord]) for x in ds_list[1:])
    if dim_equal:
        return ds_list

    tolerance = rel_tolerance * abs(ds_list[0][coord].diff(coord).mean().item())

    # Select common range of coordinates if the coordinate sizes differ.
    ref_dim_size = ds_list[0][coord].size
    dim_sizes_differ = any(x[coord].size != ref_dim_size for x in ds_list[1:])
    if dim_sizes_differ:
        start = max(x[coord].values[0] for x in ds_list) - tolerance
        end = min(x[coord].values[-1] for x in ds_list) + tolerance
        ds_list = [x.sel({coord: slice(start, end)}) for x in ds_list]
        # Check if the coordinate sizes agree now.
        common_dim_size = ds_list[0][coord].size
        dim_sizes_differ = any(x[coord].size != common_dim_size for x in ds_list[1:])
        if dim_sizes_differ:
            raise ValueError(
                f"{coord} dimensions seem to be too different between the datasets for them to be combined."
            )
        # Fail if the overlap of the area covered by all models is too small
        if common_dim_size < min_rel_overlap * ref_dim_size:
            raise ValueError(
                f"{coord} dimensions of the datasets cover too different ranges for them to be combined (threshold: {min_rel_overlap:%}."
            )

    # Check if the coordinates agree approximately within the given tolerance.
    ds_ref = ds_list[0]
    reference = ds_ref[coord].values
    dim_close = all(
        np.allclose(reference, x[coord].values, atol=tolerance, rtol=0)
        for x in ds_list[1:]
    )
    if not dim_close:
        raise ValueError(
            f"{coord} dimensions seem to be too different between the datasets for them to be combined."
        )
    # Use coordinates of first model as reference.
    aligned_ds_list = [ds_ref]

    for ds_p in ds_list[1:]:
        if ds_ref[coord].equals(ds_p[coord]):
            aligned_ds_list.append(ds_p)
            continue
        # Replace coordinate with coordinate of first model.
        ds_aligned = ds_p.copy(deep=False)
        ds_aligned[coord] = ds_ref[coord]
        aligned_ds_list.append(ds_aligned)

    return aligned_ds_list


def align_map_data(
    ds_all: dict[xr.Dataset | xr.DataArray],
    only_overlapping: bool = True,
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
    ds_list = align_time(ds_list, only_overlapping=only_overlapping)
    ds_list = align_lat_lon(ds_list, coord="latitude")
    ds_list = align_lat_lon(ds_list, coord="longitude")

    return dict(zip(models, ds_list))
