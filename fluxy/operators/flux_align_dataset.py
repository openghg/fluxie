import numpy as np
import xarray as xr
from typing import Literal


def align_time(ds_list: list[xr.Dataset]) -> list[xr.Dataset]:
    """
    Check the time coordinates of a list of xarray datasets and, if they differ, align them with the time coordinate of the first dataset in the list.

    Args:
        ds_list: list of xarray datasets to be time-aligned
    Returns:
        aligned_ds_list: list of xarray datasets time-aligned
    """
    time_dim_equal = [ds_list[0].time.equals(x.time) for x in ds_list[1:]]

    if all(time_dim_equal):
        return ds_list

    # Infer period of first dataset
    dtime = ds_list[0].time.values[1:] - ds_list[0].time.values[:-1]
    if any(abs(dtime - np.median(dtime)) > 0.1 * np.median(dtime)):
        raise ValueError("Unable to infer period from dataset")
    period = np.median(dtime)

    aligned_ds_list = [ds_list[0]]

    for ds_p in ds_list[1:]:
        if ds_list[0].time.equals(ds_p.time):
            aligned_ds_list.append(ds_p)
            continue

        diff_time = abs(ds_list[0].time.values - ds_p.time.values)
        if any(diff_time > 0.1 * period):
            raise ValueError(
                f"Time dimensions seem to be too different between the datasets for them to be combined "
                + f'(period of reference dataset: {period.astype("timedelta64[D]")}, max difference: {max(diff_time).astype("timedelta64[D]")})'
            )

        ds_aligned = ds_p
        ds_aligned["time"] = ds_list[0].time
        aligned_ds_list.append(ds_p)

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


def align_map_data(ds_all: dict[xr.Dataset]) -> dict[xr.Dataset]:
    """
    Prepare flux datasets for flux maps by filtering variables, removing unused dimensions, and aligning time and spatial coordinates.

    Args:
        ds_all (dict[xr.Dataset]):
            Dictionary of model names and corresponding xarray datasets.

    Returns:
        dict[xr.Dataset]:
            Aligned datasets, after removing non-geographic variables.
    """

    for key, ds in ds_all.items():
        # Step 1: Remove variables without 'time', 'latitude' and 'longitude'
        ds = ds.drop_vars(
            [
                var
                for var in ds.data_vars
                if not {"time", "latitude", "longitude"}.issubset(ds[var].dims)
            ]
        )
        # Step 2: Remove unused coordinates (dimensions that are no longer used)
        unused_dims = set(ds.dims) - set(
            dim for var_i in ds.data_vars for dim in ds[var_i].dims
        )
        ds_all[key] = ds.drop_dims(unused_dims)

    # Align dataset coordinates
    models = list(ds_all.keys())
    ds_list = list(ds_all.values())
    ds_list = align_time(ds_list)
    ds_list = align_lat_lon(ds_list, coord="latitude")
    ds_list = align_lat_lon(ds_list, coord="longitude")

    return dict(zip(models, ds_list))
