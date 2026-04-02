import xarray as xr


def calc_rolling_mean(ds: xr.Dataset, time_period: int = 3) -> xr.Dataset:
    """
    Apply a rolling mean to the dataset. For the first (last) value, the mean is calculated with the two first (last) values of the dataset.
    WARNING : Only valid for time_period = 3 for now.

    Args:
        ds : input dataset on which rolling mean will be applied
        time_period : time period for the rolling mean (only 3 is currently coded)
    Return
        dataset where rolling mean have been applied, attributes are conserved
    """

    ds_middle = ds.rolling(time=time_period, center=True).mean()
    ds_middle = ds_middle.dropna(dim="time", how="all")

    ds_first = ds.isel(time=slice(0, 2)).mean()
    ds_first = ds_first.expand_dims(
        dim={
            "time": [
                ds.time.isel(time=0).values,
            ]
        }
    )

    ds_last = ds.isel(time=slice(-2, None)).mean()
    ds_last = ds_last.expand_dims(
        dim={
            "time": [
                ds.time.isel(time=-1).values,
            ]
        }
    )

    return xr.concat(
        [ds_first, ds_middle, ds_last], dim="time", combine_attrs="no_conflicts"
    )
