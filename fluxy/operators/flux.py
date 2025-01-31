import numpy as np
import xarray as xr

def get_flux_mean(
    data: xr.DataArray, 
    season: str = None,
    ) -> xr.DataArray:
    """
    Calculate the mean flux along the 'time' dimension from a dataset, optionally for a specific season.

    Args:
        data (xr.DataArray): 
            The input data containing a 'time' dimension to calculate the mean.
        season (str, optional): 
            The season for which to calculate the mean (e.g., 'DJF', 'MAM', 'JJA', 'SON'). 
            If None, the mean is calculated over the entire 'time' dimension.

    Returns:
        xr.DataArray: 
            The computed mean flux, either over the entire time period or for the specified season.
    """
    if season is None:
        return data.mean(dim="time")

    # Group by season and check if the given season exists
    seasonal_means = data.groupby("time.season").mean(dim="time")

    if season not in seasonal_means.season.values:
        raise ValueError(f"Season '{season}' not found in the dataset.")

    return seasonal_means.sel(season=season)