import numpy as np
import xarray as xr
import pandas as pd
import calendar
import datetime

from typing import List, Tuple, Literal


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
    seasonal_means = data.groupby("time.season", restore_coord_dims=True).mean(
        dim="time"
    )

    if season not in seasonal_means.season.values:
        raise ValueError(f"Season '{season}' not found in the dataset.")

    return seasonal_means.sel(season=season)


def average_over_dates_list(
    ds: xr.Dataset,
    dates_list: List[str],
) -> Tuple[xr.Dataset, List[str]]:
    """
    Average a dataset over custom time intervals defined in `dates_list`.

    This function groups time values in `ds` based on `dates_list` and computes
    the mean for each period. Time labels indicating the date range of each
    averaged period are also generated.

    Args:
        ds (xarray.Dataset):
            Dataset with a "time" dimension.
        dates_list (list of datetime-like):
            Boundaries for averaging periods (e.g., ['2018-01-01', '2020-01-01']).

    Returns:
        ds_avg (xarray.Dataset):
            Dataset averaged over the defined time intervals.
        time_labels (list of str):
            Labels for each averaged period (format: "YYYY/MM—YYYY/MM").
    """

    # Initialize the groups array to store which group each time value belongs to
    groups = np.full(ds.time.shape, np.nan)

    # Skip the first part of the dataset, starting grouping from dates_list[0]
    # Loop through dates_list and assign each time value to the corresponding group
    for i in range(len(dates_list) - 1):  # We go until the second-to-last element
        groups[
            (ds.time.values >= dates_list[i]) & (ds.time.values < dates_list[i + 1])
        ] = (i + 1)

    # For the last group (from dates_list[-1] to the end), assign remaining times
    groups[ds.time.values >= dates_list[-1]] = len(dates_list)

    groups_da = xr.DataArray(groups, coords={"time": ds.time})

    # Remove the group that corresponds to dates before dates_list[0] (NaN values)
    ds = ds.where(~np.isnan(groups_da), drop=True)
    groups_da = groups_da.where(~np.isnan(groups_da), drop=True)

    ds_avg = ds.groupby(groups_da, restore_coord_dims=True).mean(dim="time")
    ds_avg = ds_avg.rename({"group": "time"})

    # Create labels based on the first and last date in each group
    time_labels = []
    for group in np.unique(groups_da):
        group_times = ds.time.values[groups_da == group]  # Get times in this group
        first_time = pd.to_datetime(group_times.min())
        last_time = pd.to_datetime(group_times.max())
        time_labels.append(
            f"{first_time.strftime('%Y/%m')}—{last_time.strftime('%Y/%m')}"
        )

    return ds_avg, time_labels


def average_over_months_list(
    ds: xr.Dataset,
    months_list: List,
) -> Tuple[xr.Dataset, List[str]]:
    """
    Average a dataset over specified months in `months_list`.

    This function groups time values in `ds` by the months defined in `months_list`
    and computes the mean for each group. It also generates labels for each group based
    on the months included.

    Args:
        ds (xarray.Dataset):
            Dataset with a "time" dimension.
        months_list (list of lists or ints):
            List of months (or month groups) to average (e.g., [[7,8]]).

    Returns:
        ds_avg (xarray.Dataset):
            Dataset averaged over the specified months.
        time_labels (list of str):
            Labels for each averaged period (e.g., "Jan—Mar").
    """

    groups = np.full(ds.time.shape, np.nan)
    for i, months in enumerate(months_list, start=1):
        groups[np.isin(ds.time.dt.month, months)] = i  # Assign group i

    unique_groups = np.unique(groups[~np.isnan(groups)])
    if len(months_list) != len(unique_groups):
        raise ValueError("Some months are missing.")
    groups_da = xr.DataArray(groups, coords={"time": ds.time})
    ds_avg = ds.groupby(groups_da, restore_coord_dims=True).mean(dim="time")
    ds_avg = ds_avg.rename({"group": "time"})

    time_labels = []
    for i, group in enumerate(unique_groups):
        if isinstance(months_list[i], list):
            time_labels.append("—".join(calendar.month_abbr[m] for m in months_list[i]))
        else:
            time_labels.append(calendar.month_abbr[months_list[i]])

    return ds_avg, time_labels


def average_over_seasons(
    ds: xr.Dataset,
) -> Tuple[xr.Dataset, List[str]]:
    """
    Average a dataset over seasons.

    This function groups time values in `ds` by their respective seasons and computes
    the mean for each season. It also generates labels for each season.

    Args:
        ds (xarray.Dataset):
            Dataset with a "time" dimension.

    Returns:
        ds_avg (xarray.Dataset):
            Dataset averaged over seasons.
        time_labels (list of str):
            Labels for each season (e.g., "DJF", "MAM").
    """

    groups = ds.time.dt.season
    ds_avg = ds.groupby(groups, restore_coord_dims=True).mean(dim="time")
    ds_avg = ds_avg.rename({"season": "time"})

    desired_order = ["DJF", "MAM", "JJA", "SON"]  # desired season order
    ordered_seasons = [s for s in desired_order if s in ds_avg.time.values]
    ds_avg = ds_avg.sel(time=ordered_seasons)

    time_labels = ds_avg.time.values.tolist()
    return ds_avg, time_labels


def average_over_years(
    ds: xr.Dataset,
    N: int = 1,
) -> Tuple[xr.Dataset, List[str]]:
    """
    Average a dataset over years or custom yearly intervals defined by N.

    This function groups time values in `ds` by year (or custom yearly intervals) and computes
    the mean for each group. It also generates labels for each group based on the
    first and last years in each period.

    Args:
        ds (xarray.Dataset):
            Dataset with a "time" dimension.
        N (int):
            Length of the custom period in years (default is 1, for yearly averages).

    Returns:
        ds_avg (xarray.Dataset):
            Dataset averaged over the specified periods.
        time_labels (list of str):
            Labels for each period (e.g., "2020", "2020—2022").
    """

    groups = ds.time.dt.year // N
    ds_avg = ds.groupby(groups, restore_coord_dims=True).mean(dim="time")
    ds_avg = ds_avg.rename({"year": "time"})

    first_time_groups = ds.time.groupby(groups, restore_coord_dims=True).first()
    last_time_groups = ds.time.groupby(groups, restore_coord_dims=True).last()

    ds_avg["time"] = first_time_groups.values

    time_labels = []
    for i, group in enumerate(np.unique(groups.values)):
        first_time = first_time_groups[i].dt.strftime("%Y").astype(str).values
        end_time = last_time_groups[i].dt.strftime("%Y").astype(str).values

        if first_time == end_time:
            time_labels.append(f"{first_time}")
        else:
            time_labels.append(f"{first_time}—{end_time}")
    return ds_avg, time_labels


def average_over_months(
    ds: xr.Dataset,
    N: int = 1,
) -> Tuple[xr.Dataset, List[str]]:
    """
    Average a dataset over custom monthly intervals defined by N.

    This function groups time values in `ds` by custom monthly intervals (e.g., every N months)
    and computes the mean for each period. It also generates labels for each group based on the
    first and last months in each period.

    Args:
        ds (xarray.Dataset):
            Dataset with a "time" dimension.
        N (int):
            Length of the custom period in months (default is 1, for monthly averages).

    Returns:
        ds_avg (xarray.Dataset): Dataset averaged over the specified months.
        time_labels (list of str): Labels for each period (e.g., "2020-01", "2020-03—2020-05").
    """

    groups = (ds.time.dt.year * 12 + ds.time.dt.month - 1) // N
    ds_avg = ds.groupby(groups, restore_coord_dims=True).mean(dim="time")
    ds_avg = ds_avg.rename({"group": "time"})

    first_time_groups = ds.time.groupby(groups, restore_coord_dims=True).first()
    last_time_groups = ds.time.groupby(groups, restore_coord_dims=True).last()

    ds_avg["time"] = first_time_groups.values

    time_labels = []
    for i, group in enumerate(np.unique(groups.values)):
        first_time = first_time_groups[i].dt.strftime("%Y-%m").astype(str).values
        end_time = last_time_groups[i].dt.strftime("%Y-%m").astype(str).values

        if first_time == end_time:
            time_labels.append(f"{first_time}")
        else:
            time_labels.append(f"{first_time}—{end_time}")
    return ds_avg, time_labels


def average_over_period(
    ds: xr.Dataset,
    N: int,
    chop_by: Literal["year", "month", "season"] | List,
) -> Tuple[xr.Dataset, List[str]]:
    """
    Average a dataset over a specified time period or custom intervals.

    This function allows averaging over different time periods such as years,
    months, seasons, or custom-defined intervals provided in `chop_by`.
    It calls appropriate averaging functions based on the value of `chop_by`.

    Args:
        ds (xarray.Dataset):
            Dataset with a "time" dimension.
        N (int):
            Interval length for custom periods (e.g., for months or years).
        chop_by (str, list):
            Defines how the dataset should be chopped.
            Options are: 'year', 'month', 'season', or a list of dates or months.

    Returns:
        ds_avg (xarray.Dataset):
            Dataset averaged over the specified periods.
        time_labels (list of str):
            Labels for each averaged period (e.g., "2020", "2020-03—2020-05").
    """

    if isinstance(chop_by, list):
        # Case where chop_by is a list of dates
        if all(
            isinstance(i, (str, datetime.date, np.datetime64, pd.Timestamp))
            for i in chop_by
        ):
            dates_list = np.array(
                [np.datetime64(pd.to_datetime(date), "ns") for date in chop_by]
            )
            return average_over_dates_list(ds.copy(), dates_list)

        # Case where chop_by is either a list of lists or a list of numbers.
        if all(isinstance(i, (list, int, float)) for i in chop_by):
            months_list = chop_by
            return average_over_months_list(ds.copy(), months_list)

    elif chop_by == "season":
        return average_over_seasons(ds.copy())
    elif chop_by == "year":
        return average_over_years(ds.copy(), N)
    elif chop_by == "month":
        # TODO Check if monthly inversions
        return average_over_months(ds.copy(), N)
    else:
        raise ValueError(
            f"Option {chop_by} for chop_by not implemented. Options are year, month, season, a list of starting dates or a list of month numbers."
        )
