import numpy as np
import xarray as xr
import pandas as pd
import calendar
import datetime
import warnings

from typing import List, Tuple, Literal
from fluxy.plots.utils import print_period, get_frequency


def get_flux_mean(
    data: xr.DataArray,
    season: str = None,
) -> xr.DataArray:
    """
    DEPRECATED: Use `resample_over_period` instead.

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
    warnings.warn(
        "'get_flux_mean' is deprecated and will be removed in a future release. "
        "Please use 'resample_over_period' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return resample_over_period(data, chop_by=season, resample_uncert_correlation=True)[
        0
    ]


def calculate_resampled_flux(
    flux: xr.DataArray,
    groups: xr.DataArray,
) -> xr.DataArray:
    """
    Calculate the average of a flux variable over grouped time intervals.

    Args:
        flux (xr.DataArray):
            DataArray with 'time' dimension to average.
        groups (xr.DataArray):
            DataArray grouping each time step.

    Returns:
        xr.DataArray:
            Flux averaged over each group.
    """
    da = flux.groupby(groups).mean(dim="time", keep_attrs=True)

    return da


def calculate_resampled_uncertainties(
    unc: xr.DataArray,
    groups: xr.DataArray,
    flux: xr.DataArray,
    resample_uncert_correlation: bool = False,
) -> xr.DataArray:
    """
    Compute resampled flux uncertainties using RMSE-like aggregation,
    using the assumption that all periods in the resampled flux average are uncorrelated.

    Args:
        unc (xr.DataArray):
            DataArray with 'percentile' and 'time' dims representing flux uncertainties.
        groups (xr.DataArray):
            DataArray grouping each time step.
        flux (xr.DataArray):
            Flux DataArray corresponding to the flux uncertainties.
        resample_uncert_correlation (bool):
            If True, uncertainties are averaged directly over groups.
            If False, uncertainties are calculated as RMSE-like aggregation.

    Returns:
        xr.DataArray:
            Resampled flux uncertainties, with grouped time dims.
    """
    if resample_uncert_correlation:
        da = unc.groupby(groups).mean(dim="time", keep_attrs=True)
    else:
        p0 = unc.isel(percentile=0)
        p1 = unc.isel(percentile=1)

        count = flux.groupby(groups).count(dim="time")

        lower = np.sqrt(((flux - p0) ** 2).groupby(groups).sum(dim="time")) / count

        upper = np.sqrt(((p1 - flux) ** 2).groupby(groups).sum(dim="time")) / count

        flux_resampled = calculate_resampled_flux(flux, groups)

        da = xr.concat(
            [flux_resampled - lower, flux_resampled + upper], dim="percentile"
        )
        da.attrs = unc.attrs
    return da


def group_sites(
    sites: xr.DataArray,
    groups: xr.DataArray,
) -> xr.DataArray:
    """
    Resample the 'sites' variable by checking for presence (1) in any time step within a group.

    Args:
        sites (xr.DataArray):
            Binary indicator (e.g. 0 or 1) with dims ('time', 'platform').
        groups (xr.DataArray):
            DataArray grouping each time step.

    Returns:
        xr.DataArray:
            Aggregated binary site indicator per group.
    """
    da = sites.groupby(groups).any(dim="time", keep_attrs=True).astype(int)
    return da


def calculate_resampled_dataset(
    ds: xr.Dataset,
    groups: xr.DataArray,
    resample_uncert_correlation: bool = False,
) -> xr.Dataset:
    """
    Resample a dataset over custom time groupings with handling of special variable types.

    - Non time-dependent or non-numerical variables are removed.
    - Variables with 'percentile' in dims are aggregated with uncertainty propagation.
    - The 'sites' variable is set to 1 if any entry in the group is 1.
    - Other variables are averaged over time.

    Args:
        ds (xr.Dataset):
            Input dataset with time-dependent variables.
        groups:
            Grouping labels corresponding to each time step.
        resample_uncert_correlation (bool):
            If True, uncertainties are averaged directly over groups.
            If False, uncertainties are calculated as RMSE-like aggregation.

    Returns:
        xr.Dataset: Dataset with resampled variables.
    """

    output_vars = {}
    for var in ds.data_vars:
        da = ds[var]
        dims = set(da.dims)

        if "time" not in dims or not np.issubdtype(da.dtype, np.number):
            continue

        elif "percentile" in dims:
            base_var = var.replace("percentile_", "")
            output_vars[var] = calculate_resampled_uncertainties(
                da, groups, ds[base_var], resample_uncert_correlation
            )

        elif var == "sites" and dims == {"time", "platform"}:
            output_vars[var] = group_sites(da, groups)

        else:
            output_vars[var] = calculate_resampled_flux(da, groups)

    return xr.Dataset(output_vars)


def average_over_dates_list(
    ds: xr.Dataset,
    dates_list: List[str],
) -> Tuple[xr.Dataset, List[str]]:
    """
    DEPRECATED: Use `resample_over_dates_list` instead.

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
    warnings.warn(
        "'average_over_dates_list' is deprecated and will be removed in a future release. "
        "Please use 'resample_over_dates_list' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return resample_over_dates_list(ds, dates_list, resample_uncert_correlation=True)


def resample_over_dates_list(
    ds: xr.Dataset,
    dates_list: List[str],
    resample_uncert_correlation: bool = False,
) -> Tuple[xr.Dataset, List[str]]:
    """
    Resample a dataset over custom time intervals defined in `dates_list`.
    Time labels indicating the date range of each resampled period are also generated.

    Args:
        ds (xarray.Dataset):
            Dataset with a "time" dimension.
        dates_list (list of datetime-like):
            Boundaries for resampling periods (e.g., ['2018-01-01', '2020-01-01']).
        resample_uncert_correlation (bool):
            If True, uncertainties are averaged directly over groups.
            If False, uncertainties are calculated as RMSE-like aggregation.

    Returns:
        ds_resampled (xarray.Dataset):
            Dataset resampled over the defined time intervals.
        time_labels (list of str):
            Labels for each resampled period (format: "YYYY/MM—YYYY/MM").
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

    # Resample dataset
    ds_resampled = calculate_resampled_dataset(
        ds, groups_da, resample_uncert_correlation
    )
    ds_resampled = ds_resampled.rename({"group": "time"})

    # Create labels based on the first and last date in each group
    time_labels = []
    for group in np.unique(groups_da):
        group_times = ds.time.values[groups_da == group]  # Get times in this group
        first_time = pd.to_datetime(group_times.min())
        last_time = pd.to_datetime(group_times.max())
        time_labels.append(
            f"{first_time.strftime('%Y/%m')}—{last_time.strftime('%Y/%m')}"
        )

    return ds_resampled, time_labels


def average_over_months_list(
    ds: xr.Dataset,
    months_list: List,
) -> Tuple[xr.Dataset, List[str]]:
    """
    DEPRECATED: Use `resample_over_months_list` instead.
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
    warnings.warn(
        "'average_over_months_list' is deprecated and will be removed in a future release. "
        "Please use 'resample_over_months_list' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return resample_over_months_list(ds, months_list, resample_uncert_correlation=True)


def resample_over_months_list(
    ds: xr.Dataset,
    months_list: List,
    resample_uncert_correlation: bool = False,
) -> Tuple[xr.Dataset, List[str]]:
    """
    Resample a dataset over specified months in `months_list`.
    It also generates labels for each group based on the months included.

    Args:
        ds (xarray.Dataset):
            Dataset with a "time" dimension.
        months_list (list of lists or ints):
            List of months (or month groups) to resample (e.g., [[7,8]]).
        resample_uncert_correlation (bool):
            If True, uncertainties are averaged directly over groups.
            If False, uncertainties are calculated as RMSE-like aggregation.

    Returns:
        ds_resampled (xarray.Dataset):
            Dataset resampled over the specified months.
        time_labels (list of str):
            Labels for each resampled period (e.g., "Jan—Mar").
    """
    # Define groupings
    groups = np.full(ds.time.shape, np.nan)
    for i, months in enumerate(months_list, start=1):
        groups[np.isin(ds.time.dt.month, months)] = i  # Assign group i

    unique_groups = np.unique(groups[~np.isnan(groups)])
    if len(months_list) != len(unique_groups):
        raise ValueError("Some months are missing.")
    groups_da = xr.DataArray(groups, coords={"time": ds.time})

    # Resample dataset
    ds_resampled = calculate_resampled_dataset(
        ds, groups_da, resample_uncert_correlation
    )
    ds_resampled = ds_resampled.rename({"group": "time"})

    # Make time labels
    time_labels = []
    for i, group in enumerate(unique_groups):
        if isinstance(months_list[i], list):
            time_labels.append("—".join(calendar.month_abbr[m] for m in months_list[i]))
        else:
            time_labels.append(calendar.month_abbr[months_list[i]])

    return ds_resampled, time_labels


def average_over_seasons(
    ds: xr.Dataset,
) -> Tuple[xr.Dataset, List[str]]:
    """
    DEPRECATED: Use `resample_over_seasons` instead.
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
    warnings.warn(
        "'average_over_seasons' is deprecated and will be removed in a future release. "
        "Please use 'resample_over_seasons' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return resample_over_seasons(ds, resample_uncert_correlation=True)


def resample_over_seasons(
    ds: xr.Dataset,
    season: Literal["DJF", "MAM", "JJA", "SON"] = None,
    resample_uncert_correlation: bool = False,
) -> Tuple[xr.Dataset, List[str]]:
    """
    Resample a dataset over seasons.
    It also generates labels for each season.

    Args:
        ds (xarray.Dataset):
            Dataset with a "time" dimension.
        season (str, optional):
            The season to resample ds over.
        resample_uncert_correlation (bool):
            If True, uncertainties are averaged directly over groups.
            If False, uncertainties are calculated as RMSE-like aggregation.


    Returns:
        ds_resampled (xarray.Dataset):
            Dataset resampled over seasons or a given season.
        time_labels (list of str):
            Labels for each season (e.g., "Dec - Feb").
    """
    # Define groupings
    groups = ds.time.dt.season

    # Resample dataset
    ds_resampled = calculate_resampled_dataset(ds, groups, resample_uncert_correlation)
    ds_resampled = ds_resampled.rename({"season": "time"})

    desired_order = ["DJF", "MAM", "JJA", "SON"]  # desired season order
    ordered_seasons = [s for s in desired_order if s in ds_resampled.time.values]
    ds_resampled = ds_resampled.sel(time=ordered_seasons)

    # Make time labels
    season_labels = {
        "DJF": "Dec - Feb",
        "MAM": "Mar - May",
        "JJA": "Jun - Aug",
        "SON": "Sep - Nov",
    }
    time_labels = [season_labels[s] for s in ordered_seasons]

    if season is not None:
        ds_resampled = ds_resampled.sel(time=season)
        if "sites" in ds_resampled.data_vars:
            ds_resampled["sites"] = ds_resampled["sites"].expand_dims(time=[season])

        freq = get_frequency(ds)
        time_labels = print_period(ds, freq, season)
        ds_resampled.attrs["time_label"] = time_labels  # needed for print_cbar_label
    return ds_resampled, time_labels


def resample_over_whole_period(
    ds: xr.Dataset,
    resample_uncert_correlation: bool = False,
) -> Tuple[xr.Dataset, List[str]]:
    """
    Resample a dataset over time for the entire period.
    It also generates a label for the period.

    Args:
        ds (xarray.Dataset):
            Dataset with a "time" dimension.
        resample_uncert_correlation (bool):
            If True, uncertainties are averaged.
            If False, uncertainties are calculated as RMSE-like aggregation.

    Returns:
        ds_resampled (xarray.Dataset):
            Dataset resampled over the entire period.
        time_labels (list of str):
            Label of the entire period (e.g., "2020", "2020—2022").
    """

    # Define groupings
    groups = xr.DataArray(np.zeros(len(ds.time), dtype=int), dims="time")

    # Resample dataset
    ds_resampled = calculate_resampled_dataset(ds, groups, resample_uncert_correlation)
    ds_resampled = ds_resampled.isel(group=0, drop=True)
    if "sites" in ds_resampled.data_vars:
        ds_resampled["sites"] = ds_resampled["sites"].expand_dims(
            time=[ds.time.min().values]
        )

    # Make time label
    freq = get_frequency(ds)
    time_labels = print_period(ds, freq)
    ds_resampled.attrs["time_label"] = time_labels  # needed for print_cbar_label

    return ds_resampled, time_labels


def average_over_years(
    ds: xr.Dataset,
    N: int = 1,
) -> Tuple[xr.Dataset, List[str]]:
    """
    DEPRECATED: Use `resample_over_years` instead.
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
    warnings.warn(
        "'average_over_years' is deprecated and will be removed in a future release. "
        "Please use 'resample_over_years' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return resample_over_years(ds, N, resample_uncert_correlation=True)


def resample_over_years(
    ds: xr.Dataset,
    N: int = 1,
    resample_uncert_correlation: bool = False,
) -> Tuple[xr.Dataset, List[str]]:
    """
    Resample a dataset over years or custom yearly intervals defined by N.
    It also generates labels for each group based on the first and last years in each period.

    Args:
        ds (xarray.Dataset):
            Dataset with a "time" dimension.
        N (int):
            Length of the custom period in years (default is 1, for yearly resamples).
        resample_uncert_correlation (bool):
            If True, uncertainties are averaged directly over groups.
            If False, uncertainties are calculated as RMSE-like aggregation.

    Returns:
        ds_resampled (xarray.Dataset):
            Dataset resampled over the specified periods.
        time_labels (list of str):
            Labels for each period (e.g., "2020", "2020—2022").
    """

    # Define groupings
    groups = (ds.time.dt.year - ds.time.dt.year.min()) // N
    group_labels = np.unique(groups)

    # Get first and last timestamps per group
    first_time_groups = ds.time.groupby(groups, restore_coord_dims=True).first()
    last_time_groups = ds.time.groupby(groups, restore_coord_dims=True).last()

    # Make time labels
    time_labels = []
    for first, last in zip(first_time_groups, last_time_groups):
        y1 = first.dt.strftime("%Y").item()
        y2 = last.dt.strftime("%Y").item()
        time_labels.append(y1 if y1 == y2 else f"{y1}—{y2}")

    # Resample dataset
    ds_resampled = calculate_resampled_dataset(ds, groups, resample_uncert_correlation)
    ds_resampled = ds_resampled.rename({"year": "time"})
    ds_resampled["time"] = first_time_groups.values
    ds_resampled["time"].attrs = ds["time"].attrs

    return ds_resampled, time_labels


def average_over_months(
    ds: xr.Dataset,
    N: int = 1,
) -> Tuple[xr.Dataset, List[str]]:
    """
    DEPRECATED: Use `resample_over_months` instead.
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
    warnings.warn(
        "'average_over_months' is deprecated and will be removed in a future release. "
        "Please use 'resample_over_months' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return resample_over_months(ds, N, resample_uncert_correlation=True)


def resample_over_months(
    ds: xr.Dataset,
    N: int = 1,
    resample_uncert_correlation: bool = False,
) -> Tuple[xr.Dataset, List[str]]:
    """
    Resample a dataset over custom monthly intervals defined by N.

    This function groups time values in `ds` by custom monthly intervals (e.g., every N months)
    It also generates labels for each group based on the first and last years in each period.

    Args:
        ds (xarray.Dataset):
            Dataset with a "time" dimension.
        N (int):
            Length of the custom period in months (default is 1, for monthly resamples).
        resample_uncert_correlation (bool):
            If True, uncertainties are averaged directly over groups.
            If False, uncertainties are calculated as RMSE-like aggregation.

    Returns:
        ds_resampled (xarray.Dataset):
            Dataset resampled over the specified months.
        time_labels (list of str):
            Labels for each period (e.g., "2020/01", "2020/03—2020/05").
    """

    # Define groupings
    ref_month_index = (
        ds.time.dt.year.min() * 12 + ds.time.dt.month[ds.time.dt.year.argmin()] - 1
    )
    groups = ((ds.time.dt.year * 12 + ds.time.dt.month - 1) - ref_month_index) // N
    group_labels = np.unique(groups)

    # Get first and last timestamps per group
    first_time_groups = ds.time.groupby(groups, restore_coord_dims=True).first()
    last_time_groups = ds.time.groupby(groups, restore_coord_dims=True).last()

    # Make time labels
    time_labels = []
    for first, last in zip(first_time_groups, last_time_groups):
        y1 = first.dt.strftime("%Y/%m").item()
        y2 = last.dt.strftime("%Y/%m").item()
        time_labels.append(y1 if y1 == y2 else f"{y1}—{y2}")

    # Resample dataset
    ds_resampled = calculate_resampled_dataset(ds, groups, resample_uncert_correlation)
    ds_resampled = ds_resampled.rename({"group": "time"})
    ds_resampled["time"] = first_time_groups.values
    ds_resampled["time"].attrs = ds["time"].attrs

    return ds_resampled, time_labels


def average_over_period(
    ds: xr.Dataset,
    N: int = 1,
    chop_by: Literal["year", "month", "season"] | List = "year",
) -> Tuple[xr.Dataset, List[str]]:
    """
    DEPRECATED: Use `resample_over_period` instead.
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
    warnings.warn(
        "'average_over_period' is deprecated and will be removed in a future release. "
        "Please use 'resample_over_period' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return resample_over_period(ds, N, chop_by, resample_uncert_correlation=True)


def resample_over_period(
    ds: xr.Dataset,
    N: int = 1,
    chop_by: (
        Literal["year", "month", "season"] | List | Literal["DJF", "MAM", "JJA", "SON"]
    ) = None,
    resample_uncert_correlation: bool = False,
) -> Tuple[xr.Dataset, List[str]]:
    """
    Resample a dataset over a specified time period or custom intervals.

    This function allows resampling over different time periods such as years,
    months, seasons, the entire period, or custom-defined intervals provided in `chop_by`.
    It calls appropriate resampling functions based on the value of `chop_by`.

    Args:
        ds (xarray.Dataset):
            Dataset with a "time" dimension.
        N (int):
            Interval length for custom periods (e.g., for months or years).
        chop_by (str, list):
            Defines how the dataset should be chopped.
            Options are: 'year', 'month', 'season', None, a list of dates or months, or a season.
        resample_uncert_correlation (bool):
            If True, uncertainties are averaged directly over groups.
            If False, uncertainties are calculated as RMSE-like aggregation.

    Returns:
        ds_avg (xarray.Dataset):
            Dataset resampled over the specified periods.
        time_labels (list of str):
            Labels for each resampled period (e.g., "2020", "2020/03—2020/05").
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
            return resample_over_dates_list(
                ds.copy(), dates_list, resample_uncert_correlation
            )

        # Case where chop_by is either a list of lists or a list of numbers.
        if all(isinstance(i, (list, int, float)) for i in chop_by):
            months_list = chop_by
            return resample_over_months_list(
                ds.copy(), months_list, resample_uncert_correlation
            )
    elif chop_by in ["DJF", "MAM", "JJA", "SON"]:
        return resample_over_seasons(
            ds.copy(),
            season=chop_by,
            resample_uncert_correlation=resample_uncert_correlation,
        )
    elif chop_by == "season":
        return resample_over_seasons(
            ds.copy(), resample_uncert_correlation=resample_uncert_correlation
        )
    elif chop_by is None:
        return resample_over_whole_period(ds.copy(), resample_uncert_correlation)
    elif chop_by == "year":
        return resample_over_years(ds.copy(), N, resample_uncert_correlation)
    elif chop_by == "month":
        return resample_over_months(ds.copy(), N, resample_uncert_correlation)
    else:
        raise ValueError(
            f"Option {chop_by} for chop_by not implemented. Options are year, month, season, a list of starting dates or a list of month numbers."
        )
