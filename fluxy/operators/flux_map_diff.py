import xarray as xr


def define_var_plot(
    ds: xr.Dataset,
    var: str,
) -> xr.DataArray:
    """
    Define the variable to be plotted based on the specified `var` string.

    This function returns the appropriate variable from the dataset `ds` based on
    the value of `var`. It supports predefined differences between flux variables
    or directly returns a variable from the dataset.

    Args:
        ds (xarray.Dataset):
            The input dataset containing various flux variables.
        var (str):
            The variable name or difference type to be plotted. Options for difference include:
                   'posterior_prior_diff', 'posterior_mean_diff',
                   'posterior_prior_diff_inversion_grid', 'posterior_mean_diff_inversion_grid'.

    Returns:
        var_plot (xarray.DataArray):
            The variable or computed difference to be plotted.
    """

    if var == "posterior_prior_diff":
        var_plot = ds["flux_total_posterior"] - ds["flux_total_prior"]
    elif var == "posterior_mean_diff":
        var_plot = ds["flux_total_posterior"] - ds["flux_total_posterior"].mean(
            dim="time"
        )
    elif var == "posterior_prior_diff_inversion_grid":
        var_plot = ds["flux_total_posterior_inversion_grid"] - ds["flux_total_prior"]
    elif var == "posterior_mean_diff_inversion_grid":
        var_plot = ds["flux_total_posterior_inversion_grid"] - ds[
            "flux_total_posterior_inversion_grid"
        ].mean(dim="time")
    else:
        if var not in ds:
            raise ValueError(f"'{var}' not found in dataset(s)")
        var_plot = ds[var]

    return var_plot


def make_diff_ds(
    ds1: xr.Dataset,
    ds2: xr.Dataset,
):
    """
    Create a difference xarray.Dataset between two model datasets.

    Args:
        ds1 (xr.Dataset): 
            First model dataset.
        ds2 (xr.Dataset): 
            Second model dataset.

    Returns:
        xr.Dataset: A new xarray.Dataset containing the computed differences or combinations for the supported variables.
    """

    diff = {}

    for var in ds1.data_vars:

        v1 = ds1[var]
        v2 = ds2[var]

        dims = v1.dims

        if set(dims) == {"time", "latitude", "longitude"}:
            diff[var] = v1 - v2

        elif set(dims) == {"time", "percentile", "latitude", "longitude"}:

            min_p0 = xr.ufuncs.minimum(v1.sel(percentile=v1.percentile[0]), v2.sel(percentile=v2.percentile[0]))
            max_p1 = xr.ufuncs.maximum(v1.sel(percentile=v1.percentile[1]), v2.sel(percentile=v2.percentile[1]))

            diff[var] = xr.concat([min_p0, max_p1], dim="percentile")
        
        elif var == "sites" and set(dims) == {"time", "platform"}:
            sites1, sites2 = xr.align(ds1[var], ds2[var], join="outer", fill_value=0)
            diff["sites"] = xr.where((sites1 == 1) | (sites2 == 1), 1, 0)

        else:
            logger.info(f"Variable '{var}' with dims {dims} not processed.")

        diff[var].attrs = v1.attrs # Copy attributes from ds1

    diff = xr.Dataset(diff)
    diff.attrs["frequency"] = ds1.attrs.get("frequency", "")

    return diff
