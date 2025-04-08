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
