import xarray as xr


def define_var_plot(
    ds: xr.Dataset,
    var: str | list[str],
    sector: str = 'total'
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
    if not isinstance(var, list):
        var = [var]

    ds_output = xr.Dataset()
    for var_p in var:
        unit_var = f"flux_{sector}_posterior"

        if var_p == "posterior_prior_diff":
            ds_output[var_p] = ds[f"flux_{sector}_posterior"] - ds[f"flux_{sector}_prior"]
        elif var_p == "posterior_mean_diff":
            ds_output[var_p] = ds[f"flux_{sector}_posterior"] - ds[
                f"flux_{sector}_posterior"
            ].mean(dim="time")
        elif var_p == "posterior_prior_diff_inversion_grid":
            ds_output[var_p] = (
                ds[f"flux_{sector}_posterior_inversion_grid"] - ds[f"flux_{sector}_prior"]
            )
        elif var_p == "posterior_mean_diff_inversion_grid":
            ds_output[var_p] = ds[f"flux_{sector}_posterior_inversion_grid"] - ds[
                f"flux_{sector}_posterior_inversion_grid"
            ].mean(dim="time")
        else:
            if var_p not in ds:
                raise ValueError(f"'{var_p}' not found in dataset(s)")
            ds_output[var_p] = ds[var_p]
            unit_var = var_p

        ds_output[var_p].attrs["units"] = ds[unit_var].attrs.get("units")

    ds_output.attrs = ds.attrs

    return ds_output
