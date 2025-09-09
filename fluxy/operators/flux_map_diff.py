import xarray as xr
import logging

logger = logging.getLogger(__name__)


def define_var_plot(
    ds: xr.Dataset, var: str | list[str], sector: str = "total"
) -> xr.Dataset:
    """
    Define the variable to be plotted based on the specified `var` string.

    This function returns the appropriate variable from the dataset `ds` based on
    the value of `var`. It supports predefined differences between flux variables
    or directly returns a variable from the dataset.

    Args:
        ds (xarray.Dataset):
            The input dataset containing various flux variables.
        var (str | list[str]):
            The variable name or difference type to be plotted. Options for difference include:
                   'posterior_prior_diff', 'posterior_mean_diff',
                   'posterior_prior_diff_inversion_grid', 'posterior_mean_diff_inversion_grid'.

    Returns:
        var_plot (xarray.Dataset):
            The variable or computed difference to be plotted.
    """
    if not isinstance(var, list):
        var = [var]

    ds_output = xr.Dataset()
    for var_p in var:
        unit_var = f"flux_{sector}_posterior"

        if var_p == "posterior_prior_diff":
            ds_output[var_p] = (
                ds[f"flux_{sector}_posterior"] - ds[f"flux_{sector}_prior"]
            )

        elif var_p == "posterior_mean_diff":
            ds_output[var_p] = ds[f"flux_{sector}_posterior"] - ds[
                f"flux_{sector}_posterior"
            ].mean(dim="time")

        elif var_p == "posterior_prior_diff_inversion_grid":
            if (
                f"flux_{sector}_prior_inversion_grid" in ds
                and f"flux_{sector}_posterior_inversion_grid" in ds
            ):  # ensure both prior and posterior exists for inversion grid
                prior = f"flux_{sector}_prior_inversion_grid"
                posterior = f"flux_{sector}_posterior_inversion_grid"
            else:
                prior = f"flux_{sector}_prior"
                posterior = f"flux_{sector}_posterior"
                logger.warning(
                    f"'flux_{sector}_prior_inversion_grid' not found in dataset(s) (inversion system = {ds.attrs['inversion_system']}), comparison is made on prior grid for this model."
                )
            ds_output[var_p] = ds[posterior] - ds[prior]

        elif var_p == "posterior_mean_diff_inversion_grid":
            ds_output[var_p] = ds[f"flux_{sector}_posterior_inversion_grid"] - ds[
                f"flux_{sector}_posterior_inversion_grid"
            ].mean(dim="time")

        else:
            if var_p in ds:
                var_p_bis = var_p
            else:
                if (
                    "_inversion_grid" in var_p
                ):  # fix if variable doesn't exist on inversion grid
                    var_p_bis = var_p.replace("_inversion_grid", "")
                    logger.warning(
                        f"'{var_p}' not found in dataset(s) (inversion system = {ds.attrs['inversion_system']}), replaced by '{var_p_bis}' for this model."
                    )
                else:
                    raise ValueError(f"'{var_p}' not found in dataset(s)")
            ds_output[var_p] = ds[var_p_bis]
            unit_var = var_p_bis

        ds_output[var_p].attrs["units"] = ds[unit_var].attrs.get("units")

    ds_output.attrs = ds.attrs

    # Add back sites variable if exists
    if "sites" in ds.data_vars:
        ds_output["sites"] = ds["sites"]

    return ds_output


def make_model_diff_ds(
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
            diff[var].attrs = v1.attrs  # Copy attributes from ds1

        elif var == "sites" and set(dims) == {"time", "platform"}:
            sites1, sites2 = xr.align(ds1[var], ds2[var], join="outer", fill_value=0)
            diff["sites"] = xr.where((sites1 == 1) | (sites2 == 1), 1, 0)
            diff["sites"].attrs = v1.attrs  # Copy attributes from ds1

        else:
            logger.info(
                f"Variable '{var}' with dims {dims} not processed."
            )  # eg, not keeping percentiles
    diff = xr.Dataset(diff)
    diff.attrs["frequency"] = ds1.attrs.get("frequency", "")

    return diff
