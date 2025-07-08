import xarray as xr

from fluxy.operators.flux_align_dataset import align_time


def combine_dataset(
    ds_all: dict[str, xr.Dataset], plot_combined: list[bool]
) -> dict[str, xr.Dataset]:
    """
    Args:
        ds_all: xarray datasets of fluxes.
        plot_combined: If True, the model is included in combined average result to be plotted.
             List must be of same size as models, e.g. [False, True, True].
    Returns:
        A dictionnary with 'combined' as key and the combined dataset as value.
    """
    ds_to_combined = [ds for i, ds in enumerate(ds_all.values()) if plot_combined[i]]
    ds_to_combined_aligned = align_time(ds_to_combined)

    ds_combined = xr.concat(ds_to_combined_aligned, "model")

    ds_output = xr.Dataset(
        {
            "posterior": ds_combined["posterior"].mean(dim="model"),
            "prior": ds_combined["prior"].mean(dim="model"),
            "posterior_lower": ds_combined["posterior_lower"].min(dim="model"),
            "posterior_upper": ds_combined["posterior_upper"].max(dim="model"),
            "prior_lower": ds_combined["prior_lower"].min(dim="model"),
            "prior_upper": ds_combined["prior_upper"].max(dim="model")
        }
    )
    return {"combined": ds_output}


def combine_map_dataset(
    ds_all: dict[str, xr.Dataset]
) -> dict[str, xr.Dataset]:
    """
    Combine multiple xarray datasets along the 'model' dimension and return the mean dataset.
    - Variables with "percentile" dimension will be collapsed into min (p0) and max (p1).
    - Variable "sites" with dims (time, platform) will be set to 1 if any entry in models is 1.
    - Other variables with dims (time, ...) will be averaged.

    Args:
        ds_all: xarray datasets of fluxes indexed by model names.

    Returns:
        A dictionary with a single key 'combined', containing the mean/min-max of all datasets along the 'model' dimension.
    """

    models = list(ds_all.keys())
    ds_list = list(ds_all.values())

    ds_combined = xr.concat(ds_list, dim="model", combine_attrs="override")

    for var in ds_combined.data_vars:
        v = ds_combined[var]
        dims = set(v.dims)

        if "percentile" in dims:
            p0 = v.isel(percentile=0).min(dim="model", keep_attrs=True)
            p1 = v.isel(percentile=1).max(dim="model", keep_attrs=True)
            ds_combined[var] = xr.concat([p0, p1], dim="percentile", combine_attrs="override")

        elif var == "sites" and dims == {"time", "platform"}:
            ds_combined[var] = v.any(dim="model", keep_attrs=True).astype(int)

        elif "time" in dims:
            ds_combined[var] = v.mean(dim="model", keep_attrs=True)

    ds_dict = {"combined": ds_combined}

    return ds_dict
