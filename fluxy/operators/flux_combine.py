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

    ds_combined = xr.concat(
        ds_to_combined_aligned, "model", combine_attrs="drop_conflicts"
    )

    ds_output = xr.Dataset(
        {
            "posterior": ds_combined["posterior"].mean(dim="model"),
            "prior": ds_combined["prior"].mean(dim="model"),
            "posterior_lower": ds_combined["posterior_lower"].min(dim="model"),
            "posterior_upper": ds_combined["posterior_upper"].max(dim="model"),
            "prior_lower": ds_combined["prior_lower"].min(dim="model"),
            "prior_upper": ds_combined["prior_upper"].max(dim="model"),
        }
    )

    ds_output.attrs = ds_combined.attrs

    return {"combined": ds_output}


def combine_map_dataset(ds_all: dict[str, xr.Dataset]) -> dict[str, xr.Dataset]:
    """
    Combine multiple xarray datasets along the 'model' dimension and return the mean dataset.

    Args:
        ds_all: xarray datasets of fluxes indexed by model names.

    Returns:
        A dictionary with a single key 'combined', containing the mean of all datasets along the 'model' dimension.
    """

    models = list(ds_all.keys())
    ds_list = list(ds_all.values())

    ds_dict = {
        "combined": xr.concat(ds_list, dim="model", combine_attrs="override").mean(
            dim="model", keep_attrs=True
        )
    }

    return ds_dict
