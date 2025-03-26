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
    Returns
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
        }
    )
    return {"combined": ds_output}
