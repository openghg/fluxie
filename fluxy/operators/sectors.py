import numpy as np
import xarray as xr

from fluxy.operators.utils import apply_to_dict_or_single


def sectors_group_from_config_or_dict(
    sector_groups: dict[str, list[str]] | None = None,
    sectors_config: dict[str, str] | None = None,
    species: str | None = None,
) -> dict[str, list[str]]:
    """Group sectors based on a configuration or a dictionary.

    Args:
        sector_groups: A dictionary mapping sector names to their group names.
            If provided, it will be used to group the sectors.
        sectors_config: A dictionary containing sector configuration.
            If provided, it will be used to map old sectors to new ones.
        species: The species name to modify sector groups for.

    Returns:
        A dictionary mapping group names to their sub-sectors.
    """
    if sector_groups is None and sectors_config is None:
        raise ValueError("Either 'sector_groups' or 'sectors_config' must be provided.")

    if sector_groups is not None:
        if sectors_config is not None:
            raise ValueError(
                "Both 'sector_groups' and 'sectors_config' are provided. "
                "Please provide only one of them."
            )
        return sector_groups

    # Deep copy the dict of dict to avoid modifying the original
    sector_groups = {k: v.copy() for k, v in sectors_config["sectors_groups"].items()}

    if (
        "sector_groups_per_species" in sectors_config
        and species in sectors_config["sector_groups_per_species"]
    ):
        # Modify sector groups for specific species
        additional_groups = sectors_config["sector_groups_per_species"][species]
        # If the values are the keys of the other dict, so we need to update
        for key, values in additional_groups.items():
            if key not in sector_groups:
                sector_groups[key] = []
            for val in values:
                new_values = sector_groups.pop(val, values)
                sector_groups[key].extend(new_values)

    return sector_groups


@apply_to_dict_or_single
def group_sectors(
    ds: xr.Dataset,
    sector_groups: dict[str, list[str]] | None = None,
    sectors_config: dict[str, str] | None = None,
) -> xr.Dataset:
    """Group sectors in the dataset based on the provided sector groups.

    One of the two must be provided:
    `sector_groups` or `sectors_config`.

    Args:
        ds: The xarray dataset to group.
        sector_groups: A dictionary mapping sector names to their group names.
        sectors_config: A dictionary containing sector configuration.

    Returns:
        The grouped xarray dataset.
    """
    if "sector" not in ds.dims:
        return ds

    sector_groups = sectors_group_from_config_or_dict(
        sector_groups=sector_groups,
        sectors_config=sectors_config,
        species=ds.attrs.get("species", None),
    )

    vars_with_sector = [var for var in ds.data_vars if "sector" in ds[var].dims]

    old_to_new_sectors = {
        old_sector: new_sector
        for new_sector, old_sectors in sector_groups.items()
        for old_sector in old_sectors
    }

    # Map the dict to the 'sector' dimension
    ds = ds.assign_coords(
        sector=np.vectorize(
            lambda x: old_to_new_sectors.get(x, x),
        )(ds["sector"].values)
    )
    ds_out = ds[vars_with_sector].groupby("sector").sum(dim="sector")
    # Add the variables not with the sector dimension
    ds_out = xr.merge([ds_out, ds.drop_vars(vars_with_sector).drop_dims("sector")])
    return ds_out
