import numpy as np
import xarray as xr

from matplotlib.cm import get_cmap

from fluxy.operators.regions import extract_region_inventory_flux


def retrieve_inventories(
    data_dir: str,
    country: str,
    species: str,
    start_date: str,
    end_date: str,
    unit: str,
    s_data: dict[str, dict],
    r_data: dict[str, dict],
    inventory_years: str | list[str] | None,
    inventory_filename: str,
    sectors: str | list[str] = "total",
) -> list[xr.Dataset]:
    """
    Load (in a list) inventories data to be plotted.
    If multiple sectors are asked for, each dataset of the list will have a sector dimension.
    NOTE: Call _retrieve_inventories_sector recursively for each sector.

    Args:
        data_dir: directory which contains the data (should have inside a directory named 'inventory').
        country: Region of interest.
        species: Gas species, e.g. 'ch4'.
        start_date: Start date of the data to plot (used to slice inventory data).
        end_date: End date of the data to plot (used to slice inventory data).
        unit: unit in which the inventory should be converted.
        s_data: Dictionary of species with information for plotting (read from json file).
        r_data: Dictionary with country and region names (read from json file).
        inventory_years: List of inventory data from different years to include. If None, only plots the most recent inventory data.
        inventory_filename: Name of inventory file: {inventory_filename}_{species}_{inventory_year}
        sectors: Emissions sector(s), default 'total'
    Returns:
        inventories_list : list of inventory data to be plotted.

    """

    if not isinstance(inventory_years, list):
        inventory_years = [inventory_years]

    if isinstance(sectors, str):
        return _retrieve_inventories_sector(
            data_dir,
            country,
            species,
            start_date,
            end_date,
            unit,
            s_data,
            r_data,
            inventory_years,
            inventory_filename,
            sectors,
        )

    ds_sectors = {y: list() for y in inventory_years}
    for sector in sectors:
        tmp = _retrieve_inventories_sector(
            data_dir,
            country,
            species,
            start_date,
            end_date,
            unit,
            s_data,
            r_data,
            inventory_years,
            inventory_filename,
            sector,
        )
        for i, y in enumerate(inventory_years):
            ds_sectors[y].append(
                tmp[i].expand_dims(
                    dim={
                        "sector": [
                            sector,
                        ]
                    }
                )
            )

    return [xr.concat(ds_sectors[y], dim="sector") for y in inventory_years]


def _retrieve_inventories_sector(
    data_dir: str,
    country: str,
    species: str,
    start_date: str,
    end_date: str,
    unit: str,
    s_data: dict[str, dict],
    r_data: dict[str, dict],
    inventory_years: list[str] | list[None],
    inventory_filename: str,
    sector: str,
) -> list[xr.Dataset]:
    """
    Load (in a list) inventories data to be plotted.

    Args:
        data_dir: directory which contains the data (should have inside a directory named 'inventory').
        country: Region of interest.
        species: Gas species, e.g. 'ch4'.
        start_date: Start date of the data to plot (used to slice inventory data).
        end_date: End date of the data to plot (used to slice inventory data).
        unit: unit in which the inventory should be converted.
        s_data: Dictionary of species with information for plotting (read from json file).
        r_data: Dictionary with country and region names (read from json file).
        inventory_years: List of inventory data from different years to include. If None, only plots the most recent inventory data.
        inventory_filename: Name of inventory file: {inventory_filename}_{species}_{inventory_year}
        sector: Emissions sector
    Returns:
        inventories_list : list of inventory data to be plotted.

    """

    inventories_list = list()

    inv_cmap = get_cmap("Greys")
    inv_colors = [inv_cmap(i) for i in np.linspace(0.5, 0.9, len(inventory_years))]

    for year, inv_color in zip(inventory_years, inv_colors):
        ds_inv = extract_region_inventory_flux(
            data_dir,
            country,
            species,
            unit,
            s_data,
            r_data,
            inventory_year=year,
            inventory_filename=inventory_filename,
            sector=sector,
        )
        ds_inv.attrs["plot_color"] = inv_color
        inventories_list.append(ds_inv.sel(time=slice(start_date, end_date)))

    return inventories_list
