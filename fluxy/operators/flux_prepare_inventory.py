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
    inventory_years: list[str] | None = None,
) -> list[xr.Dataset]:
    """
    Load (in a list) inventories data to be plotted.

    Args:
        data_dir: directory which contains the data (should have inside a directory named 'inventory').
        species: Gas species, e.g. 'ch4'.
        start_date: Start dates of the data to plot (used to slice inventory data).
        end_date: Start dates of the data to plot (used to slice inventory data).
        unit: unit in which the inventory should be converted.
        s_data: Dictionary of species with information for plotting (read from json file).
        r_data: Dictionary with country and region names (read from json file).
        inventory_years: List of inventory data from different years to include. If None, only plots the most recent inventory data.

    Returns:
        inventories_list : list of inventory data to be plotted.

    """
    inventories_list = list()

    if inventory_years is None:
        inventory_years = [None]

    inv_cmap = get_cmap("Greys")
    inv_colors = [inv_cmap(i) for i in np.linspace(0.9, 0.5, len(inventory_years))]

    for year, inv_color in zip(inventory_years, inv_colors):
        ds_inv = extract_region_inventory_flux(
            data_dir, country, species, unit, s_data, r_data, inventory_year=year
        )
        ds_inv.attrs["plot_color"] = inv_color
        inventories_list.append(ds_inv.sel(time=slice(start_date, end_date)))

    return inventories_list
