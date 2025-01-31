import numpy as np
import xarray as xr

from matplotlib.cm import get_cmap

from fluxy.operators.regions import extract_region_inventory_flux

def derive_inventories(data_dir: str,
                       country: str,
                       species: str,
                       start_date: str,
                       end_date: str,
                       s_data: dict[str,dict],
                       scale_co2eq: bool = False,
                       inventory_years: list[str] | None = None
                       )->list[xr.Dataset]:
    """
    Load (in a list) inventories data to be plotted.
    
    Args:
        data_dir: directory which contains the data (should have inside a directory named 'inventory').
        species: Gas species, e.g. 'ch4'.
        start_date: Start dates of the data to plot (used to slice inventory data).
        end_date: Start dates of the data to plot (used to slice inventory data).
        s_data: Dictionary of species with information for plotting (read from json file).
        scale_co2eq: If True, adapt y-axis label to CO2-eq.
        inventory_years: List of inventory data from different years to include. If None, only plots the most recent inventory data.
    
    Returns:
        inventories_list : list of inventory data to be plotted.

    """
    inventories_list =  list()

    if type(inventory_years) is not list:
        inventory_years = [inventory_years]
    
    inv_cmap = get_cmap('Greys')
    inv_colors = [inv_cmap(i) for i in np.linspace(0.9,0.5,len(inventory_years))]

    for year, inv_color in zip(inventory_years,inv_colors):    
        ds_inv = extract_region_inventory_flux(data_dir,country,species,
                                               s_data,scale_co2eq,
                                               inventory_year=year)
        ds_inv.attrs['plot_color'] = inv_color
        inventories_list.append(ds_inv.sel(time=slice(start_date,end_date)))

    return inventories_list
        