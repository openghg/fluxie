import numpy as np

from matplotlib.cm import get_cmap

from fluxy.operators.regions import extract_region_inventory_flux

def derive_inventories(data_dir,country,species,start_date,end_date,
                       s_data,scale_co2eq,inventory_years):
    inventories_list =  list()

    if type(inventory_years) is not list:
        inventory_years = [inventory_years]
    
    inv_cmap = get_cmap('Greys')
    inv_colors = [inv_cmap(i) for i in np.linspace(0.9,0.5,len(inventory_years))]

    for year, inv_color in zip(inventory_years,inv_colors):    
        ds_inv = extract_region_inventory_flux(country,data_dir,species,
                                               s_data,scale_co2eq,
                                               inventory_year=year)
        ds_inv.attrs['plot_color'] = inv_color
        inventories_list.append(ds_inv.sel(time=slice(start_date,end_date)))

    return inventories_list
        