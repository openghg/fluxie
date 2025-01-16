import glob
import os
import numpy as np
import pandas as pd
import xarray as xr
from fluxy import config

def extract_region_flux(ds,m,country,verbose=True):
    """
    Finds the index of a chosen region name and extracts the country flux
    variables for this region.
    Either extracts values directly from the dataset (if this region definition
    exists in the file) or calculates values by taking the sum of smaller regions
    (if this region definition does not exist in the file).
    """
    #########################################################################################
    # To be move to read_flux
    m0 = m.split('_')[0]
    
    if m0 == 'elris':

        ds['country'] = ds['country'].astype('str')
        ds = ds.set_index(countrynumber='country').rename({'countrynumber':'country'})

    elif m0 == 'intem':
        
        ds = ds.rename({'countrynumber':'country'})

        if 'BEL' not in ds.country and 'LUX'  not in ds.country:
            if verbose: print(f'\nNOTE: InTEM does not estimate separate BELGIUM emissions.')
            if verbose: print(f'So a population ratio of {config.bel_pop_r} is being used to scale InTEM\'s total BELGIUM+LUXEMBOURG estimate.\n')

            r = config.bel_pop_r

            variables_with_country = [var for var in ds.data_vars if 'country' in ds[var].dims]
            numerical_vars = [var for var in variables_with_country if np.issubdtype(ds[var].dtype, np.number) and var != 'country_fraction']

            ds_bel = r * ds[numerical_vars].sel(country='BEL-LUX')
            ds_lux = (1-r) * ds[numerical_vars].sel(country='BEL-LUX')

            del ds_bel['country']
            del ds_lux['country']

            ds_bel['countryname'] = xr.DataArray(data = ['BELGIUM',]*ds_bel.time.size,
                                                 dims = ['time',],
                                                 coords = {'time':ds_bel.time},
                                                 attrs = ds.countryname.attrs)
            ds_lux['countryname'] = xr.DataArray(data = ['LUXEMBOURG',]*ds_lux.time.size,
                                                 dims = ['time',],
                                                 coords = {'time':ds_lux.time},
                                                 attrs = ds.countryname.attrs)
            
            ds_bellux = xr.concat([ds_bel, ds_lux], pd.Index(['BEL','LUX'], name='country'))
            ds = xr.merge([ds,ds_bellux])

    elif m0 == 'rhime':

        for k,v in config.countrycodes_dict.items():
            ds['country'] = ds['country'].str.replace(k,v)

    elif m0 == 'flexinvert':

        ds['percentile_country_flux_total_posterior'] = xr.concat([ds_tmp['country_flux_total_posterior']
                                                                   - ds_tmp['country_flux_error_posterior'],
                                                                   ds_tmp['country_flux_total_posterior']
                                                                   + ds_tmp['country_flux_error_posterior']],
                                                                   pd.Index([0,1], name = 'percentile'))
        
        ds['percentile_country_flux_total_prior'] = xr.concat([ds_tmp['country_flux_total_prior']
                                                               - ds_tmp['country_flux_error_prior'],
                                                               ds_tmp['country_flux_total_prior']
                                                               + ds_tmp['country_flux_error_prior']],
                                                               pd.Index([0,1], name = 'percentile'))
    #########################################################################################

    c_key = 'country'
    country_search = config.countrycodes_dict[country]
    #search for existing region names

    available_countries = ds[c_key].values
    
    if country_search not in available_countries and country in config.regions_dict.keys() :
        region_search = config.regions_dict[country]

        if verbose: print(f'{country} emissions are not present in {m}. Considering covariance matrix and sum of individual countries: {region_search}.')

        country_list = region_search.split('-')
        ds_tmp = ds.sel({'country':country_list})

        ds_tmp['region_flux_total_posterior'] = ds_tmp.country_flux_total_posterior.sum(dim='country')
        ds_tmp['region_flux_total_prior'] = ds_tmp.country_flux_total_prior.sum(dim='country')

        ds_tmp['sigma_region_flux_total_prior'] = np.sqrt(((ds.country_flux_total_prior-
                                                            ds.percentile_country_flux_total_prior.isel(percentile=config.model_q_indices[m0][0])
                                                            )**2
                                                           ).sum(dim='country')
                                                          )
        if 'covariance_country_flux_total_posterior' in ds.variables:
            """
            sigma2 = np.zeros(np.shape(ds['covariance_country_flux_total_posterior'])[0])

            for i in range(len(sigma2)):
                sigma2[i] = country_index_vec.dot(ds['covariance_country_flux_total_posterior'].values[i,:,:].dot(country_index_vec))

            sigma_region_flux_total_posterior = np.sqrt(sigma2)
            """
            print(f'WARNING: stuff need to be implemented here (operators/regions.py)')
            ds_tmp['sigma_region_flux_total_posterior'] = np.nan * ds_tmp['region_flux_total_posterior']
            
        else:
            print(f'WARNING: Covariance matrix is not available for {m}. A posteriori uncertainty of {country} emissions will not be plotted.')
            ds_tmp['sigma_region_flux_total_posterior'] = np.nan * ds_tmp['region_flux_total_posterior']
            
        ds_tmp['region_flux_total_posterior_lower'] = ds_tmp['region_flux_total_posterior'] - ds_tmp['sigma_region_flux_total_posterior']
        ds_tmp['region_flux_total_posterior_upper'] = ds_tmp['region_flux_total_posterior'] + ds_tmp['sigma_region_flux_total_posterior']
        ds_tmp['region_flux_total_prior_lower'] = ds_tmp['region_flux_total_prior'] - ds_tmp['sigma_region_flux_total_prior']
        ds_tmp['region_flux_total_prior_upper'] = ds_tmp['region_flux_total_prior'] + ds_tmp['sigma_region_flux_total_prior']
        
    elif country_search in available_countries: 
        ds_tmp = ds.sel({c_key:country_search})
    
        ds_tmp['region_flux_total_posterior'] = ds_tmp['country_flux_total_posterior']
        ds_tmp['region_flux_total_prior'] = ds_tmp['country_flux_total_prior']
        
        ds_tmp['region_flux_total_posterior_lower'] = ds_tmp['percentile_country_flux_total_posterior'].isel(percentile=config.model_q_indices[m0][0])
        ds_tmp['region_flux_total_posterior_upper'] = ds_tmp['percentile_country_flux_total_posterior'].isel(percentile=config.model_q_indices[m0][1])
        ds_tmp['region_flux_total_prior_lower'] = ds_tmp['percentile_country_flux_total_prior'].isel(percentile=config.model_q_indices[m0][0])
        ds_tmp['region_flux_total_prior_upper'] = ds_tmp['percentile_country_flux_total_prior'].isel(percentile=config.model_q_indices[m0][1])

    else:
        raise ValueError(f'{country_search} ({country}) is not available for {m}')
            
    ds_tmp['region_flux_total_posterior_lower'] = ds_tmp['region_flux_total_posterior_lower'].where(ds_tmp['region_flux_total_posterior_lower'] >= 0, 0)
    ds_tmp['region_flux_total_prior_lower'] = ds_tmp['region_flux_total_prior_lower'].where(ds_tmp['region_flux_total_prior_lower'] >= 0, 0)

    return ds_tmp[['region_flux_total_posterior','region_flux_total_prior',
                    'region_flux_total_posterior_lower','region_flux_total_posterior_upper',
                    'region_flux_total_prior_lower','region_flux_total_prior_upper']]


def extract_region_inventory_flux(country,data_dir,species,
                                  s_data,scale_co2eq,start_date,end_date,
                                  inventory_year=None):
    """
    Extracts inventory flux values for regions that exists,
    or calculates total inventory flux values for aggregated regions.
    """
    
    gwp = 1
    scale_factor = s_data[species]["units_scaling"]["intem"]

    # Update scaling factors
    if scale_co2eq and ('all' not in species):
        gwp = s_data[species]["gwp"]
        if (s_data[species]["units_print"] == "G"): #units_print is expected to be either G or T
            scale_factor = scale_factor * 1e3 #Convert to Tg

    if inventory_year == None:
        
        try:
            with xr.open_dataset(sorted(glob.glob(os.path.join(data_dir,'inventory',f'UNFCCC_inventory_{species}_*.nc')))[-1]) as f:
                inv_ds = f
        except:
            with xr.open_dataset(os.path.join(data_dir,'inventory',f'UNFCCC_inventory_{s_data[species]["model_species"]["intem"]}.nc')) as f:
                inv_ds = f
            
    else:
        try:
            with xr.open_dataset(sorted(glob.glob(os.path.join(data_dir,'inventory',f'UNFCCC_inventory_{species}_{inventory_year}.nc')))[-1]) as f:
                    inv_ds = f
        except:
            print(f'No {species} inventory data available for year {inventory_year}.')
            inventory_flux = None
            inventory_time = None

    try:
        #inv_ds = inv_ds.sel(time=slice(start_date,end_date))
        inv_c_index = np.where(inv_ds['country'].values == country)[0][0]
        inventory_flux = inv_ds['inventory'].values[:,inv_c_index]/scale_factor * gwp
        inventory_time = inv_ds.time.values

    except:
        try:
            region_search = config.regions_dict[country]
            country_list = region_search.split('-')

            inv_c_index = [0]*len(country_list)
            inv_c_value = np.zeros(len(inv_ds.time.values))

            print(f'No inventory data available for {country}. Considering sum of individual countries: {region_search}')

            for i,var in enumerate(country_list):
                try:
                    inv_key = [k for k, code in config.countrycodes_dict.items() if code == var]
                    inv_c_index[i] = np.where(inv_ds['country'].values == inv_key[0])[0][0]
                    inv_c_temp = inv_ds['inventory'].values[:,inv_c_index[i]]
                    if np.all(np.isnan(inv_c_temp) == True):
                        inv_c_temp = np.zeros(len(inv_ds.time.values))
                        print(f'WARNING: Inventory data for {inv_key[0]} is NaN. Inventory value for {country} will not include {inv_key[0]} contributions.')

                    inv_c_value = inv_c_value + inv_c_temp
                    inventory_flux = inv_c_value/scale_factor * gwp
                    inventory_time = inv_ds.time.values

                except:
                    try:
                        print(f'WARNING: No inventory data available for {inv_key[0]}. Inventory value for {country} will not include {inv_key[0]} contributions.')
                    except:
                        print(f'ERROR: {var} does not exist in country dictionary!')
                    inventory_flux = None
                    inventory_time = None

        except:
            print(f'No inventory data available for {country}')
            inventory_flux = None
            inventory_time = None
    
    return inventory_flux,inventory_time
