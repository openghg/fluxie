import glob
import os
import numpy as np
import xarray as xr
from fluxy import config

def extract_region_flux(ds,m,country,apply_pop_scale,verbose=True):
    """
    Finds the index of a chosen region name and extracts the country flux
    variables for this region.
    Either extracts values directly from the dataset (if this region definition
    exists in the file) or calculates values by taking the sum of smaller regions
    (if this region definition does not exist in the file).
    """
    m0 = m.split('_')[0]
    
    if m0=='elris':
        ds['country'] = ds['country'].astype('str')
        ds = ds.set_index(countrynumber='country').rename({'countrynumber':'country'})
    elif m0=='intem':
        ds = ds.rename({'countrynumber':'country'})

    c_key = 'country'
    print(m0,country)
    #search for existing region names

    country_list = ds[c_key].values

    if m0 == 'intem' and country == 'BELGIUM' :
        if not apply_pop_scale: raise ValueError("you have to apply popscale with BELGIUM and InTEM emissions as InTEM does not estimate separate BELGIUM emissions.")
        country_search = 'BEL-LUX'
        if verbose: print(f'\nNOTE: InTEM does not estimate separate BELGIUM emissions.')
        if verbose: print(f'So a population ratio of {config.bel_pop_r} is being used to scale InTEM\'s total BELGIUM+LUXEMBOURG estimate.\n')
        r = config.bel_pop_r
    else:
        r = 1

        if country in country_list :
            country_search = country

        elif config.countrycodes_dict[country] in country_list :
            country_search = config.countrycodes_dict[country]

        elif config.regions_dict_old[country] in country_list :
            country_search = config.regions_dict_old[country]

        ds_tmp = ds.sel({c_key:country_search})


        
    ds_tmp['region_flux_total_posterior'] = ds_tmp['country_flux_total_posterior']*r
    ds_tmp['region_flux_total_prior'] = ds_tmp['country_flux_total_prior']*r
    
    if m0 == 'flexinvert':
        ds_tmp['region_flux_total_posterior_lower'] = r * (ds_tmp['country_flux_total_posterior']
                                                            - ds_tmp['country_flux_error_posterior'])
        ds_tmp['region_flux_total_posterior_upper'] = r * (ds_tmp['country_flux_total_posterior']
                                                            + ds_tmp['country_flux_error_posterior'])
        ds_tmp['region_flux_total_prior_lower'] = r * (ds_tmp['country_flux_total_prior']
                                                        - ds_tmp['country_flux_error_prior'])
        ds_tmp['region_flux_total_prior_upper'] = r * (ds_tmp['country_flux_total_prior']
                                                        + ds_tmp['country_flux_error_prior'])
    else:
        ds_tmp['region_flux_total_posterior_lower'] = r * ds_tmp['percentile_country_flux_total_posterior'].isel(percentile=config.model_q_indices[m0][0])
        ds_tmp['region_flux_total_posterior_upper'] = r * ds_tmp['percentile_country_flux_total_posterior'].isel(percentile=config.model_q_indices[m0][1])
        ds_tmp['region_flux_total_prior_lower'] = r * ds_tmp['percentile_country_flux_total_prior'].isel(percentile=config.model_q_indices[m0][0])
        ds_tmp['region_flux_total_prior_upper'] = r * ds_tmp['percentile_country_flux_total_prior'].isel(percentile=config.model_q_indices[m0][1])
        
    ds_tmp['region_flux_total_posterior_lower'] = ds_tmp['region_flux_total_posterior_lower'].where(ds_tmp['region_flux_total_posterior_lower'] >= 0, 0)
    ds_tmp['region_flux_total_prior_lower'] = ds_tmp['region_flux_total_prior_lower'].where(ds_tmp['region_flux_total_prior_lower'] >= 0, 0)
    return ds_tmp[['region_flux_total_posterior','region_flux_total_prior',
                    'region_flux_total_posterior_lower','region_flux_total_posterior_upper',
                    'region_flux_total_prior_lower','region_flux_total_prior_upper']]

    #calculate values for region names that don't exist in the file
    if False:
        
        if True:
            region_search = config.regions_dict[country]
            if verbose: print(f'{country} emissions are not present in {m}. Considering covariance matrix and sum of individual countries: {region_search}.')

            country_list = region_search.split('-')

            country_index_vec = np.zeros(len(ds[c_key]))
            sigma2_region_flux_total_prior = 0
            region_flux_total_posterior = 0
            region_flux_total_prior = 0

            # Compute sum of prior/posterior emissions and prior uncertainty
            for var in country_list:
                if True:
                    country_index = np.where(ds[c_key].values.astype(str) == var)[0][0]
                    country_index_vec[country_index] = 1

                    region_flux_total_posterior = region_flux_total_posterior + ds['country_flux_total_posterior'].values[:,country_index]
                    region_flux_total_prior     = region_flux_total_prior + ds['country_flux_total_prior'].values[:,country_index]

                    sigma_country_prior = ds['country_flux_total_prior'].values[:,country_index] - ds['percentile_country_flux_total_prior'].values[:,config.model_q_indices[m0][0],country_index]
                    sigma2_region_flux_total_prior = sigma2_region_flux_total_prior + sigma_country_prior**2

                else:
                    print(f'WARNING: {var} emissions are not present in {m}. This country will be neglected in {country} emissions.')
                    sigma2_region_flux_total_prior = np.zeros(ds.time.values.shape[0])
                    
            sigma_region_flux_total_prior = np.sqrt(sigma2_region_flux_total_prior)
        
            # Compute posterior uncertainty from covariance matrix
            if 'covariance_country_flux_total_posterior' in ds.variables():
                sigma2 = np.zeros(np.shape(ds['covariance_country_flux_total_posterior'])[0])

                for i in range(len(sigma2)):
                    sigma2[i] = country_index_vec.dot(ds['covariance_country_flux_total_posterior'].values[i,:,:].dot(country_index_vec))

                sigma_region_flux_total_posterior = np.sqrt(sigma2)
            else:
                print(f'WARNING: Covariance matrix is not available for {m}. A posteriori uncertainty of {country} emissions will not be plotted.')
                sigma_region_flux_total_posterior = np.zeros(ds.time.values.shape[0])
                
            region_time = ds.time.values
            region_flux_total_posterior_lower = region_flux_total_posterior - sigma_region_flux_total_posterior
            region_flux_total_posterior_upper = region_flux_total_posterior + sigma_region_flux_total_posterior
            region_flux_total_prior_lower = region_flux_total_prior - sigma_region_flux_total_prior
            region_flux_total_prior_upper = region_flux_total_prior + sigma_region_flux_total_prior

            region_flux_total_posterior_lower[region_flux_total_posterior_lower < 0.] = 0.
            region_flux_total_prior_lower[region_flux_total_prior_lower < 0.] = 0.

        else:
            #print(f'ERROR: Could not find {country} emissions for {m}.')
            print(f'Skipping read in of {m}.')
            
            region_time = None
            region_flux_total_posterior,region_flux_total_prior = None,None
            region_flux_total_posterior_lower,region_flux_total_posterior_upper = None,None
            region_flux_total_prior_lower,region_flux_total_prior_upper = None,None
    
    return (region_time,region_flux_total_posterior,region_flux_total_prior,
            region_flux_total_posterior_lower,region_flux_total_posterior_upper,
            region_flux_total_prior_lower,region_flux_total_prior_upper)


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
