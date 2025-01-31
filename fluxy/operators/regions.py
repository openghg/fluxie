import glob
import os
import numpy as np
import pandas as pd
import xarray as xr
from fluxy import config

def extract_region_flux(ds_all: dict[str, xr.Dataset],
                        country: str, 
                        verbose: bool =True
                        )-> dict[str, xr.Dataset]:
    """
    Finds the index of a chosen region name and extracts the country flux
    variables for this region.
    Either extracts values directly from the dataset (if this region definition
    exists in the file) or calculates values by taking the sum of smaller regions
    (if this region definition does not exist in the file).
    
    Args:
        ds_all: xarray datasets of fluxes, scaled and sliced between 
            chosen dates.
        country: name of the country to extract.
        verbose: if you s=want lots of message
    
    Returns:
        ds_output: dictionnary of datasets. The dataset variables are :
            - 'region_flux_total_posterior',
            - 'region_flux_total_prior',
            - 'region_flux_total_posterior_lower',
            - 'region_flux_total_posterior_upper',
            - 'region_flux_total_prior_lower',
            - 'region_flux_total_prior_upper'
    """
    ds_output = dict()

    for m,ds in ds_all.items():
        #########################################################################################
        # To be move to read_flux
        m0 = m.split('_')[0]
        min_percentile_index = config.model_q_indices[m0][0]
        max_percentile_index = config.model_q_indices[m0][1]
        
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
                                                                ds.percentile_country_flux_total_prior.isel(percentile=min_percentile_index)
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
            
            ds_tmp['region_flux_total_posterior_lower'] = ds_tmp['percentile_country_flux_total_posterior'].isel(percentile=min_percentile_index)
            ds_tmp['region_flux_total_posterior_upper'] = ds_tmp['percentile_country_flux_total_posterior'].isel(percentile=max_percentile_index)
            ds_tmp['region_flux_total_prior_lower'] = ds_tmp['percentile_country_flux_total_prior'].isel(percentile=min_percentile_index)
            ds_tmp['region_flux_total_prior_upper'] = ds_tmp['percentile_country_flux_total_prior'].isel(percentile=max_percentile_index)

        else:
            raise ValueError(f'{country_search} ({country}) is not available for {m}')
                
        ds_tmp['region_flux_total_posterior_lower'] = ds_tmp['region_flux_total_posterior_lower'].clip(min = 0)
        ds_tmp['region_flux_total_prior_lower'] = ds_tmp['region_flux_total_prior_lower'].clip(min = 0)

        ds_output[m] = ds_tmp[['region_flux_total_posterior','region_flux_total_prior',
                               'region_flux_total_posterior_lower','region_flux_total_posterior_upper',
                               'region_flux_total_prior_lower','region_flux_total_prior_upper']]
    return ds_output


def extract_region_inventory_flux(data_dir: str,
                                  country: str,
                                  species: str,
                                  s_data: dict[str,dict],
                                  scale_co2eq: bool = False
                                  )->xr.Dataset:
    """
    Extracts inventory flux values for regions that exists,
    or calculates total inventory flux values for aggregated regions.
    
    Args:
        data_dir: directory which contains the data (should have inside a directory named 'inventory').
        species: Gas species, e.g. 'ch4'.
        s_data: Dictionary of species with information for plotting (read from json file).
        scale_co2eq: If True, adapt y-axis label to CO2-eq.
        
    Returns:
        dataset with country selected

    """
    
    gwp = 1
    scale_factor = s_data[species]["units_scaling"]["intem"]

    # Update scaling factors
    if scale_co2eq and ('all' not in species):
        gwp = s_data[species]["gwp"]
        if s_data[species]["units_print"] == "G": #units_print is expected to be either G or T
            scale_factor = scale_factor * 1e3 #Convert to Tg

    if inventory_year is not None:
        filepath = os.path.join(data_dir,'inventory',f'UNFCCC_inventory_{species}_{inventory_year}.nc')
    else :
        filelist = sorted(glob.glob(os.path.join(data_dir,'inventory',f'UNFCCC_inventory_{species}_*.nc')))
        if filelist :
            filepath = filelist[-1]
            inventory_year = int(filepath.split('_')[-1].split('.')[0])
        else :
            filepath = os.path.join(data_dir,'inventory',f'UNFCCC_inventory_{s_data[species]["model_species"]["intem"]}.nc')
            inventory_year = None

    inv_ds = xr.open_dataset(filepath)['inventory'] / scale_factor * gwp
    inv_ds.attrs['year'] = inventory_year

    if country in inv_ds['country']:
        return inv_ds.sel(country=country)

    else:
        region_search = config.regions_dict[country]
        country_list = region_search.split('-')
        print(f'No inventory data available for {country}. Considering sum of individual countries: {region_search}')

        country_list_update = [country if country in inv_ds['country'] 
                               else dict(map(reversed, config.countrycodes_dict.items()))[country] 
                               for country in country_list]
        return inv_ds.sel(country=country_list_update).sum(dim='country',keep_attrs=True)