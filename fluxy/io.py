import os
import glob
import xarray as xr
import numpy as np
import json 

from pathlib import Path

from fluxy import config
from fluxy.operators.regions import extract_region_flux
from fluxy.operators.select import slice_flux


# Get the location of this file 
this_file = Path(__file__).parent.parent
configs_dir = this_file / 'configs'



def read_json(
    
):
    ...


def read_flux(data_dir,species,models,s_data,m_data,period_override=None,verbose=True):
    """
    Extracts flux and country flux timeseries from each model.
    
    Args:
        data_dir (str): 
            Path to top data directory.
        species (str): 
            Gas species, e.g. 'ch4'.
        models (list of str): 
            Keys specifying model names, e.g. ['intem','elris']
        s_data (dict of dict):
            Dictionary of species with information for plotting (read from json file).
        m_data (dict of dict):
            Dictionary of inversion runs with filename and plot label (read from json file).
        period_override (list of str) (optional):
            Inversion periods to include, to override the standards in species_info.json.
            Must be the same length as models, e.g. ['monthly',None,'yearly']
        verbose (logical) (optional):
            If True, print execution tracking messages.
                                       
    Returns:
        ds_all (dictionary of datasets): 
            xarray dataset read directly from each model's flux netCDF.
    """
    
    period_all = {}
    
    if period_override != None and len(period_override) != len(models):
        print('ERROR: if using period_override, this list must be the same length as models.')
        return None
    
    for i,m in enumerate(models):
        if period_override is not None:
            if period_override[i] is not None:
                period_all[m] = period_override[i]
            else:
                period_all[m] = s_data[species]["period"]
        else:
            period_all[m] = s_data[species]["period"]
    
    ds_all = {}

    for m in models:
        if verbose: print(f'\nAttempting to read data from {m}')
        
        m0 = m.split('_')[0]
        
        model_dir = m_data[m]["filename"].split('_')[0]

        try:
            filepath = glob.glob(os.path.join(data_dir,model_dir,species,
                                              f'{m_data[m]["filename"]}_{s_data[species]["model_species"][m0]}_{period_all[m]}.nc'))
            if verbose: print(f'Reading data from: {filepath[0]}')
            with xr.open_dataset(filepath[0]) as in_ds:
                ds_all[m] = in_ds
                if verbose: print('Done!')
        except:
            try:
                if (m_data[m]["filename"].split('_')[-1] == 'std*'):
                    alternative_filename = f'{m_data[m]["filename"][0:-5]}_{m0}_obs_{m0}_baseline_optimized'
                    filepath = glob.glob(os.path.join(data_dir,model_dir,species,f'{alternative_filename}_{s_data[species]["model_species"][m0]}_{period_all[m]}.nc'))
                    print(f'Cannot find {m} file for {species}. Reading data from: {filepath[0]}')
                    with xr.open_dataset(filepath[0]) as in_ds:
                        ds_all[m] = in_ds
                    print('Done!')
                else:
                    print(f'\nFailed!')
                    print(f'Cannot find {m} file for {species}. This data will not be included.')
            except:
                print(f'Failed!')
                print(f'Cannot find {m} file for {species}. This data will not be included.')
    
    return ds_all


def read_flux_total_fgases(data_dir,species,models,s_data,m_data,regions,
                           start_date,end_date,period_override=None,apply_pop_scale=True):
    """
    Reads in fluxes from a list of gases and sums/averages totals and uncertainties,
    to produce one dataset which can be used with plotting functions in the rest 
    of the notebook.

    Args:
        data_dir (str): 
            Path to top data directory.
        species (str): 
            'all_hfc' or 'all_pfc'
        models (list of str): 
            Keys specifying model names, e.g. ['intem','elris']
        regions (list of str):
            Region names used to extract fluxes. Only these regions can then be plotted.
        s_data (dict of dict):
            Dictionary of species with information for plotting (read from json file).
        start_date (str):
            Date to slice data from, e.g. '2021-01-01'
        end_date (str):
            Date to slice data to, e.g. '2022-01-01' would include all
            data up to 2021-12-31.
        period_override (list of str) (optional):
            Inversion periods to include, to override the standards in species_info.json.
            Must be the same length as models, e.g. ['monthly',None,'yearly']
                                       
    Returns:
        ds_all (dictionary of datasets): 
            xarray dataset read directly from each model's flux netCDF.
    """
    
    if species == 'all_hfc':
        all_species = ['hfc125','hfc134a','hfc143a','hfc152a','hfc23',
                       'hfc227ea','hfc245fa','hfc32','hfc365mfc'] #,'hfc4310mee']
    elif species == 'all_pfc':
        all_species = ['cf4','pfc116','pfc218','pfc318']
    else:
        raise ValueError('This function can only be used to read total hfc (all_hfc) or total pfc (all_pfc).')
        
       
    if type(start_date) is str:
        print('\nNOTE: Using same start and end date for all models')
        print('If this fails with an error message related to region_time dimensions, check the availablility\n'+
              'of data from all models for all timestamps.\n'+
              'To fix this error, set start_date and end_date as lists with the correct start and end times\nfor each model.')
        start_date = [start_date]*len(models)
        end_date = [end_date]*len(models)

    if period_override == None:
        period_override = [None]*len(all_species)
        
    ds_all = {}
    ds_in = {}
    missing_species = {}

    for m,model in enumerate(models):
        
        #longrun = False
        #if 'longrun' in model:
        #    model = model.split('_')[0]
        #    models[m] = model
        #    longrun = True

        missing_species[model] = []
        m0 = model.split('_')[0]
        
        for s,species in enumerate(all_species):

            
            #dictionary containing datasets for each species, these are then summed/averaged across the time coordinate
            ds_out = {}
            
            #tries to read from standard filename
            try:
                model_read = f'{m0}_{s_data[species]["std_run"][m0]}'
                if 'longrun' in model: model_read = f'{model_read}_longrun'
                
                ds_in[model] = read_flux(data_dir,species,[model_read],s_data,m_data,period_override[s],verbose=False)[model_read]    #edit read_flux so that it searches for correct filename per gas
                ds_in[model] = slice_flux(ds_in,start_date[m],end_date[m],s_data,scale_units=False,species=None)[model]

            except:
                ds_in[model] = None
                if species not in missing_species[model]:
                    missing_species[model].append(species)

            for r,region in enumerate(regions):
                
                try:
                    region_time,region_flux_total_posterior,region_flux_total_prior,\
                    region_flux_total_posterior_lower,region_flux_total_posterior_upper,\
                    region_flux_total_prior_lower,region_flux_total_prior_upper = extract_region_flux(ds_in,model,m0,region,apply_pop_scale,verbose=False)
                    
                    #for percentiles, first convert to upper and lower standard deviations (difference from mean)
                    region_flux_total_posterior_lower = (region_flux_total_posterior-region_flux_total_posterior_lower) * 1e3 * s_data[species]['gwp'] * 1e-12
                    region_flux_total_posterior_upper = (region_flux_total_posterior_upper-region_flux_total_posterior) * 1e3 * s_data[species]['gwp'] * 1e-12
                    region_flux_total_prior_lower = (region_flux_total_prior-region_flux_total_prior_lower) * 1e3 * s_data[species]['gwp'] * 1e-12
                    region_flux_total_prior_upper = (region_flux_total_prior_upper-region_flux_total_prior) * 1e3 * s_data[species]['gwp'] * 1e-12
                    
                    region_flux_total_posterior = region_flux_total_posterior * 1e3 * s_data[species]['gwp'] * 1e-12
                    region_flux_total_prior = region_flux_total_prior * 1e3 * s_data[species]['gwp'] * 1e-12

                    #fix to replace rhime's timestamps, which aren't always in the centre of the inversion period
                    # which breaks the .sum() steps below if trying to include data from missing species
                    if 'rhime' in model:
                        region_time = np.arange(np.datetime64(start_date[m]).astype('datetime64[Y]'),np.datetime64(end_date[m]).astype('datetime64[Y]'),
                                        np.timedelta64(1,'Y')).astype('datetime64[ns]')
                        region_time_extended = np.arange(np.datetime64(start_date[m]).astype('datetime64[Y]'),
                                                        np.datetime64(end_date[m]).astype('datetime64[Y]')+np.timedelta64(1,'Y'),
                                                        np.timedelta64(1,'Y')).astype('datetime64[ns]')
                        time_diff = []
                        for t,test_time in enumerate(region_time):
                            time_diff.append((region_time_extended[t+1] - region_time_extended[t])/2)
                        region_time = region_time + time_diff
                
                except:
                    # create empty set of values for this region and species, so it can be skipped if needed
                    print(f'No {species} {region} fluxes found for {model} check directory paths and netcdf contents.')
                    region_time = np.arange(np.datetime64(start_date[m]).astype('datetime64[Y]'),np.datetime64(end_date[m]).astype('datetime64[Y]'),
                                            np.timedelta64(1,'Y')).astype('datetime64[ns]')
                    region_time_extended = np.arange(np.datetime64(start_date[m]).astype('datetime64[Y]'),
                                                    np.datetime64(end_date[m]).astype('datetime64[Y]')+np.timedelta64(1,'Y'),
                                                    np.timedelta64(1,'Y')).astype('datetime64[ns]')
                    time_diff = []
                    for t,test_time in enumerate(region_time):
                        time_diff.append((region_time_extended[t+1] - region_time_extended[t])/2)
                    region_time = region_time + time_diff
                    
                    region_flux_total_posterior_lower,region_flux_total_posterior_upper = np.ones(region_time.shape)*np.nan,np.ones(region_time.shape)*np.nan
                    region_flux_total_prior_lower,region_flux_total_prior_upper = np.ones(region_time.shape)*np.nan,np.ones(region_time.shape)*np.nan
                    region_flux_total_posterior,region_flux_total_prior = np.ones(region_time.shape)*np.nan,np.ones(region_time.shape)*np.nan
                 
                    if species not in missing_species[model]:
                        missing_species[model].append(species)
                        
                #print(f'{model} {species}')
                #print(region_time)
                 
                if r == 0:
                    country_out = np.array([region])
                    region_flux_total_posterior_out = np.expand_dims(region_flux_total_posterior,axis=1)
                    region_flux_total_prior_out = np.expand_dims(region_flux_total_prior,axis=1)
                    region_flux_total_posterior_lower_out = np.expand_dims(region_flux_total_posterior_lower,axis=1)
                    region_flux_total_posterior_upper_out = np.expand_dims(region_flux_total_posterior_upper,axis=1)
                    region_flux_total_prior_lower_out = np.expand_dims(region_flux_total_prior_lower,axis=1)
                    region_flux_total_prior_upper_out = np.expand_dims(region_flux_total_prior_upper,axis=1)
                else:
                    country_out = np.hstack((country_out,np.array([region])))
                    region_flux_total_posterior_out = np.concatenate((region_flux_total_posterior_out,np.expand_dims(region_flux_total_posterior,axis=1)),axis=1)
                    region_flux_total_prior_out = np.concatenate((region_flux_total_prior_out,np.expand_dims(region_flux_total_prior,axis=1)),axis=1)
                    region_flux_total_posterior_lower_out = np.concatenate((region_flux_total_posterior_lower_out,np.expand_dims(region_flux_total_posterior_lower,axis=1)),axis=1)
                    region_flux_total_posterior_upper_out = np.concatenate((region_flux_total_posterior_upper_out,np.expand_dims(region_flux_total_posterior_upper,axis=1)),axis=1)
                    region_flux_total_prior_lower_out = np.concatenate((region_flux_total_prior_lower_out,np.expand_dims(region_flux_total_prior_lower,axis=1)),axis=1)
                    region_flux_total_prior_upper_out = np.concatenate((region_flux_total_prior_upper_out,np.expand_dims(region_flux_total_prior_upper,axis=1)),axis=1)
                #should be of shape (time,n_country)
                    
            ds_out = xr.Dataset({'region_flux_total_posterior_out':(['region_time','country_out'],region_flux_total_posterior_out),
                                'region_flux_total_prior_out':(['region_time','country_out'],region_flux_total_prior_out),
                                'region_flux_total_posterior_lower_out':(['region_time','country_out'],region_flux_total_posterior_lower_out),
                                'region_flux_total_posterior_upper_out':(['region_time','country_out'],region_flux_total_posterior_upper_out),
                                'region_flux_total_prior_lower_out':(['region_time','country_out'],region_flux_total_prior_lower_out),
                                'region_flux_total_prior_upper_out':(['region_time','country_out'],region_flux_total_prior_upper_out)},
                                coords={'region_time':(['region_time'],region_time),
                                        'country_out':(['country_out'],country_out),
                                        'percentile':(['percentile'],np.array([0.159,0.841]))})
            
            if s == 0:
                ds_out_species_total = ds_out.copy()
            else:
                region_flux_total_posterior_all_species = xr.concat((ds_out_species_total['region_flux_total_posterior_out'],
                                                                    ds_out['region_flux_total_posterior_out']),dim='stack').sum(dim='stack')   #this works when ds_out and ds_out_species_total have different time coordinates
                region_flux_total_prior_all_species = xr.concat((ds_out_species_total['region_flux_total_prior_out'],
                                                                    ds_out['region_flux_total_prior_out']),dim='stack').sum(dim='stack')
                region_flux_total_posterior_lower_all_species = np.sqrt(xr.concat((ds_out_species_total['region_flux_total_posterior_lower_out']**2,
                                                                    ds_out['region_flux_total_posterior_lower_out']**2),dim='stack').sum(dim='stack'))
                region_flux_total_posterior_upper_all_species = np.sqrt(xr.concat((ds_out_species_total['region_flux_total_posterior_upper_out']**2,
                                                                    ds_out['region_flux_total_posterior_upper_out']**2),dim='stack').sum(dim='stack'))
                region_flux_total_prior_lower_all_species = np.sqrt(xr.concat((ds_out_species_total['region_flux_total_prior_lower_out']**2,
                                                                    ds_out['region_flux_total_prior_lower_out']**2),dim='stack').sum(dim='stack'))
                region_flux_total_prior_upper_all_species = np.sqrt(xr.concat((ds_out_species_total['region_flux_total_prior_upper_out']**2,
                                                                    ds_out['region_flux_total_prior_upper_out']**2),dim='stack').sum(dim='stack'))
                
                ds_out_species_total = xr.Dataset({'region_flux_total_posterior_out':(['region_time','country_out'],region_flux_total_posterior_all_species.values),
                                'region_flux_total_prior_out':(['region_time','country_out'],region_flux_total_prior_all_species.values),
                                'region_flux_total_posterior_lower_out':(['region_time','country_out'],region_flux_total_posterior_lower_all_species.values),
                                'region_flux_total_posterior_upper_out':(['region_time','country_out'],region_flux_total_posterior_upper_all_species.values),
                                'region_flux_total_prior_lower_out':(['region_time','country_out'],region_flux_total_prior_lower_all_species.values),
                                'region_flux_total_prior_upper_out':(['region_time','country_out'],region_flux_total_prior_upper_all_species.values)},
                                coords={'region_time':(['region_time'],region_flux_total_prior_all_species['region_time'].values),
                                        'country_out':(['country_out'],country_out)}) 
                
            #print(f"Total = {ds_out_species_total['region_flux_total_posterior_lower_out'].values}")
                
            if m0 == 'intem':
                country_coord_name = 'countrynumber'
            else:
                country_coord_name = 'country'
                
        country_shortnames = []
        for c in ds_out_species_total['country_out'].values:
            try:
                country_shortnames.append(config.countrycodes_dict[c])
            except:
                country_shortnames.append(config.regions_dict_old[c])
                
        ds_all[model] = xr.Dataset({'country_flux_total_prior':(['time',country_coord_name],ds_out_species_total['region_flux_total_prior_out'].values),
                                    'country_flux_total_posterior':(['time',country_coord_name],ds_out_species_total['region_flux_total_posterior_out'].values),
                                    'percentile_country_flux_total_prior':(['time','percentile',country_coord_name],
                                                                        np.concatenate((np.expand_dims((ds_out_species_total['region_flux_total_prior_out'].values-ds_out_species_total['region_flux_total_prior_lower_out'].values),axis=1),
                                                                                    np.expand_dims((ds_out_species_total['region_flux_total_prior_out'].values+ds_out_species_total['region_flux_total_prior_upper_out'].values),axis=1)),axis=1)),
                                    'percentile_country_flux_total_posterior':(['time','percentile',country_coord_name],
                                                                        np.concatenate((np.expand_dims((ds_out_species_total['region_flux_total_posterior_out'].values-ds_out_species_total['region_flux_total_posterior_lower_out'].values),axis=1),
                                                                                    np.expand_dims((ds_out_species_total['region_flux_total_posterior_out'].values+ds_out_species_total['region_flux_total_posterior_upper_out'].values),axis=1)),axis=1))},
                            coords={'time':(['time'],ds_out_species_total['region_time'].values),
                                    country_coord_name:([country_coord_name],np.array(country_shortnames))}) 
    
    missing = []
    for model in models:
        if missing_species[model] != []:
            missing.append(model)
        else:
            print(f'\nAll species succesfully read for {model}!')
            
    for m in missing:
        print(f'\nWARNING: Model {m} is missing species: {missing_species[m]}')

    print('\nTo change the files used as the standard for each HFC/PFC, edit variable std_run in species_info.json')

    return ds_all



def read_mf(data_dir,species,models,s_data,m_data,period_override=None):
    """
    Extracts mole fraction timeseries data from each model.
    Args:
        data_dir (str): 
            Path to top data directory.
        species (str): 
            Gas species, e.g. 'ch4'.
        models (list of str): 
            Keys specifying model names, e.g. ['intem','elris']
        s_data (dict of dict):
            Dictionary of species with information for plotting (read from json file).
        m_data (dict of dict):
            Dictionary of inversion runs with filename and plot label (read from json file).
        period_override (list of str) (optional):
            Inversion periods to include, to override the standards in species_info.json.
            Must be the same length as models, e.g. ['monthly',None,'yearly']
    Returns:
        ds_all (dictionary of datasets): 
            xarray dataset read directly from each model's mole fraction netCDF.
    """

    period_all = {}
    
    if period_override != None and len(period_override) != len(models):
        print('ERROR: if using period_override, this list must be the same length as models.')
        return None
    
    for i,m in enumerate(models):
        if period_override is not None:
            if period_override[i] is not None:
                period_all[m] = period_override[i]
            else:
                period_all[m] = s_data[species]["period"]
        else:
            period_all[m] = s_data[species]["period"]

    ds_all = {}

    for m in models:
        
        m0 = m.split('_')[0]
        model_dir = m_data[m]["filename"].split('_')[0]
        
        print(f'\nAttempting to read data from {m}')
        try:
            filepath = glob.glob(os.path.join(data_dir,model_dir,species,f'{m_data[m]["filename"]}_{s_data[species]["model_species"][m0]}_{period_all[m]}_concentrations.nc'))
            print(f'Reading data from: {filepath[0]}')
            with xr.open_dataset(filepath[0]) as in_ds:
                ds_all[m] = in_ds
            print('Done!')
        except:
            try:
                if (m_data[m]["filename"].split('_')[-1] == 'std*'):
                    alternative_filename = f'{m_data[m]["filename"][0:-5]}_{m0}_obs_{m0}_baseline_optimized'
                    filepath = glob.glob(os.path.join(data_dir,model_dir,species,f'{alternative_filename}_{s_data[species]["model_species"][m0]}_{period_all[m]}_concentrations.nc'))
                    print(f'Cannot find {m} file for {species}. Reading data from: {filepath[0]}')
                    with xr.open_dataset(filepath[0]) as in_ds:
                        ds_all[m] = in_ds
                    print('Done!')
                else:
                    print(f'Cannot find {m} file for {species}.')
            except:
                print(f'Cannot find {m} file for {species}.')
            
    return ds_all


def extract_site_info(sites):
    """
    Uses info from site_info.json to create a dictionary
    of sites with latitude and longitudes.
    """
    
    site_info_filename = configs_dir / 'site_info.json'

    with open(site_info_filename, "r") as f:
        site_data = json.load(f)
        
    site_info = {}
    
    for s in sites:
        site_info[s] = {'latitude':site_data[s][list(site_data[s].keys())[0]]['latitude'],
                        'longitude':site_data[s][list(site_data[s].keys())[0]]['longitude']}
    
    return site_info