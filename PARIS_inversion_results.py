import xarray as xr
import numpy as np
from pandas import to_datetime
import matplotlib.pyplot as plt
import os
import glob
import math
from matplotlib.dates import YearLocator, MonthLocator
from matplotlib.ticker import NullFormatter, AutoMinorLocator, MultipleLocator
import pprint
import cartopy
from json import load
import inspect
from IPython.utils import io
import sys
import cartopy.crs as ccrs
from matplotlib import colors, ticker
from collections import Counter
import pandas as pd
import socket

model_q_indices = {'intem':[0,1],
                   'rhime':[0,1],
                   'elris':[0,1]}

point_source_dict = {'paris':[2.340430,48.860050],
                     'london':[-0.1278, 51.5074],
                     'nw_england':[-2.796870,53.774820],
                     'edinburgh':[-3.1883, 55.9533],
                     'cardiff':[-3.1791, 51.4816],
                     'belfast':[-5.9302, 54.5973],
                     'birmingham':[-1.8904, 52.4862],
                     'manchester':[-2.2426, 53.4808],
                     'dublin':[-6.2675, 53.3441],
                     'paris':[2.3522, 48.8566],
                     'berlin':[13.4050, 52.5200],
                     'hamburg':[9.993682, 53.551086],
                     'munich':[11.5820, 48.1351],
                     'frankfurt':[8.6821, 50.1109],
                     'cologne':[6.953101, 50.935173],
                     'rome':[12.496366, 41.902782],
                     'milan':[9.1900, 45.4642],
                     'brussels':[4.3517, 50.8503],
                     'bern':[7.4474,46.9480],
                     'marseille':[5.3755,43.2953],
                     'amsterdam':[4.8945, 52.3667]
                     }

countrycodes_dict = {'IRELAND':'IRL',
                     'UK':'GBR',
                     'FRANCE':'FRA',
                     'NETHERLANDS':'NLD',
                     'GERMANY':'DEU',
                     'DENMARK':'DNK',
                     'SWITZERLAND':'CHE',
                     'AUSTRIA':'AUT',
                     'ITALY':'ITA',
                     'BELGIUM': 'BEL',
                     'LUXEMBOURG': 'LUX',
                     'HUNGARY':'HUN',
                     'SWEDEN':'SWE',
                     'POLAND':'POL',
                     'CZECHIA':'CZE',
                     'CROATIA':'HRV',
                     'SLOVAKIA':'SVK',
                     'FINLAND':'FIN',
                     'SLOVENIA':'SVN',
                     'GREECE':'GRC',
                     'SPAIN':'ESP',
                     'PORTUGAL':'PRT',
                     'NORWAY':'NOR',
                     'ENGLAND':'ENG',
                     'NORTHERNIRELAND':'NIR',
                     'WALES':'WAL',
                     'SCOTLAND':'SCO'}

regions_dict = {'BELUX':'BEL-LUX',
                'BENELUX':'BEL-LUX-NLD',
                'CW_EU':'AUT-BEL-CHE-CZE-DEU-ESP-FRA-GBR-HRV-HUN-IRL-ITA-LUX-NLD-POL-PRT-SVK-SVN',
                'EU_GRP2':'AUT-BEL-CHE-DEU-DNK-FRA-GBR-IRL-ITA-LUX-NLD',
                'NW_EU':'BEL-DEU-DNK-FRA-GBR-IRL-LUX-NLD',
                'NW_EU2':'BEL-DEU-FRA-GBR-IRL-LUX-NLD',
                'NW_EU_CONTINENT':'BEL-DEU-FRA-LUX-NLD'}

regions_dict_old = {'CW_EU':'AUT-BEL-CHE-CZE-DEU-ESP-FRA-GBR-HRV-HUN-IRL-ITA-LUX-NLD-POL-PRT-SVK-SVK'}

regions_print = {'SCOTLAND':'Scotland',
                 'ENGLAND':'England',
                 'NORTHERNIRELAND':'Northern Ireland',
                 'WALES':'Wales'}

countrycodes_dict.update(regions_dict)

# population from 2018 to 2023 (at Jan 1 each year)
bel_pop = np.array([11.399,11.455,11.522,11.555,11.618,11.723])
lux_pop = np.array([0.602,0.614,0.626,0.635,0.645,0.661])
bel_pop_r = np.round(np.mean(bel_pop/(bel_pop+lux_pop)),3)

#####################################################################

def initialize_settings(ppt_mode=False):
    """
    Extracts species and models info from json files.
    Defines standard colors for plotting.

    Args:
        ppt_mode (logical) (optional):
            If True, use bigger fonts (ideal for presentation slides)

    Returns:
        s_data (dict of dict):
            Dictionary of species with information for plotting (read from json file).
        m_data (dict of dict):
            Dictionary of inversion runs with filename and plot label (read from json file).
        model_colors (dict of lists):
            Default lists of colors to be used by each model.
        annotate_coords (dict of lists):
            Coordinates to annotate histogram.
    """

    ### read in species info file

    filename = os.path.join(os.getcwd(),'species_info.json')

    if os.path.exists(filename) == False:
        print('ERROR: Cannot find species_info.json file. Check that this exists in the same directory as your notebook.')

    with open(filename, "r") as f:
        s_data = load(f)

    ### read in models info file

    filename = os.path.join(os.getcwd(),'models_info.json')

    if os.path.exists(filename) == False:
        print('ERROR: Cannot find models_info.json file. Check that this exists in the same directory as your notebook.')

    with open(filename, "r") as f:
        m_data = load(f)

    print('NOTE: If plotting units or scales look odd, edit species_info.json to fix this.')

    ### define colors

    model_colors = {'intem':[['royalblue','royalblue'],
                             ['grey','darkorange'],
                             ['darkorange','darkorange'],
                             ['firebrick','pink'],
                             ['purple','mediumpurple']],
                    'elris':[['purple','mediumpurple'],
                             ['deeppink','pink'],
                             ['darkorange','red']],
                    'rhime':[['darkgreen','green'],
                             ['limegreen','palegreen'],
                             ['olive','lightgreen']]}

    ### font settings & annotate_coords

    if (ppt_mode):
        plt.rc('font', size=15)
        plt.rc('axes', titlesize=18)
        plt.rc('axes', labelsize=16)
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        plt.rc('legend', fontsize=14)

        annotate_coords = {0:[0.58,0.65],
                           1:[0.58,0.40],
                           2:[0.58,0.15]}

        print('WARNING: Using big fonts. You might need to shrink the labels.')
    else:
        plt.rc('font', size=12)
        plt.rc('axes', titlesize=12)
        plt.rc('axes', labelsize=12)
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        plt.rc('legend', fontsize=10)

        annotate_coords = {0:[0.6,0.80],
                           1:[0.6,0.60],
                           2:[0.6,0.40]}

    return s_data,m_data,model_colors,annotate_coords

#####################################################################

def set_model_colors(models,model_colors):
    """
    Not in use
    """
    cList = [['darkorange','darkslateblue'],
             ['red','lightsalmon'],
             ['green','lightgreen'],
             ['purple','mediumpurple'],
             ['black','grey']]
    mc = dict()
    if np.unique([m.split('_')[0] for m in models]).size==len(models):
        mc = {m:model_colors[m.split('_')[0]][0] for m in models}
    else:
        for i,m in enumerate(models):
            mc[m] = cList[i]
    return mc

#####################################################################

def set_model_colors_2(models,model_colors):
    """
    Sets plotting colors for each model (updates model_colors).

    Args:
        models (list of str):
            Keys specifying model names, e.g. ['intem','elris']
        model_colors (dict of lists):
            Default lists of colors to be used by each model.

    Returns:
        mc (dict of lists):
            List of colors to be used by each model.
    """

    mc = dict()
    m0_list = np.unique([m.split('_')[0] for m in models])

    # If the different models result from a single inversion system
    if len(m0_list) == 1:
        inv_models = list(model_colors.keys())
        i = 0
        j = 0
        # Use model_colors in order
        for m in models:
            if j == len(model_colors[inv_models[i]]):
                i = i+1
                j = 0

            try:
                mc[m] = model_colors[inv_models[i]][j]
                j = j+1
            except:
                print('ERROR: Number of models is greater than number of colors in model_colors.')

    # If results from multiple inversion systems will be plotted together
    else:
        tmp_m0 = models[0].split('_')[0]
        j = 0
        for m in models:
            m0 = m.split('_')[0]
            if m0 != tmp_m0:
                tmp_m0 = m0
                j = 0

            try:
                mc[m] = model_colors[m0][j]
                j = j+1
            except:
                print(f'ERROR: Trying to use color number {j+1}, but there are only {j} colors defined for {m0}.')

    return mc

def set_colormaps(cols):
    """ 
    Picks the color map to use and spreads the range of data over ten colors.
    Has an 11th color for anything over the max value if a custom range is 
    specified.  There is the choice of giving color bar labels for every 
    color or for every two colors.
    Code written by MO GHG inversion group.
    """
    
    if cols == 'greyscale':
        c0 = 'white'
        c1 = '#F0F0F0'
        c2 = '#D8D8D8'
        c3 = '#C0C0C0'
        c4 = '#A8A8A8'
        c5 = '#909090'
        c6 = '#787878'
        c7 = '#606060'
        c8 = '#484848'
        c9 = '#303030'
        c10 = '#181818'
        c11 = '#000000'
        
    elif cols == 'bluescale':
        c0 = 'white'
        c1 = '#E6E6FF'
        c2 = '#CCCCFF'
        c3 = '#B2B2FF'
        c4 = '#9999FF'
        c5 = '#8080FF'
        c6 = '#6666FF'
        c7 = '#4D4DFF'
        c8 = '#3333FF'
        c9 = '#1919FF'
        c10 = '#0000FF'
        c11 = '#000099'
        
    elif cols == 'green':
        c0 = 'white'
        c1 = '#ecfeea'
        c2 = '#ceefd4'
        c3 = '#b1e0bf'
        c4 = '#96d1ab'
        c5 = '#7dc198'
        c6 = '#66b187'
        c7 = '#51a076'
        c8 = '#408f67'
        c9 = '#337e5a'
        c10 = '#2c6d4e'
        c11 = '#2a5c44'
        
    else:
        print("invalid colour map selected")
        sys.exit()
        
    cmap = colors.ListedColormap([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10])
    cmap.set_over(c11)
    cmap.set_under(c0)
    cmap.set_bad("white")
    
    return cmap

#####################################################################

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
    
    for i,m in enumerate(models):
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

#####################################################################

def slice_flux(ds_all,start_date,end_date,s_data,
               scale_units=True,scale_co2eq=False,species=None):
    """
    Slices the flux datasets to within given time limits and 
    scales fluxes into Tg/Gg based on the species.
    
    Args:
        ds_all (dictionary of datasets): 
            xarray datasets read directly from each model's flux netCDF.
        start_date (str): 
            Date to slice data from, e.g. '2021-01-01'
        end_date (str): 
            Date to slice data to, e.g. '2022-01-01' would include all
            data up to 2021-12-31.
        s_data (dict of dict):
            Dictionary of species with information for plotting (read from json file).
        scale_units (bool): 
            If True, scales country fluxes to Tg or Gy per year.
        scale_co2eq (bool):
            If True, converts country fluxes to CO2-eq in Tg per year.
        species (str):
            Gas species, used to choose scaling units, e.g. 'ch4'.
    Returns:
        ds_all (dictionary of datasets):
            xarray datasets, scaled and sliced between chosen dates.
    
    """
    
    #variables that aren't scaled by units
    skip_var = ['flux_total_prior','flux_total_posterior','percentile_flux_total_prior',
                'percentile_flux_total_posterior','countryname','country',
                'country_fraction','outer_region_fraction',
                'covariance_country_flux_total_posterior','flux_total_posterior_inversion_grid']

    for m in ds_all.keys():
        
        m0 = m.split('_')[0]
        
        print(f'\nMasking data from {m}')
        try:
            ds_all[m] = ds_all[m].sel(time=slice(start_date,end_date))
        except:
            ds_all[m] = None
            print(f'No {m} fluxes found between {start_date} and {end_date}')
            print(f'Skipping {m}')
            
        if scale_units == True:
            gwp = 1
            scale_factor = s_data[species]["units_scaling"][m0]
            # Update scaling factors
            if scale_co2eq == True:
                gwp = s_data[species]["gwp"]
                if (s_data[species]["units_print"] == "G"): #units_print is expected to be either G or T
                    scale_factor = scale_factor * 1e3 #Convert to Tg
                    # Note: units_print is not re-written because it would go back to
                    #       its original value if initialize_settings is re-run.

            print(f'Scaling {m} country fluxes by {scale_factor*gwp}')
            if ds_all[m] is not None:
                var_names = [k for k in ds_all[m].keys() if k not in skip_var]
                for v in var_names:
                    ds_all[m][v].values = ds_all[m][v].values/scale_factor * gwp

                cov_var = 'covariance_country_flux_total_posterior'
                if cov_var in ds_all[m].keys():
                    ds_all[m][cov_var].values = ds_all[m][cov_var].values/scale_factor**2 * gwp**2
                    print(f'Scaling covariance in {m} by {scale_factor**2 * gwp**2}')
        
    return ds_all

#####################################################################

def read_flux_total_fgases(data_dir,species,models,s_data,m_data,regions,
                           start_date,end_date,period_override=None):
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
                       'hfc227ea','hfc245fa','hfc32','hfc365mfc','hfc4310mee']
    elif species == 'all_pfc':
        all_species = ['cf4','pfc116','pfc218','pfc318']
    else:
        print('This function can only be used to read total hfc (all_hfc) or total pfc (all_pfc).')
        sys.exit()
       
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
        
        longrun = False
        if 'longrun' in model:
            #model = model.split('_')[0]
            #models[m] = model
            longrun = True

        missing_species[model] = []
        m0 = model.split('_')[0]
        
        for s,species in enumerate(all_species):

            #dictionary containing datasets for each species, these are then summed/averaged across the time coordinate
            ds_out = {}
            
            #tries to read from standard filename
            try:
                if '20' in model:
                    model_read = model
                else:
                    model_read = f'{m0}_{s_data[species]["std_run"][m0]}'
                if longrun: model_read = f'{m0}_{s_data[species]["std_run"][m0+"_longrun"]}'
                #if longrun: model_read = f'{model_read}_longrun'
            

                ds_in[model] = read_flux(data_dir,species,[model_read],s_data,m_data,period_override[s],verbose=True)[model_read]    #edit read_flux so that it searches for correct filename per gas
                
                if len(models) > 1:
                    print('Dropping spatial flux variables, to reduce memory usage. Comment out this section if using this function for spatial flux plotting.')
                    ds_in[model] = ds_in[model].drop_vars(['flux_total_prior','percentile_flux_total_prior',
                                                                'flux_total_posterior','percentile_flux_total_posterior',
                                                                'flux_total_prior_out','percentile_flux_total_prior_out',
                                                                'flux_total_posterior_inversion_grid',
                                                                'percentile_flux_total_posterior_inversion_grid',
                                                                'country_fraction','outer_region_fraction'],errors='ignore')
                
                
                with io.capture_output() as captured:
                    ds_in[model] = slice_flux(ds_in,start_date[m],end_date[m],s_data,scale_units=False,species=None)[model]
                
            except:
                ds_in[model] = None
                if species not in missing_species[model]:
                    missing_species[model].append(species)
            

            for r,region in enumerate(regions):
                
                try:
                    region_time,region_flux_total_posterior,region_flux_total_prior,\
                    region_flux_total_posterior_lower,region_flux_total_posterior_upper,\
                    region_flux_total_prior_lower,region_flux_total_prior_upper = extract_region_flux(ds_in,model,m0,region,verbose=False,sector='total')
                    
                    #for percentiles, first convert to upper and lower standard deviations (difference from mean)
                    region_flux_total_posterior_lower = (region_flux_total_posterior-region_flux_total_posterior_lower) * 1e3 * s_data[species]['gwp'] * 1e-12
                    region_flux_total_posterior_upper = (region_flux_total_posterior_upper-region_flux_total_posterior) * 1e3 * s_data[species]['gwp'] * 1e-12
                    region_flux_total_prior_lower = (region_flux_total_prior-region_flux_total_prior_lower) * 1e3 * s_data[species]['gwp'] * 1e-12
                    region_flux_total_prior_upper = (region_flux_total_prior_upper-region_flux_total_prior) * 1e3 * s_data[species]['gwp'] * 1e-12
                    
                    region_flux_total_posterior = region_flux_total_posterior * 1e3 * s_data[species]['gwp'] * 1e-12
                    region_flux_total_prior = region_flux_total_prior * 1e3 * s_data[species]['gwp'] * 1e-12

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
                country_shortnames.append(countrycodes_dict[c])
            except:
                country_shortnames.append(regions_dict_old[c])
                
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

#####################################################################
def calculate_resample_uncertainty(ds_all_original,ds_all_p,rtime,
                                   resample_uncert_correlation=False):
    """
    Recalculates resampled flux uncertainty, using the assumption
    that all periods in the resampled flux average are uncorrelated.
    Args:
        ds_all_original (dictionary of datasets):
            Extracted flux datasets from flux-format netcdfs.
        ds_all_p (dictionary of datasets):
            Same as above, but resampled down to lower time resolution, 
            using the .mean() method.
        rtime (str):
            Time used for resampling, e.g. 'YS' or 'QS-DEC'.
        resample_uncert_correlation (bool, default False):
            If False, recalculates resampled flux uncertainties with
            uncorrelated assumption. If True, does not recalculate
            uncertainties, and uses the mean uncertainty.
    Returns:
        ds_all_p (dictionary of datasets):
            Same dataset as above, but with updated 'percentile_...'
            terms.
    """
    
    if resample_uncert_correlation == False:

        for v in ds_all_original.keys():
            if 'percentile_country' in v:
                n_periods = ds_all_original[v].resample(time=rtime).count()[:,0,:]   #number of periods in each average
                lower = (ds_all_original[v.replace('percentile_','')] - ds_all_original[v][:,0,:])    #recalculate upper and low standard deviations
                upper = (ds_all_original[v][:,1,:] - ds_all_original[v.replace('percentile_','')])
                if rtime == '1m':
                    lower_resampled = np.sqrt(((lower**2).groupby("time.month").sum(dim="time")))/n_periods  #resample using sqrt of variances,divided by number of periods
                    upper_resampled = np.sqrt(((upper**2).groupby("time.month").sum(dim="time")))/n_periods
                else:
                    lower_resampled = np.sqrt(((lower**2).resample(time=rtime).sum(dim="time")))/n_periods  #resample using sqrt of variances,divided by number of periods
                    upper_resampled = np.sqrt(((upper**2).resample(time=rtime).sum(dim="time")))/n_periods
                lower_out = ds_all_p[v.replace('percentile_','')] - lower_resampled  #recalculated percentile upper and lower bounds
                upper_out = ds_all_p[v.replace('percentile_','')] + upper_resampled
                
                ds_all_p[v] = xr.DataArray(np.concatenate((np.expand_dims(lower_out,axis=1),
                                                                np.expand_dims(upper_out,axis=1)),axis=1),
                                                dims=ds_all_p[v].dims)

    return ds_all_p
    
#####################################################################

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

#####################################################################

def slice_mf(ds_all,s_data,start_date=None,end_date=None,site=None,
             baseline_site=None,data_dir=None,
             scale_units=False,
             species=None):
    """
    Slices down the mole fraction timeseries data, to within the
    given time limits, and/or for the chosen site.
    
    Args:
        ds_all (dictionary of datasets): 
            xarray datasets read directly from each model's flux netCDF.
        s_data (dict of dict):
            Dictionary of species with information for plotting (read from json file).
        start_date (str): 
            Date to slice data from, e.g. '2021-01-01'
        end_date (str): 
            Date to slice data to, e.g. '2022-01-01' would include all
            data up to 2021-12-31.
        site (str):
            Obs site to select data from, e.g. 'MHD'.
        baseline_site (str):
            Site used to define baseline at, options for 'MHD', 'JFJ', or 'CMN'.
            If None, does not mask timeseries by baseline times.
        data_dir (str): 
            Path to top data directory, used to read baseline info files.
        scale_units (bool): 
            If True, scales country fluxes to Tg or Gy per year.
        species (str):
            Gas species, used to choose scaling units, e.g. 'ch4'.
    Returns:
        ds_all (dictionary of datasets):
            xarray datasets, scaled and sliced between chosen dates and for 
            chosen site.
    """
    
    if baseline_site is not None:
        with xr.open_dataset(os.path.join(data_dir,f'intem_baseline_timestamps/{baseline_site}_InTEM_baseline_timestamps.nc')) as f:
            baseline = f.sel(time=slice(start_date,end_date))
    
    for m in ds_all.keys():
        print(f'\nMasking data from {m}')
        
        if 'Yav' in ds_all[m].keys():
            offset = int(np.mean(ds_all[m]['Yav'].values))
        else:
            offset = (ds_all[m].time.values[1].astype('datetime64[h]') - ds_all[m].time.values[0].astype('datetime64[h]')).astype(int)

        # round seconds to integer (correction for elris)
        if 'elris' in m:
            ds_all[m]['time'] = ds_all[m]['time'].dt.round('s')

        if site is not None:
            try:
                site_index = np.where(ds_all[m]['sitenames'].astype(str) == site)[0][0]
                ds_all[m] = ds_all[m].sel(time=slice(start_date,end_date),
                                        nsite=site_index)
            except:
                ds_all[m] = None
                print(f'No {m} obs found for {site} between {start_date} and {end_date}')
        else:
            try:
                ds_all[m] = ds_all[m].sel(time=slice(start_date,end_date))
            except:
                ds_all[m] = None
                print(f'No {m} obs found between {start_date} and {end_date}')
                
        if scale_units == True:
            print(f'Scaling {m} units by {s_data[species]["mf_units_scaling"]}')
            if ds_all[m] is not None:
                var_names = [k for k in ds_all[m].keys() if k not in ['sitenames','Yav','median_poll_uncert_flag']]
                for v in var_names:
                    ds_all[m][v] = ds_all[m][v]/s_data[species]["mf_units_scaling"]
      
        if baseline_site is not None:
            print('Masking timeseries to only include baseline times')
            
            try:
                                        
                #average baseline mask over obs averaging period
                b = baseline.resample(time=f'{offset}H').mean()
                #adjust baseline mask time back to centre of av period (resample removes this)
                b['time'] = b['time'] + np.timedelta64(offset,'h')/2
                                    
                #mask baseline mask again, to only include timestamps where every period in the averaging period is classified as baseline
                b_masked = b.sel(time=b['time'].values[np.where(b['baseline'] == 1.)])
                                
                #mask dataset using only baseline times
                both_times = np.isin(ds_all[m].time.values,b_masked.time.values)
                                
                ds_all[m] = ds_all[m].sel(time=both_times)
                    
            except:
                print('Failed to mask {m} data by baseline times')
    
    check_keys = list(ds_all.keys())
    for m in check_keys:
        if ds_all[m] is None:
            ds_all.pop(m)
                
    return ds_all

#####################################################################

def stats_mf(ds_all,var='Yapost'):
    """
    Calculates the Pearson correlation coefficent and normalised root
    mean square error, of the fit between the posterior mean mf and the 
    observed mole fraction.
    
    Args:
        ds_all (dictionary of datasets):
            xarray datasets from slice_mf(), sliced between chosen dates
            but still containing all sites.
        var (str):
            Stats are calculated for this variable. Either 'Yapost' or 'Yapriori'.
    Returns:
        pearson (dictionary of dictionaries):
            Pearson correlation coeffiecient, for each site and for each model.
        nrmse (dictionary of dictionaries):
            Normalised root mean square error, for each site and for each model.
    """
    
    sites_all = np.array([])

    for i,m in enumerate(ds_all.keys()):
        sites_all = np.hstack((sites_all,ds_all[m]['sitenames'].values.astype(str)))
    
    sites_unique,sites_index = np.unique(sites_all,return_index=True)
    sites_all = sites_all[np.sort(sites_index)]
    
    pearson = {}
    nrmse = {}
    #std = {}

    for site in sites_all:
        pearson[site] = {}
        nrmse[site] = {}
        #std[site] = {}
        for i,m in enumerate(ds_all.keys()):
            if site in ds_all[m]['sitenames'].values.astype('str'):
                s = np.where(ds_all[m]['sitenames'].values.astype('str') == site)[0][0]
                if ds_all[m]['Yobs'].values[:,s][~np.isnan(ds_all[m]['Yobs'].values[:,s])].shape[0] != 0:
                    pearson[site][m] = np.round(np.corrcoef(ds_all[m]['Yobs'].values[:,s][~np.isnan(ds_all[m]['Yobs'].values[:,s])],
                                                            ds_all[m][var].values[:,s][~np.isnan(ds_all[m]['Yobs'].values[:,s])])[0,1],3)
                    nrmse[site][m] = np.round(np.sqrt(np.mean((ds_all[m][var].values[:,s][~np.isnan(ds_all[m]['Yobs'].values[:,s])]-
                                                    ds_all[m]['Yobs'].values[:,s][~np.isnan(ds_all[m]['Yobs'].values[:,s])])**2))/np.mean(ds_all[m]['Yobs'].values[:,s][~np.isnan(ds_all[m]['Yobs'].values[:,s])]),3)
                    #std[site][m] = np.std(ds_all[m]['Yapost'].values[:,s][~np.isnan(ds_all[m]['Yobs'].values[:,s])]-
                    #                      ds_all[m]['Yobs'].values[:,s][~np.isnan(ds_all[m]['Yobs'].values[:,s])])
                    
                else:
                    pearson[site][m] = np.nan
                    nrmse[site][m] = np.nan
                    #std[site][m] = np.nan
            else:
                pearson[site][m] = np.nan
                nrmse[site][m] = np.nan
                #std[site][m] = np.nan
                
    for site in sites_all:
        if all([np.isnan(v) for v in pearson[site].values()]) == True:
            del pearson[site]
            del nrmse[site]
            #del std[site]
            
    print('\nPearson correlation coefficient:')
    pprint.pprint(pearson,sort_dicts=False)
    
    print('\nNormalised RMSE')
    pprint.pprint(nrmse,sort_dicts=False)
    
    return pearson,nrmse

#####################################################################

def extract_site_info(sites):
    """
    Uses info from site_info.json to create a dictionary
    of sites with latitude and longitudes.
    """
    
    site_info_filename = os.path.join(os.getcwd(),'site_info.json')

    with open(site_info_filename, "r") as f:
        site_data = load(f)
        
    site_info = {}
    
    for s in sites:
        site_info[s] = {'latitude':site_data[s][list(site_data[s].keys())[0]]['latitude'],
                        'longitude':site_data[s][list(site_data[s].keys())[0]]['longitude']}
    
    return site_info

#####################################################################
def extract_region_flux(ds_all,m,m0,country,verbose=True,sector='total'):
    """
    Finds the index of a chosen region name and extracts the country flux
    variables for this region.
    Either extracts values directly from the dataset (if this region definition
    exists in the file) or calculates values by taking the sum of smaller regions
    (if this region definition does not exist in the file).
    By default, extracts the 'total' flux, but sector can be set to extract
    any other named sector.
    """
    
    if m0 == 'intem':
        c_key = 'countrynumber'
    elif m0 == 'rhime':
        c_key = 'country'
    elif m0 == 'elris':
        c_key = 'country'
        
    #search for existing region names
    try:
        try:
            try:
                if m0 == 'intem' and country == 'BELGIUM':
                    country_search = 'BEL-LUX'
                    if verbose: print(f'\nNOTE: InTEM does not estimate separate BELGIUM emissions.')
                    if verbose: print(f'So a population ratio of {bel_pop_r} is being used to scale InTEM\'s total BELGIUM+LUXEMBOURG estimate.\n')
                    r = bel_pop_r
                else:
                    country_search = countrycodes_dict[country]
                    r = 1
                country_index = np.where(ds_all[m][c_key].values.astype(str) == country_search)[0][0]

            # fix for RHIME which reports regions emissions with the regions_dict key names
            except:
                
                country_index = np.where(ds_all[m][c_key].values.astype(str) == country)[0][0]
                r = 1
                
        # fix for error in CW_EU definition in countrycodes_dict and older InTEM netCDF files  
        except:
            
            country_search = regions_dict_old[country]
            country_index = np.where(ds_all[m][c_key].values.astype(str) == country_search)[0][0]
            r = 1
            
        region_time = ds_all[m].time.values
        region_flux_posterior = ds_all[m][f'country_flux_{sector}_posterior'].values[:,country_index]*r
        region_flux_prior = ds_all[m][f'country_flux_{sector}_prior'].values[:,country_index]*r
        region_flux_posterior_lower = ds_all[m][f'percentile_country_flux_{sector}_posterior'].values[:,model_q_indices[m0][0],country_index]*r
        region_flux_posterior_upper = ds_all[m][f'percentile_country_flux_{sector}_posterior'].values[:,model_q_indices[m0][1],country_index]*r
        region_flux_prior_lower = ds_all[m][f'percentile_country_flux_{sector}_prior'].values[:,model_q_indices[m0][0],country_index]*r
        region_flux_prior_upper = ds_all[m][f'percentile_country_flux_{sector}_prior'].values[:,model_q_indices[m0][1],country_index]*r
        
        region_flux_posterior_lower[region_flux_posterior_lower < 0.] = 0.
        region_flux_prior_lower[region_flux_prior_lower < 0.] = 0.
    
    #calculate values for region names that don't exist in the file
    except:
        
        try:
            region_search = regions_dict[country]
            if verbose: print(f'{country} emissions are not present in {m}. Considering covariance matrix and sum of individual countries: {region_search}.')

            country_list = region_search.split('-')

            if m0 == 'intem':
                c_key = 'countrynumber'
            elif m0 == 'rhime':
                c_key = 'country'
            elif m0 == 'elris':
                c_key = 'country'

            country_index_vec = np.zeros(len(ds_all[m][c_key]))
            sigma2_region_flux_prior = 0
            region_flux_posterior = 0
            region_flux_prior = 0

            # Compute sum of prior/posterior emissions and prior uncertainty
            for var in country_list:
                try:
                    country_index = np.where(ds_all[m][c_key].values.astype(str) == var)[0][0]
                    country_index_vec[country_index] = 1

                    region_flux_posterior = region_flux_posterior + ds_all[m][f'country_flux_{sector}_posterior'].values[:,country_index]
                    region_flux_prior     = region_flux_prior + ds_all[m][f'country_flux_{sector}_prior'].values[:,country_index]

                    sigma_country_prior = ds_all[m][f'country_flux_{sector}_prior'].values[:,country_index] - ds_all[m][f'percentile_country_flux_{sector}_prior'].values[:,model_q_indices[m0][0],country_index]
                    sigma2_region_flux_prior = sigma2_region_flux_prior + sigma_country_prior**2

                except:
                    print(f'WARNING: {var} emissions are not present in {m}. This country will be neglected in {country} emissions.')
                    sigma2_region_flux_prior = np.zeros(ds_all[m].time.values.shape[0])
                    
            sigma_region_flux_prior = np.sqrt(sigma2_region_flux_prior)
        
            # Compute posterior uncertainty from covariance matrix
            try:
                sigma2 = np.zeros(np.shape(ds_all[m][f'covariance_country_flux_{sector}_posterior'])[0])

                for i in range(len(sigma2)):
                    sigma2[i] = country_index_vec.dot(ds_all[m][f'covariance_country_flux_{sector}_posterior'].values[i,:,:].dot(country_index_vec))

                sigma_region_flux_posterior = np.sqrt(sigma2)
            except:
                print(f'WARNING: Covariance matrix is not available for {m}. A posteriori uncertainty of {country} emissions will not be plotted.')
                sigma_region_flux_posterior = np.zeros(ds_all[m].time.values.shape[0])
                
            region_time = ds_all[m].time.values
            region_flux_posterior_lower = region_flux_posterior - sigma_region_flux_posterior
            region_flux_posterior_upper = region_flux_posterior + sigma_region_flux_posterior
            region_flux_prior_lower = region_flux_prior - sigma_region_flux_prior
            region_flux_prior_upper = region_flux_prior + sigma_region_flux_prior

            region_flux_posterior_lower[region_flux_posterior_lower < 0.] = 0.
            region_flux_prior_lower[region_flux_prior_lower < 0.] = 0.

        except:
            print(f'ERROR: Could not find {country} emissions for {m}.')
            print(f'Skipping read in of {m}.')
            
            region_time = None
            region_flux_posterior,region_flux_prior = None,None
            region_flux_posterior_lower,region_flux_posterior_upper = None,None
            region_flux_prior_lower,region_flux_prior_upper = None,None
    
    return (region_time,region_flux_posterior,region_flux_prior,
            region_flux_posterior_lower,region_flux_posterior_upper,
            region_flux_prior_lower,region_flux_prior_upper)
    
#####################################################################
def extract_region_inventory_flux(country,data_dir,species,
                                  s_data,scale_co2eq,start_date,end_date,
                                  inventory_year=None,sector='total',filename='UNFCCC_inventory'):
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
            
    if inventory_year == None and country == 'NW_EU2':
        inventory_year = '2024'
        print('WARNING: using hardcoded year 2024 for NW_EU2 inventory data. Edit extract_region_inventory_flux function to update this.')
            
    if inventory_year == None:
                
        try:
            try:
                with xr.open_dataset(sorted(glob.glob(os.path.join(data_dir,'inventory',f'{filename}_{species}_*.nc')))[-1]) as f:
                    inv_ds = f
            except:
                with xr.open_dataset(os.path.join(data_dir,'inventory',f'{filename}_{s_data[species]["model_species"]["intem"]}.nc')) as f:
                    inv_ds = f
        except:
            inventory_flux = None
            inventory_time = None
            
    else:
        try:
            with xr.open_dataset(sorted(glob.glob(os.path.join(data_dir,'inventory',f'{filename}_{species}_{inventory_year}.nc')))[-1]) as f:
                    inv_ds = f
        except:
            print(f'No {species} inventory data available for year {inventory_year}.')
            inventory_flux = None
            inventory_time = None

    #try:
    inv_ds = inv_ds.sel(time=slice(start_date,end_date))
    inv_c_index = np.where(inv_ds['country'].values == country)[0][0]
    
    print(inv_ds.keys())

    if sector == 'total' and filename == 'UNFCCC_inventory':
        inventory_flux = inv_ds['inventory'].values[:,inv_c_index]/scale_factor * gwp
    else:
        inventory_flux = inv_ds[f'inventory_total'].values[:,inv_c_index]/scale_factor * gwp
        
    try:
        inventory_std = inv_ds['inventory_std'].values[:,inv_c_index]/scale_factor * gwp
    except:
        inventory_std = np.zeros(inventory_flux.shape)
        
    inventory_time = inv_ds.time.values
    
    print(inventory_flux)
    '''
    except:
        try:
            region_search = regions_dict[country]
            country_list = region_search.split('-')

            inv_c_index = [0]*len(country_list)
            inv_c_value = np.zeros(len(inv_ds.time.values))
            inv_c_std_value = np.zeros(len(inv_ds.time.values))

            print(f'No inventory data available for {country}. Considering sum of individual countries: {region_search}')

            for i,var in enumerate(country_list):
                try:
                    inv_key = [k for k, code in countrycodes_dict.items() if code == var]
                    inv_c_index[i] = np.where(inv_ds['country'].values == inv_key[0])[0][0]
                    if sector == 'total' and filename == 'UNFCCC_inventory':
                        inv_c_temp = inv_ds['inventory'].values[:,inv_c_index[i]]
                        
                    else:
                        inv_c_temp = inv_ds[f'inventory_{sector}'].values[:,inv_c_index[i]]
                        
                    try:
                        inv_c_std_temp = inv_ds['inventory_std'].values[:,inv_c_index[i]]
                    except:
                        inv_c_std_temp = np.zeros(inv_c_temp.shape)
                    if np.all(np.isnan(inv_c_temp) == True):
                        inv_c_temp = np.zeros(len(inv_ds.time.values))
                        print(f'WARNING: Inventory data for {inv_key[0]} is NaN. Inventory value for {country} will not include {inv_key[0]} contributions.')

                    inv_c_value = inv_c_value + inv_c_temp
                    inv_c_std_value = np.mean(np.vstack((inv_c_std_value,inv_c_std_temp)),axis=0)
                    inventory_flux = inv_c_value/scale_factor * gwp
                    inventory_std = inv_c_std_value/scale_factor * gwp
                    inventory_time = inv_ds.time.values

                except:
                    try:
                        print(f'WARNING: No inventory data available for {inv_key[0]}. Inventory value for {country} will not include {inv_key[0]} contributions.')
                    except:
                        print(f'ERROR: {var} does not exist in country dictionary!')
                    inventory_flux = None
                    inventory_std = None
                    inventory_time = None

        except:
            print(f'No inventory data available for {country}')
            inventory_flux = None
            inventory_std = None
            inventory_time = None
    '''
    return inventory_flux,inventory_std,inventory_time

#####################################################################
def resample_flux(ds_all,species,resample,period_override,s_data,
                  resample_uncert_correlation):
    """
    Uses xarray to create annual or seasonal averages from monthly inversion runs.
    """
    
    ds_all_original = {m:ds_all[m].copy() for m in ds_all.keys()}
    ds_all_p = {}
    
    # Create annual mean xarrays if needed
    for i,m in enumerate(ds_all.keys()):
        if resample[i] is not None:
            if (resample[i] == 'year'):
                rtime = 'YS'
            elif (resample[i] == 'season'):
                rtime = 'QS-DEC'
            elif (resample[i] == 'month'):
                rtime = 'MS'
            else:
                print(f'ERROR: Option resample=\'{resample[i]}\' is not available. Try \'year\' \'month\' or \'season\'.')
                return None
            
            if period_override is not None: 
                if 'elris' in m:
                    del ds_all_original[m]['covariance_country_flux_total_posterior']
                if 'monthly' in period_override[i]:
                    #if resample[i] == 'season':
                    #    ds_all_p[m] = ds_all_original[m].groupby("time.season").mean().sel(season='J')
                    #else:
                    ds_all_p[m] = ds_all_original[m].resample(time=rtime).mean(dim="time")
                else:
                    ds_all_p[m] = ds_all_original[m].copy()
                if 'elris' in m and period_override[i] == 'monthly':
                    ds_all_p[m]['country'] = ds_all_p[m]['country'].isel(time=0).drop('time')
                    ds_all_p[m]['country_fraction'] = ds_all_p[m]['country_fraction'].isel(time=0).drop('time')
                    ds_all_p[m] = ds_all_p[m].assign({'covariance_country_flux_total_posterior':
                                                    ds_all[m]['covariance_country_flux_total_posterior'].resample(time=rtime).mean(dim="time")})
                            
            elif s_data[species]["period"]=='monthly' or s_data[species]["period"]=='3monthly':
                if 'elris' in m:
                    del ds_all_original[m]['covariance_country_flux_total_posterior']
                #if resample == 'season':
                #    ds_all_p[m] = ds_all_original[m].groupby("time.season").mean().sel(season='J')
                #else:
                ds_all_p[m] = ds_all_original[m].resample(time=rtime).mean(dim="time")
                if 'elris' in m:
                    ds_all_p[m]['country'] = ds_all_p[m]['country'].isel(time=0).drop('time')
                    ds_all_p[m]['country_fraction'] = ds_all_p[m]['country_fraction'].isel(time=0).drop('time')
                    ds_all_p[m] = ds_all_p[m].assign({'covariance_country_flux_total_posterior':
                                                    ds_all[m]['covariance_country_flux_total_posterior'].resample(time=rtime).mean(dim="time")})
            
            else:
                ds_all_p[m] = ds_all[m].copy()
            ds_all_p[m] = calculate_resample_uncertainty(ds_all_original[m],ds_all_p[m],rtime,
                                                  resample_uncert_correlation=resample_uncert_correlation)
        
        # shift timestamps of averaged data forwards to centre of inversion period
            time_mid = np.array([]).astype('datetime64[ns]')
            time_diff_all = np.array([]).astype('timedelta64[ns]')
            
            for i,t in enumerate(ds_all_p[m].time.values):
                if i < ds_all_p[m].time.values.shape[0]-1:
                    time_diff = (ds_all_p[m].time.values[i+1].astype('datetime64[ns]') - ds_all_p[m].time.values[i].astype('datetime64[ns]'))/2
                    time_diff_all = np.hstack((time_diff_all,time_diff))
                    time_mid = np.hstack((time_mid,t+time_diff))
                else:
                    try:
                        av_diff = Counter(time_diff_all).most_common()[0][0]
                    except:
                        av_diff = np.mean(time_diff_all)
                    time_mid = np.hstack((time_mid,t+av_diff))
                    
            ds_all_p[m]['time'] = time_mid
            
        else:
            ds_all_p[m] = ds_all[m].copy()
       
    return ds_all_p

#####################################################################
def calc_rolling_mean(data,n_periods):
    """
    Calculates a rolling mean using the equation:
        mean[i] = sum(flux[i-a]:flux[i+a])/n_periods
        where a = (n_periods - 1) / 2
        
    The start and end values (which are not surrounded by the complete number
    of n_periods required for a full average) are calculated using the maximum
    number of surrounding periods available.
    """
    
    if n_periods%2 != 1:
        print('ERROR: rolling_mean is the total number of periods include in the average. So it must be an odd number.')
        sys.exit()

    a = int((n_periods - 1) / 2)    #number of periods either size of central period to include in average

    rolling_mean = np.zeros(data.shape)

    all_indices = np.arange(0,data.shape[0],1).astype(int)

    for i,v in enumerate(all_indices):
        if (i-a) in all_indices and (i+a) in all_indices:
            rolling_mean[i] = np.sum(data[i-a:i+a+1])/n_periods
        #start of timeseries
        elif (i+a) in all_indices:  
            rolling_mean[i] = np.sum(data[:i+a+1])/data[:i+a+1].shape[0]
        #end of timeseries
        elif (i-a) in all_indices:  
            rolling_mean[i] = np.sum(data[i-a:])/data[i-a:].shape[0]
        
    return rolling_mean
    
#####################################################################
def plot_country_flux(ds_all_flux_scaled,species,regions,
                      s_data,m_data,model_colors,
                      start_date,end_date,ppt_mode=False,
                      scale_co2eq=False,
                      plot_inventory=True,inventory_years=None,
                      data_dir=None,fix_y_axes=False,plot_prior=False,
                      add_prior_unc=False, set_global_leg=False,
                      country_codes_as_titles=None,
                      skip_country_title=False,plot_separate=True,
                      plot_combined=False,resample=None,
                      resample_uncert_correlation=False,
                      plot_resample_and_original=False,
                      period_override=None,plot_grid=True,
                      inventory_start_date=None,nid_style_plot=False,
                      rolling_mean=False,models_priority=None,
                      combine_to_one_timeseries=False,seasonal_mean=False):
    """
    Timeseries plot of prior and posterior country fluxes, from list of 
    areas in plot_regions.
    
    Args:
        ds_all_flux_scaled (dictionary of datasets):
            xarray datasets of fluxes, scaled and sliced between 
            chosen dates.
        species (str): 
            Gas species, e.g. 'ch4'.
        plot_regions (list of str):
            Country or regions to plot, e.g. ['UNITED KINGDOM','SWITZERLAND']
        s_data (dict of dict):
            Dictionary of species with information for plotting (read from json file).
        m_data (dict of dict):
            Dictionary of inversion runs with filename and plot label (read from json file).
        start_date (str) and end_date (str):
            Start and end dates of the data to plot.
            Used to slice inventory data.
        ppt_mode (logical) (optional):
            If True, adjust global legend position to accomodate bigger fonts.
        scale_co2eq (bool):
            If True, adapt y-axis label to CO2-eq.
        model_colors (dict of str):
            Models and corresponding colours used to plot the model.
        plot_inventory (bool):
            If True, plots inventory flux estimates as bars in each plot.
        inventory_years (list of str, optional):
            List of inventory data from different years to include. If None, only plots 
            the most recent inventory data.
        data_dir (str): 
            Path to top data directory, used to read inventory data files.
        fix_y_axes (bool):
            If True, uses a consistent y axis for all plots.
        add_prior_unc (bool):
            If True, plots prior uncertainty as shaded area.
        set_global_leg (bool):
            If True, plots one single legend instead of one legend per subplot.
        country_codes_as_titles (bool)
            If True, uses list of country codes as titles, instead of the region names.
        skip_country_title(bool, default False):
            If True, does not plot country title.
        plot_separate (bool):
            If True, plots model results as separate lines.
        plot_combined (bool):
            If True, plots combined average results from all models.
        resample (str):
            Option to be passed to resample built-in function of xarray Dataset. 
            For yearly average, 'YS' option should be used; 'QS-DEC' for seasonaly average.
            See http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
        resample_uncert_correlation (bool, default False):
            If True, calculates the resampled uncertainty as the mean from all averaged periods.
            If False, recalculates uncertainty assuming no correlation between all averaged periods,
            by taking the square root of the summed variances, divided by the number of averaging 
            periods.
        plot_resample_and_original (bool):
            If True, plots both the resampled data and the data as its original frequency.
            If False, only plots the resampled data.
        period_override (list of str, optional):
            Inversion periods to include, to override the standards in species_info.json.
            Must be the same length as models, e.g. ['monthly',None,'yearly']
        plot_grid (bool, default True):
            Plot background grid lines. 
        inventory_start_date (str) (optional):
            start_date for inventory data, to override start_date for inversion data.
        nid_style_plot (bool) (default False):
            If True, adjusts plot formatting slightly for use in NIR/NISC reports,
            e.g. uses different colours/format for inventory bars.
        rolling_mean (int or list of int) (optional):
            If not None, calculates a rolling mean over this number of periods.
            e.g. if set to 3, will calculate the mean for each timestamp from values
            between timestamp-1 and timestamp+1.
    Returns:
        fig (figure): 
            A plot per country/region.
    """
    
    models = list(ds_all_flux_scaled.keys())
    
    if type(plot_prior) == bool:
        plot_prior = [plot_prior] * len(ds_all_flux_scaled.keys())
    elif len(plot_prior) != len(models):
        print('ERROR: plot_prior must be a single boolean, or the same length as models')
        return None
    
    if type(start_date) == list:
        start_date = str(min(start_date))
        end_date = str(max(end_date))
        
    if type(inventory_start_date) == list:
        print('ERROR: inventory_start_date must be a single string')
        return None
        
    if len(rolling_mean) != len(models):
        print('ERROR: rolling_mean must be the same length as models')
        return None

    if len(resample) != len(models):
        print('ERROR: resample must be the same length as models')
        return None
        
    if rolling_mean == None:
        rolling_mean = [None] * len(ds_all_flux_scaled.keys())
    print(f'\nApplying rolling means {dict(zip(ds_all_flux_scaled.keys(),rolling_mean))}.')
    
    print(f'\nApplying resampling: {dict(zip(ds_all_flux_scaled.keys(),resample))}.\n')
        
    ### remove spatial flux variables to speed up later processing

    for m in models:
        ds_all_flux_scaled[m] = ds_all_flux_scaled[m].drop_vars(['flux_total_prior','percentile_flux_total_prior',
                                                                'flux_total_posterior','percentile_flux_total_posterior',
                                                                'flux_total_prior_out','percentile_flux_total_prior_out',
                                                                'flux_total_posterior_inversion_grid',
                                                                'percentile_flux_total_posterior_inversion_grid',
                                                                'country_fraction','outer_region_fraction'],errors='ignore')

    ds_all = ds_all_flux_scaled.copy()

    ### resample from monthly to yearly, if needed
    
    ds_all_p = resample_flux(ds_all_flux_scaled,species,resample,period_override,s_data,
                                    resample_uncert_correlation)
    
    rolling_mean_adjusted = []
    plot_prior_adjusted = []
    rolling_mean_count = 0
    
    for i,m in enumerate(models):
        
        rolling_mean_adjusted.append(rolling_mean[rolling_mean_count])
        plot_prior_adjusted.append(plot_prior[rolling_mean_count])
        
        if plot_resample_and_original == True:
            if resample[i] is not None:
                ds_all_p[f'{m}_original'] = ds_all_flux_scaled[m].copy()
                rolling_mean_adjusted.append(rolling_mean[rolling_mean_count])
                plot_prior_adjusted.append(plot_prior[rolling_mean_count])
        rolling_mean_count += 1
        

    ### calculate rolling means, if needed
            
    for i,m in enumerate(models):
        if rolling_mean_adjusted[i]:
            for var in list(ds_all_p[m].keys()):
                if var != 'countryname':
                    for c in range(ds_all_p[m]['countrynumber'].values.shape[0]):
                        if 'percentile' in var:
                            for p in range(2):
                                ds_all_p[m][var][:,p,c] = calc_rolling_mean(ds_all_p[m][var].values[:,p,c],rolling_mean_adjusted[i])
                        else:
                            ds_all_p[m][var][:,c] = calc_rolling_mean(ds_all_p[m][var].values[:,c],rolling_mean_adjusted[i])

    ### combine monthly and yearly model output into one timeseries, if needed
    
    ds_merged = {}
    
    if combine_to_one_timeseries == True:
        
        # set all timestamps to the start of the year, this is needed for combing monthly and yearly model output into one timeseries
        ds_merged_input = ds_all_p.copy()
        for i,m in enumerate(models):
            ds_merged_input[m]['time'] = ds_merged_input[m].time.values.astype('datetime64[Y]').astype('datetime64[ns]')

        if len(models) == 2:
            ds_merged[models[0]] = ds_merged_input[models[models_priority[0]]].combine_first(ds_merged_input[models[models_priority[1]]])
        elif len(models) == 1:
            ds_merged[models[0]] = ds_merged_input[models[0]].copy()
    
        # shift timestamps of averaged data forwards to centre of inversion period
        time_mid = np.array([]).astype('datetime64[ns]')
        time_diff_all = np.array([]).astype('timedelta64[ns]')
        
        for i,m in enumerate(list(ds_merged.keys())):
        
            for i,t in enumerate(ds_merged[m].time.values):
                if i < ds_merged[m].time.values.shape[0]-1:
                    time_diff = (ds_merged[m].time.values[i+1].astype('datetime64[ns]') - ds_merged[m].time.values[i].astype('datetime64[ns]'))/2
                    time_diff_all = np.hstack((time_diff_all,time_diff))
                    time_mid = np.hstack((time_mid,t+time_diff))
                else:
                    try:
                        av_diff = Counter(time_diff_all).most_common()[0][0]
                    except:
                        av_diff = np.mean(time_diff_all)
                    time_mid = np.hstack((time_mid,t+av_diff))
                    
            ds_merged[m]['time'] = time_mid
    
    else:
        ds_merged = ds_all_p.copy()
    

    min_start = str(min(ds_merged[models[0]].time.values).astype('datetime64[Y]').astype('datetime64[D]'))
    max_end = str(max(ds_merged[models[0]].time.values).astype('datetime64[Y]').astype('datetime64[D]'))

    ### extract region fluxes from merged, resampled and rolling-mean-ed datasets
    
    period_all = {}
    
    for m,model in enumerate(list(ds_merged.keys())):
         # Get inversion period
        if period_override is not None:
            if period_override[m] == 'monthly':
                period_all[model] = 'monthly'
            elif period_override[m] == 'yearly':
                period_all[model] = 'yearly'
            else:
                period_all[model] = s_data[species]["period"]
        else:
            period_all[model] = s_data[species]["period"]

    region_time = {}
    region_flux = {}
    region_flux_prior = {}
    region_flux_lower = {}
    region_flux_upper = {}
    region_flux_prior_lower = {}
    region_flux_prior_upper = {}

    for m,model in enumerate(list(ds_merged.keys())):
        
        region_time[model] = {}
        region_flux[model] = {}
        region_flux_prior[model] = {}
        region_flux_lower[model] = {}
        region_flux_upper[model] = {}
        region_flux_prior_lower[model] = {}
        region_flux_prior_upper[model] = {}
        
        for r,region in enumerate(regions):
            
            m0 = model.split('_')[0]
            
            region_time[model][region],region_flux[model][region],region_flux_prior[model][region],\
            region_flux_lower[model][region],region_flux_upper[model][region],\
            region_flux_prior_lower[model][region],region_flux_prior_upper[model][region] = extract_region_flux(ds_merged,model,m0,region,sector='total')

            #region_time[model][region] = np.array([str(t.astype('datetime64[Y]')) for t in region_time[model][region]])
                        
            if ('region_time_years' in locals()) == False:
                region_time_years = region_time[model][region].astype('datetime64[Y]')
            else:
                region_time_years = np.hstack((region_time_years,region_time[model][region].astype('datetime64[Y]')))
    
    ### calculate the average results from all models
    
    if plot_combined == True:
        print('WARNING: Plotting combined data (as an average of all model inputs) only works if all models have data across the same time period.')
    
        region_time_combined = {}
        region_flux_combined = {}
        region_flux_prior_combined = {}
        region_flux_lower_combined = {}
        region_flux_upper_combined = {}
        
        for m,model in enumerate(list(ds_merged.keys())):
            
            if m == 0:
                region_time_combined[region] = region_time[model][region].copy()
                region_flux_combined[region] = region_flux[model][region].copy()
                region_flux_lower_combined[region] = region_flux_lower[model][region].copy()
                region_flux_upper_combined[region] = region_flux_upper[model][region].copy()
                region_flux_prior_combined[region] = region_flux_prior[model][region]
                
            else:
                region_flux_combined[region] = np.vstack((region_flux_combined[region],
                                                          region_flux[model][region].copy()))
                region_flux_lower_combined[region] = np.vstack((region_flux_lower_combined[region],
                                                          region_flux_lower[model][region].copy()))
                region_flux_upper_combined[region] = np.vstack((region_flux_upper_combined[region],
                                                          region_flux_upper[model][region].copy()))
                region_flux_prior_combined[region] = np.vstack((region_flux_prior_combined[region],
                                                          region_flux_prior[region]))
    
        for r,region in enumerate(regions):
        
            region_flux_combined[region] = np.mean(region_flux_combined[region],axis=0)
            region_flux_lower_combined[region] = np.mean(region_flux_lower_combined[region],axis=0)        
            region_flux_upper_combined[region] = np.mean(region_flux_upper_combined[region],axis=0)        
            region_flux_prior_combined[region] = np.mean(region_flux_prior_combined[region],axis=0)
            
    ### calculates monthly averages (e.g. average Jan results from all years)
    
    if seasonal_mean == True:
        
        region_time_month = {}
        region_flux_month = {}
        region_flux_lower_month = {}
        region_flux_upper_month = {}
        
        for m,model in enumerate(list(ds_merged.keys())):
            
            region_time_month[model] = {}
            region_flux_month[model] = {}
            region_flux_lower_month[model] = {}
            region_flux_upper_month[model] = {}
            
            for r,region in enumerate(regions):
                for j,month in enumerate(range(1,13,1)):
                    month_id = np.where(pd.DatetimeIndex(region_time[model][region]).month == month)
                    
                     #recalc posterior monthly uncertainty, assuming no correlation between all month e.g. all January fluxes
                    lower = region_flux[model][region][month_id] - region_flux_lower[model][region][month_id]   
                    upper = region_flux_upper[model][region][month_id] - region_flux[model][region][month_id]
                    lower_av = np.sqrt(np.sum(lower**2))/(lower.shape[0])
                    upper_av = np.sqrt(np.sum(upper**2))/(upper.shape[0])
                    lower_perc =  np.mean(region_flux[model][region][month_id]) - lower_av
                    upper_perc =  np.mean(region_flux[model][region][month_id]) + upper_av
                    
                    if j == 0:
                        region_time_month[model][region] = np.array([j])
                        region_flux_month[model][region] = np.mean(region_flux[model][region][month_id])
                        region_flux_lower_month[model][region] = lower_perc
                        region_flux_upper_month[model][region] = upper_perc
                    else:
                        region_time_month[model][region] = np.hstack((region_time_month[model][region],np.array([j])))
                        region_flux_month[model][region] = np.hstack((region_flux_month[model][region],np.mean(region_flux[model][region][month_id])))
                        region_flux_lower_month[model][region] = np.hstack((region_flux_lower_month[model][region],lower_perc))
                        region_flux_upper_month[model][region] = np.hstack((region_flux_upper_month[model][region],upper_perc))
                        
                region_time[model][region] = region_time_month[model][region].astype('timedelta64[M]') + np.datetime64('2018-01')
                region_flux[model][region] = region_flux_month[model][region]
                region_flux_lower[model][region] = region_flux_lower_month[model][region]
                region_flux_upper[model][region] = region_flux_upper_month[model][region]

    if inventory_start_date is None:
        inventory_start_date = start_date
        
    ### start creating plot
    
    max_cf = np.zeros(len(regions))
    min_x = []
    max_x = []
    
    if len(regions) == 4:
        n_cols = 2
        n_rows = 2
    elif len(regions) < 4:
        n_cols = len(regions)
        n_rows = 1
    elif len(regions) == 6:
        n_cols = 3
        n_rows = 2
    elif len(regions) > 4:
        n_cols = 4
        n_rows = math.ceil(len(regions)/4)
        
    fig = plt.figure(constrained_layout=True,figsize=(n_cols*6,n_rows*4))
    gs = fig.add_gridspec(n_rows,n_cols)
    
    # used to iterate through subplots
    count = 0
    ds_count = 0

    for i,country in enumerate(regions):
        
        ax = fig.add_subplot(gs[count])
        
        if plot_inventory == True:
            
            if inventory_years == None:
                try:
                    search_years = sorted(glob.glob(os.path.join(data_dir,'inventory',f'UNFCCC_inventory_{species}_*.nc')))
                    inventory_years = [search_years[-1][-7:-3]]
                except:
                    inventory_years = [None]

            if nid_style_plot == True:
                if len(inventory_years) == 1:
                    inv_colours = ['black']
                    inv_linestyle = [None]
                    inv_fill = ['None']#'gainsboro']
                else:
                    inv_colours = ['firebrick']+(['black']*(len(inventory_years)-1))
                    inv_linestyle = ['dashed']+([None]*(len(inventory_years)-1))
                    inv_fill = ['None']+(['None']*(len(inventory_years)-1))#'gainsboro']
            else:
                inv_linestyle = [None,None]
                inv_fill = ['None','None']
                  
            for y,i_year in enumerate(inventory_years):
                
                try:
                    inventory_flux,inventory_std,inventory_time = extract_region_inventory_flux(country,data_dir,species,s_data,scale_co2eq,
                                                                                inventory_start_date,end_date,
                                                                                inventory_year=i_year)
                except:
                    print(f'Could not find inventory data for {species}, continuing without this.')
                    inventory_flux = None
                    inventory_std = None
                    inventory_time = None
                
                if inventory_time is not None:
                    inventory_time = inventory_time.astype('datetime64[M]') + np.timedelta64(5,'M')
                
                if inventory_flux is not None:
                    if np.any(inventory_std > 0.) == True and i_year == max(inventory_years):
                        ax.bar(inventory_time,inventory_flux,
                               np.timedelta64(280, 'D'),color=inv_fill[y],edgecolor=inv_colours[y],align='center',#align='edge',
                               label=f'Inventory {i_year}',zorder=0,linewidth=1.2,
                               yerr=inventory_std,capsize=2,linestyle=inv_linestyle[y])
                    else:
                        ax.bar(inventory_time,inventory_flux,
                                    np.timedelta64(280, 'D'),color=inv_fill[y],edgecolor=inv_colours[y],align='center',#align='edge',
                                    label=f'Inventory {i_year}',zorder=0,linewidth=1.2,
                                    linestyle=inv_linestyle[y])
                                
                    if ('region_time_years' in locals()) == False:
                        region_time_years = inventory_time.astype('datetime64[Y]')
                    else:
                        region_time_years = np.hstack((region_time_years,inventory_time.astype('datetime64[Y]')))

        else:
            print(f'plot_inventory set to False.')
            inventory_time = None
            inventory_flux = None
            inventory_std = None
        
        if plot_separate == True:
            
            for m,model in enumerate(list(ds_merged.keys())):
                model_name = model.removesuffix('_original')
                if ds_count == 0:
                    include_label = m_data[model_name]["label"]
                    include_label_prior = f'{m_data[model_name]["label"]} prior'
                else:
                    include_label = None
                    include_label_prior = None
                                        
                ax.plot(region_time[model][region],
                        region_flux[model][region],
                        label=include_label,color=model_colors[model_name][0],linewidth=2)
                
                ax.fill_between(region_time[model][region],
                        region_flux_lower[model][region],
                        region_flux_upper[model][region],
                                    alpha=0.3,color=model_colors[model_name][0])
            
            if not(plot_combined):
                if plot_prior_adjusted[m] == True:
                    ax.plot(region_time[model][region],
                                region_flux_prior[model][region],
                                label=include_label_prior,color=model_colors[model_name][0],linestyle='dashed')
            
                    if add_prior_unc == True:
                        ax.fill_between(region_time[model][region],
                                            region_flux_prior_lower[model][region],
                                            region_flux_prior_upper[model][region],
                                            alpha=0.1,color=model_colors[model_name][0])
                        max_cf[i] = np.max((max_cf[i],np.nanmax(region_flux_prior_upper[model][region])))
        

            min_x.append(np.min(region_time[model][region]).astype('datetime64[M]'))
            max_x.append(np.max(region_time[model][region]).astype('datetime64[M]'))
            
            if inventory_time is not None:
                min_x.append(np.min(inventory_time).astype('datetime64[M]'))
                max_x.append(np.max(inventory_time).astype('datetime64[M]'))
            max_cf[i] = np.max((max_cf[i],np.nanmax(region_flux_upper[model][region])))
            if plot_inventory == True:
                if inventory_flux is not None:
                    max_cf[i] = np.nanmax((max_cf[i],np.nanmax(inventory_flux+inventory_std)))


        if plot_combined == True:
                      
            ax.plot(region_time_combined[model][region].astype('datetime64[ns]'),
                            region_flux_combined[region],
                            label='Mean posterior',color='black',linewidth=3.5)
            ax.plot(region_time_combined[model][region].astype('datetime64[ns]'),
                                region_flux_prior_combined[region],
                                label='Mean prior',color='black',linestyle='dashed')
            
            ax.fill_between(region_time_combined[region].astype('datetime64[ns]'),
                                            region_flux_lower_combined[region],    
                                            region_flux_upper_combined[region],
                                            alpha=0.3,color='grey',label='Min/max of post uncertainty')

            ds_count += 1
            
        #format each subplot
        if 'all' in species:
            y_label_append = ' CO$_2$-eq'
            units_print = "T"
        elif scale_co2eq:
            y_label_append = ' CO$_2$-eq'
            units_print = "T"
        else:
            y_label_append = ''
            units_print = s_data[species]["units_print"]
        
        ax.set_ylabel(f'{s_data[species]["species_print"]} ({units_print}g y$^{{-1}}${y_label_append})')
        
        if any(period_all[p] == 'monthly' for p in list(ds_merged.keys())) and all(r != 'year' for r in resample):
            ax.set_xlim([np.min(min_x)-np.timedelta64(2,'M'),
                            np.max(max_x)+np.timedelta64(2,'M')])
        else:
            ax.set_xlim([np.min(min_x)-np.timedelta64(10,'M'),
                            np.max(max_x)+np.timedelta64(10,'M')])   
         
        #ax.set_xlim([np.min(min_x)-np.timedelta64(12,'M'),
        #                   np.max(max_x)+np.timedelta64(10,'M')]) 
        
        if seasonal_mean == True:
            ax.set_xlim([np.min(min_x)-np.timedelta64(8,'D'),
                         np.max(max_x)+np.timedelta64(8,'D')])
        #ax.set_xlim([np.datetime64('1990-01-01'),np.datetime64('2025-01-01')])         
        
        ncol = 2
        if set_global_leg == False:
            leg = ax.legend(ncol=ncol,borderpad=.4,columnspacing=1.0)#,loc='upper right')
            if plot_inventory == True:
                try:
                    for l in leg.legend_handles[:-len(inventory_years)]:
                        l.set_linewidth(3.0)
                except:
                    for l in leg.legendHandles[:-len(inventory_years)]:
                        l.set_linewidth(3.0)
            else:
                try:
                    for l in leg.legend_handles:
                        l.set_linewidth(3.0)
                except:
                    for l in leg.legendHandles:
                        l.set_linewidth(3.0)
        
        if country == 'NW_EU2':
            print_country = 'NW EUROPE'
        elif country == 'CW_EU':
            print_country = 'CENTRAL W EUROPE'
        elif country == 'NW_EU_CONTINENT':
            print_country = 'NW CONTINENTAL EUROPE'
        else:
            print_country = country
        
        if country_codes_as_titles == True:
            try:
                ax.set_title(f'{print_country}\n{regions_dict[country]}')
            except:
                ax.set_title(f'{print_country}')
        else:        
            if skip_country_title == False:
                ax.set_title(f'{print_country}')
            
        if plot_grid == True:
            ax.grid(visible=True,which='major',alpha=0.4)
        
        # x axis labels used for longer timeseries
        region_time_years = sorted(np.unique(region_time_years))

        #region_time_years = np.arange(np.datetime64('1998'),
        #                              np.datetime64('2025'),
        #                              np.timedelta64(1,'Y'))
        
        if seasonal_mean == True:
            ax.set_xticks((np.arange(np.min(min_x),np.max(max_x)+np.timedelta64(1,'M'),np.timedelta64(1,'M'))))
            ax.set_xticklabels(np.arange(1,13,1))    
            ax.set_xlabel('Month')   
        
        elif (region_time_years[-1]-region_time_years[0]).astype('timedelta64[Y]') > 8:
            ax.set_xticks(region_time_years[::2]+np.timedelta64(5,'M'))
            ax.set_xticklabels(region_time_years[::2],rotation=90)
            ax.xaxis.set_minor_formatter(NullFormatter())

        else:
            ax.xaxis.set_minor_locator(MonthLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.xaxis.set_major_locator(YearLocator())
        
        #ax.set_xticks((np.arange(np.datetime64('1990'),
        #                        np.datetime64('2026'),np.timedelta64(1,'Y')).astype('datetime64[M]')+np.timedelta64(5,'M'))[::2])
        #ax.set_xticklabels(np.arange(1990,2026,1)[::2],rotation=90)
        
        count += 1
        
        if set_global_leg:
            if n_rows > 1:
                if (ppt_mode):
                    legend_loc = (0.5, 1.1)
                else:
                    legend_loc = (0.5, 1.07)
            else:
                legend_loc = (0.5, 1.15)
            handles, labels = ax.get_legend_handles_labels()
            ncol=0   
            if (plot_separate or resample):
                ncol=len(ds_all.keys())
            if (plot_combined and plot_separate):
                ncol=math.floor(len(ds_all.keys())/2)+2
            elif plot_combined:
                ncol=3
            if plot_inventory:
                ncol=ncol+1
            leg = fig.legend(handles, labels, loc='upper center',ncol=ncol,borderpad=.4,columnspacing=1.0,bbox_to_anchor=legend_loc)
            if plot_inventory == True:
                try:
                    for l in leg.legend_handles:
                        l.set_linewidth(3.0)
                except:
                    for l in leg.legendHandles:
                        l.set_linewidth(3.0)
            else:
                try:
                    for l in leg.legend_handles:
                        l.set_linewidth(3.0)
                except:
                    for l in leg.legendHandles:
                        l.set_linewidth(3.0)

    # loop through plots again to fix min/max axis values
    
    for i,country in enumerate(regions):
        if fix_y_axes == True:
            fig.axes[i].set_ylim([0,np.nanmax(max_cf)*1.1])  
        elif (type(fix_y_axes) == list) == True:
            fig.axes[i].set_ylim(fix_y_axes)
        elif fix_y_axes == False:
            fig.axes[i].set_ylim([0,max_cf[i]*1.1])  
    
    print('NOTE: If all the data is not within axis limits, adjust the set_ylim parameter')
    
    return fig

#####################################################################
def plot_country_flux_devolved_nations(ds_all,species,plot_regions,
                      s_data,m_data,model_colors,
                      start_date,end_date,ppt_mode=False,
                      scale_co2eq=False,
                      plot_inventory=True,inventory_years=None,
                      data_dir=None,fix_y_axes=False,plot_prior=False,
                      add_prior_unc=False, set_global_leg=False,
                      country_codes_as_titles=None,
                      skip_country_title=False,resample=None,
                      resample_uncert_correlation=False,
                      plot_resample_and_original=False,
                      period_override=None,plot_grid=True,
                      inventory_start_date=None,nid_style_plot=False,
                      rolling_mean=False):
    """
    Timeseries plot of prior and posterior country fluxes, from list of 
    areas in plot_regions.
    
    Args:
        ds_all (dictionary of datasets):
            xarray datasets of fluxes, scaled and sliced between 
            chosen dates.
        species (str): 
            Gas species, e.g. 'ch4'.
        plot_regions (list of str):
            Country or regions to plot, e.g. ['UNITED KINGDOM','SWITZERLAND']
        s_data (dict of dict):
            Dictionary of species with information for plotting (read from json file).
        m_data (dict of dict):
            Dictionary of inversion runs with filename and plot label (read from json file).
        start_date (str) and end_date (str):
            Start and end dates of the data to plot.
            Used to slice inventory data.
        ppt_mode (logical) (optional):
            If True, adjust global legend position to accomodate bigger fonts.
        scale_co2eq (bool):
            If True, adapt y-axis label to CO2-eq.
        model_colors (dict of str):
            Models and corresponding colours used to plot the model.
        plot_inventory (bool):
            If True, plots inventory flux estimates as bars in each plot.
        inventory_years (list of str, optional):
            List of inventory data from different years to include. If None, only plots 
            the most recent inventory data.
        data_dir (str): 
            Path to top data directory, used to read inventory data files.
        fix_y_axes (bool):
            If True, uses a consistent y axis for all plots.
        add_prior_unc (bool):
            If True, plots prior uncertainty as shaded area.
        set_global_leg (bool):
            If True, plots one single legend instead of one legend per subplot.
        country_codes_as_titles (bool)
            If True, uses list of country codes as titles, instead of the region names.
        skip_country_title(bool, default False):
            If True, does not plot country title.
        plot_separate (bool):
            If True, plots model results as separate lines.
        plot_combined (bool):
            If True, plots combined average results from all models.
        resample (str):
            Option to be passed to resample built-in function of xarray Dataset. 
            For yearly average, 'YS' option should be used; 'QS-DEC' for seasonaly average.
            See http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
        resample_uncert_correlation (bool, default False):
            If True, calculates the resampled uncertainty as the mean from all averaged periods.
            If False, recalculates uncertainty assuming no correlation between all averaged periods,
            by taking the square root of the summed variances, divided by the number of averaging 
            periods.
        plot_resample_and_original (bool):
            If True, plots both the resampled data and the data as its original frequency.
            If False, only plots the resampled data.
        period_override (list of str, optional):
            Inversion periods to include, to override the standards in species_info.json.
            Must be the same length as models, e.g. ['monthly',None,'yearly']
        plot_grid (bool, default True):
            Plot background grid lines. 
        inventory_start_date (str) (optional):
            start_date for inventory data, to override start_date for inversion data.
        nid_style_plot (bool) (default False):
            If True, adjusts plot formatting slightly for use in NIR/NISC reports,
            e.g. uses different colours/format for inventory bars.
        rolling_mean (int or list of int) (optional):
            If not None, calculates a rolling mean over this number of periods.
            e.g. if set to 3, will calculate the mean for each timestamp from values
            between timestamp-1 and timestamp+1.
    Returns:
        fig (figure): 
            A plot per country/region.
    """
    
    model_colors = [['royalblue','royalblue'],
                    ['darkorange','darkorange'],
                    ['gold','pink'],
                    ['purple','mediumpurple']]
    
    if type(start_date) == list:
        start_date = str(min(start_date))
        end_date = str(max(end_date))
        
    if type(rolling_mean) == int:
        print(f'\nApplying rolling mean: {rolling_mean} to all models.\n')
        rolling_mean = [rolling_mean] * len(ds_all.keys())
    elif type(rolling_mean) == list:
        if len(rolling_mean) != len(ds_all.keys()):
            print('ERROR: rolling_mean must be a None, a single int or a list of int and None, the same length as models.')
            return None
        else:
            print(f'\nApplying rolling means {dict(zip(ds_all.keys(),rolling_mean))}.\n')
    elif rolling_mean == None:
        rolling_mean = [None] * len(ds_all.keys())
        
    if type(resample) == int:
        print(f'\nApplying resample: {resample} to all models.\n')
        resample = [resample] * len(ds_all.keys())
    elif type(resample) == list:
        if len(resample) != len(ds_all.keys()):
            print('ERROR: resample must be a None, a single int or a list of int and None, the same length as models.')
            return None
        else:
            print(f'Applying resample {dict(zip(ds_all.keys(),resample))}.\n')
    elif resample == None:
        resample = [None] * len(ds_all.keys())

    if inventory_start_date is None:
        inventory_start_date = start_date
        
    if type(plot_prior) == bool:
        plot_prior = [plot_prior] * len(ds_all.keys())
    
    ds_all_p = resample_flux(ds_all,species,resample,period_override,s_data,
                  resample_uncert_correlation)
    
    max_cf = np.zeros(len(plot_regions))
    min_x = []
    max_x = []
    period_all = {}
    
    if len(plot_regions) == 4:
        n_cols = 2
        n_rows = 2
    elif len(plot_regions) < 4:
        n_cols = len(plot_regions)
        n_rows = 1
    elif len(plot_regions) == 6:
        n_cols = 3
        n_rows = 2
    elif len(plot_regions) > 4:
        n_cols = 4
        n_rows = math.ceil(len(plot_regions)/4)
        
    fig = plt.figure(constrained_layout=True,figsize=(7,5))
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[0])
    
    # used to iterate through subplots
    count = 0

    for i,country in enumerate(plot_regions):
        
        if plot_inventory == True:
            
            if inventory_years == None:
                try:
                    search_years = sorted(glob.glob(os.path.join(data_dir,'inventory',f'UNFCCC_inventory_{species}_*.nc')))
                    inventory_years = [search_years[-1][-7:-3]]
                except:
                    inventory_years = [None]

            if nid_style_plot == True:
                if len(inventory_years) == 1:
                    inv_colours = ['black']
                    inv_linestyle = [None]
                    inv_fill = ['None']#'gainsboro']
                else:
                    inv_colours = ['firebrick']+(['black']*(len(inventory_years)-1))
                    inv_linestyle = ['dashed']+([None]*(len(inventory_years)-1))
                    inv_fill = ['None']+(['None']*(len(inventory_years)-1))#'gainsboro']
            else:
                inv_linestyle = [None,None]
                inv_fill = ['None','None']
                  
            for y,i_year in enumerate(inventory_years):
                
                try:
                    inventory_flux,inventory_std,inventory_time = extract_region_inventory_flux(country,data_dir,species,s_data,scale_co2eq,
                                                                                inventory_start_date,end_date,
                                                                                inventory_year=i_year)
                except:
                    print(f'Could not find inventory data for {species}, continuing without this.')
                    inventory_flux = None
                    inventory_std = None
                    inventory_time = None
                
                if inventory_time is not None:
                    inventory_time = inventory_time.astype('datetime64[M]') + np.timedelta64(5,'M')
                
                if inventory_flux is not None:
                    if np.any(inventory_std > 0.) == True and i_year == max(inventory_years):
                        ax.bar(inventory_time,inventory_flux,
                               np.timedelta64(280, 'D'),color=inv_fill[y],edgecolor=inv_colours[y],align='center',#align='edge',
                               label=f'Inventory {i_year}',zorder=0,linewidth=1.2,
                               yerr=inventory_std,capsize=2,linestyle=inv_linestyle[y])
                    else:
                        ax.bar(inventory_time,inventory_flux,
                                    np.timedelta64(280, 'D'),color=inv_fill[y],edgecolor=inv_colours[y],align='center',#align='edge',
                                    label=f'Inventory {i_year}',zorder=0,linewidth=1.2,
                                    linestyle=inv_linestyle[y])
                                
                    if i == 0:
                        region_time_years = inventory_time.astype('datetime64[Y]')
                    else:
                        region_time_years = np.hstack((region_time_years,inventory_time.astype('datetime64[Y]')))

        else:
            print(f'plot_inventory set to False.')
            inventory_time = None
            inventory_flux = None
            inventory_std = None

        ds_count = 0
        
        if plot_resample_and_original == True:
            all_datasets = [ds_all_p,ds_all]
        else:
            all_datasets = [ds_all_p]
        
        for d,ds in enumerate(all_datasets):
        
            post_pdfs = {}
            
            for j,m in enumerate(ds.keys()):
                
                m0 = m.split('_')[0]

                # Get inversion period
                if period_override is not None:
                    if 'monthly' in period_override[i]:
                        period_all[m] = 'monthly'
                    elif period_override[i] == 'yearly':
                        period_all[m] = 'yearly'
                    else:
                        period_all[m] = s_data[species]["period"]
                else:
                    period_all[m] = s_data[species]["period"]
                    
                region_time,region_flux_total_posterior,region_flux_total_prior,\
                region_flux_total_posterior_lower,region_flux_total_posterior_upper,\
                region_flux_total_prior_lower,region_flux_total_prior_upper = extract_region_flux(ds,m,m0,country,sector='total')
                
                if d == 0 and j == 0 and ('region_time_years' in locals()) == False:
                    region_time_years = region_time.astype('datetime64[Y]')
                else:
                    region_time_years = np.hstack((region_time_years,region_time.astype('datetime64[Y]')))
                    
                if region_time is not None:
            
                    if rolling_mean[j]:
                            var_flux_total_posterior = calc_rolling_mean(region_flux_total_posterior,rolling_mean[j])
                            var_flux_total_posterior_lower = calc_rolling_mean(region_flux_total_posterior_lower,rolling_mean[j])
                            var_flux_total_posterior_upper = calc_rolling_mean(region_flux_total_posterior_upper,rolling_mean[j])

                            var_flux_total_prior = calc_rolling_mean(region_flux_total_prior,rolling_mean[j])
                            var_flux_total_prior_lower = calc_rolling_mean(region_flux_total_prior_lower,rolling_mean[j])
                            var_flux_total_prior_upper = calc_rolling_mean(region_flux_total_prior_upper,rolling_mean[j])
                    else:
                        var_flux_total_posterior = region_flux_total_posterior
                        var_flux_total_posterior_lower = region_flux_total_posterior_lower
                        var_flux_total_posterior_upper = region_flux_total_posterior_upper

                        var_flux_total_prior = region_flux_total_prior
                        var_flux_total_prior_lower = region_flux_total_prior_lower
                        var_flux_total_prior_upper = region_flux_total_prior_upper
                                
                    include_label = f'{m_data[m]["label"]} {regions_print[country]}'
                    include_label_prior = f'{m_data[m]["label"]} {regions_print[country]} prior'
                        
                    ax.plot(region_time,
                                var_flux_total_posterior,
                                label=include_label,color=model_colors[i][0],linewidth=2)
                    
                    if plot_prior[j] == True:
                        ax.plot(region_time,
                                    region_flux_total_prior,
                                    label=include_label_prior,color=model_colors[i][0],linestyle='dashed')
                
                        if add_prior_unc == True:
                            ax.fill_between(region_time,
                                                var_flux_total_prior_lower,
                                                var_flux_total_prior_upper,
                                                alpha=0.1,color=model_colors[i][0])
                            max_cf[i] = np.max((max_cf[i],np.nanmax(var_flux_total_prior_upper)))
                
                    ax.fill_between(region_time,
                                        var_flux_total_posterior_lower,
                                        var_flux_total_posterior_upper,
                                        alpha=0.3,color=model_colors[i][0])

                    min_x.append(np.min(region_time).astype('datetime64[M]'))
                    max_x.append(np.max(region_time).astype('datetime64[M]'))
                    if inventory_time is not None:
                        min_x.append(np.min(inventory_time).astype('datetime64[M]'))
                        max_x.append(np.max(inventory_time).astype('datetime64[M]'))
                    max_cf[i] = np.max((max_cf[i],np.nanmax(region_flux_total_posterior_upper)))
                    if plot_inventory == True:
                        if inventory_flux is not None:
                            max_cf[i] = np.nanmax((max_cf[i],np.nanmax(inventory_flux+inventory_std)))

            ds_count += 1
            
        #format each subplot
        if 'all' in species:
            y_label_append = ' CO$_2$-eq'
            units_print = "T"
        elif scale_co2eq:
            y_label_append = ' CO$_2$-eq'
            units_print = "T"
        else:
            y_label_append = ''
            units_print = s_data[species]["units_print"]
        
        ax.set_ylabel(f'{s_data[species]["species_print"]} ({units_print}g y$^{{-1}}${y_label_append})')
        
        if any(period_all[p] == 'monthly' for p in list(ds.keys())) and all(r != 'year' for r in resample):
            ax.set_xlim([np.min(min_x)-np.timedelta64(1,'M'),
                            np.max(max_x)+np.timedelta64(1,'M')])
        else:
            ax.set_xlim([np.min(min_x)-np.timedelta64(10,'M'),
                            np.max(max_x)+np.timedelta64(10,'M')])    
        
        ncol = 2
        if set_global_leg == False:
            leg = ax.legend(ncol=ncol,borderpad=.4,columnspacing=1.0)#,loc='upper right')
            if plot_inventory == True and inventory_time != None:
                try:
                    for l in leg.legend_handles[:-len(inventory_years)]:
                        l.set_linewidth(3.0)
                except:
                    for l in leg.legendHandles[:-len(inventory_years)]:
                        l.set_linewidth(3.0)
            else:
                try:
                    for l in leg.legend_handles:
                        l.set_linewidth(3.0)
                except:
                    for l in leg.legendHandles:
                        l.set_linewidth(3.0)
        
        print_country = 'UK Devolved Nations'
        
        if country_codes_as_titles == True:
            try:
                ax.set_title(f'{print_country}\n{regions_dict[country]}')
            except:
                ax.set_title(f'{print_country}')
        else:        
            if skip_country_title == False:
                ax.set_title(f'{print_country}')
            
        if plot_grid == True:
            ax.grid(visible=True,which='major',alpha=0.4)
        
        # x axis labels used for longer timeseries
        region_time_years = sorted(np.unique(region_time_years))
        
        if (region_time_years[-1]-region_time_years[0]).astype('timedelta64[Y]') > 8:
            ax.set_xticks(region_time_years[::2]+np.timedelta64(5,'M'))
            ax.set_xticklabels(region_time_years[::2],rotation=90)
            ax.xaxis.set_minor_formatter(NullFormatter())

        else:
            ax.xaxis.set_minor_locator(MonthLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.xaxis.set_major_locator(YearLocator())
        
        count += 1
        
        if set_global_leg:
            if n_rows > 1:
                if (ppt_mode):
                    legend_loc = (0.5, 1.1)
                else:
                    legend_loc = (0.5, 1.07)
            else:
                legend_loc = (0.5, 1.15)
            handles, labels = ax.get_legend_handles_labels()
            ncol=3
            if plot_inventory:
                ncol=ncol+1
            leg = fig.legend(handles, labels, loc='upper center',ncol=ncol,borderpad=.4,columnspacing=1.0,bbox_to_anchor=legend_loc)
            if plot_inventory == True:
                try:
                    for l in leg.legend_handles:
                        l.set_linewidth(3.0)
                except:
                    for l in leg.legendHandles:
                        l.set_linewidth(3.0)
            else:
                try:
                    for l in leg.legend_handles:
                        l.set_linewidth(3.0)
                except:
                    for l in leg.legendHandles:
                        l.set_linewidth(3.0)

    # loop through plots again to fix min/max axis values
    
    if fix_y_axes == True:
        ax.set_ylim([0,np.nanmax(max_cf)*1.1])  
    elif (type(fix_y_axes) == list) == True:
        ax.set_ylim(fix_y_axes)
    elif fix_y_axes == False:
        ax.set_ylim([0,np.nanmax(max_cf)*1.1])  

    print('NOTE: If all the data is not within axis limits, adjust the set_ylim parameter')
    
    return fig

#####################################################################
def create_annual_report_tables(ds_all_flux_scaled,models,models_priority,period_override,
                                regions,
                                data_dir,species,s_data,start_date,end_date,inventory_years,
                                rolling_mean,resample,resample_uncert_correlation,
                                output_path_table=None,save=False,
                                sectors=['total'],include_uncert=True,use_NAEI_data=False):
    """
    Create a latex-style table of inventory and InTEM flux estimates for a 
    list of regions.
    
    Args:
        ds_all_flux_scaled (dict of datasets):
            Output from slice_flux, containing all data to be output to table.
        models (list of str):
            Model names used extract data. Keys of ds_all_flux_scaled.
        models_priority (list of int):
            Priority of which data to include in the table. E.g. if models = ['intem','intem_monthly']
            and you wanted to only include intem data when intem_monthly was not available, models_priority
            should be [1,0], where lower numbers are higher priority.
        regions (list of str):
            A column is created for each region.
        data_dir (str):
            Path to location of emissions and tseries netcdfs.
        species (str):
            Gas species.
        s_data (dict of dict):
            Dictionary of species with information for plotting (read from json file).
        start_date (list of str):
            List of dates, one for each model e.g. ['1990-01-01','2012-01-01']
        end_date (list of str):
            One for each model.
        inventory_years (list of str or None):
            List of inventory data from different years to include. If None, only plots 
            the most recent inventory data available for each region. Currently hardcoded to 2024 for NWEU2.
        rolling_mean (list of int or None):
            Number of years to use to calculate a rolling mean in the intem results, one value per model,
            e.g. [3,None].
        resample (list of str or None):
            Time period over which data is resampled, one str per model, e.g. [None,'year'].
        output_path_table (str):
            Tables saved to this path.
        save (bool):
            If True, saves tables to output_path_table.
    """
    
    if use_NAEI_data == False and sectors != ['total']:
        use_NAEI_data = True
        print('WARNING: Changing inventory data source from UNFCCC total to NAEI sector level.')
        
    if use_NAEI_data == True:
        filename = 'NAEI_sector_inventory'
    else:
        filename = 'UNFCCC_inventory'

    if len(rolling_mean) != len(models):
        print('ERROR: rolling_mean must be the same length as models')
        return None

    if len(resample) != len(models):
        print('ERROR: resample must be the same length as models')
        return None
        
    if rolling_mean == None:
        rolling_mean = [None] * len(ds_all_flux_scaled.keys())
    print(f'\nApplying rolling means {dict(zip(ds_all_flux_scaled.keys(),rolling_mean))}.')
    
    print(f'\nApplying resampling: {dict(zip(ds_all_flux_scaled.keys(),resample))}.\n')

    ### read in all data, resample, apply rolling means and extract required regions
    
    ds_all_p,ds_merged,period_all,region_time,region_flux,region_flux_lower,region_flux_upper,\
    region_flux_prior_lower,\
    region_flux_prior_upper = extract_country_sector_flux(models,ds_all_flux_scaled,species,resample,
                                                          period_override,s_data,
                                                          resample_uncert_correlation,rolling_mean=rolling_mean,
                                                    plot_resample_and_original=False,combine_to_one_timeseries=True,
                                                    models_priority=models_priority,regions=regions,sectors=sectors)
    region_flux_uncert = {}
    
    for i,model in enumerate(list(region_flux.keys())):
        region_flux_uncert[model] = {}
        for r,region in enumerate(regions):
            region_flux_uncert[model][region] = {}
            for s,sector in enumerate(sectors):
                region_flux_uncert[model][region][sector] = region_flux_upper[model][region][sector] - region_flux[model][region][sector]
                if i == 0 and s == 0 and r == 0:
                    min_start = min(region_time[model][region][sector]).astype('datetime64[D]')
                    max_end = max(region_time[model][region][sector]).astype('datetime64[D]')
                    
                else:
                    min_start = min([min_start,min(region_time[model][region][sector]).astype('datetime64[D]')])
                    max_end = max([max_end,max(region_time[model][region][sector]).astype('datetime64[D]')])
                    
                region_time[model][region][sector] = [str(t.astype('datetime64[Y]')) for t in region_time[model][region][sector]]
                    
    min_start = str(min_start.astype('datetime64[Y]').astype('datetime64[D]'))
    max_end = str(max_end.astype('datetime64[Y]').astype('datetime64[D]'))
    
    ### extract inventory data

    inventory_time_all = {}
    inventory_flux_all = {}
    inventory_flux_uncert_all = {}

    for r,region in enumerate(regions):
        inventory_time_all[region] = {}
        inventory_flux_all[region] = {}
        inventory_flux_uncert_all[region] = {}
        
        for s,sector in enumerate(sectors):

            inventory_flux_all[region][sector],inventory_flux_uncert_all[region][sector],\
            inventory_time_all[region][sector] = extract_region_inventory_flux(country=region,data_dir=data_dir,
                                                                    species=species,s_data=s_data,
                                                                    scale_co2eq=False,start_date=min_start,
                                                                    end_date=max_end,
                                                                    inventory_year=inventory_years,sector=sector,
                                                                    filename=filename)
            if inventory_time_all[region][sector] is not None:
                inventory_time_all[region][sector] = [str(t.astype('datetime64[Y]')) for t in inventory_time_all[region][sector]]
        
    ### create table header and footer
    
    m0 = models[0].split('_')[0]

    col_setup = 'cc|'*(len(regions)*len(sectors))
    
    units_scaling = s_data[species]["units_scaling"][m0]
    
    if units_scaling == 1e3:
        units_caption = 'Mg'
    elif units_scaling == 1e6:
        units_caption = 'Gg'
    elif units_scaling == 1e9:
        units_caption = 'Tg'

    latexheaderitems = ['\\begin{table}[H]',
                        '\captionsetup{width=0.9\linewidth}',
                        '\centering',
                        f'\caption{{{s_data[species]["species_print"]} emission ({units_caption} yr$^{{-1}}$) estimates with 1-$\sigma$ uncertainty.}}',
                        f'\label{{table:{species}_emit}}',
                        '{\\begin{tabular}{|l|'+col_setup+'}',
                        '\hline',
                        ]
    
    header_line2 = '& '
    header_line3 = 'Years & '
    header_line5 = '& '
    
    print(sectors)
                   
    if sectors == ['total']:
        
        for r,region in enumerate(regions):
            if region == 'NW_EU2':
                header_line2 += f'NWEU & NWEU '
            else:
                header_line2 += f'{region} & {region} '
            header_line3 += 'Inventory & InTEM '
            if r == len(regions)-1:
                header_line2 += r'\\'
                header_line3 += r'\\'
                header_line5 += r'\\'
                
            else:
                header_line2 += '& '
                header_line3 += '& '
                header_line5 += '& & & '
        
    else:

        for r,region in enumerate(regions):
            for s,sector in enumerate(sectors):
                if region == 'NW_EU2':
                    header_line2 += f'NWEU & NWEU '
                else:
                    header_line2 += f'{region} & {region}'
                header_line3 += f'Inventory {sector} & InTEM {sector} '
                if r == len(regions)-1 and s == len(sectors)-1:
                    header_line2 += r'\\'
                    header_line3 += r'\\'
                   #header_line5 += r'\\'
                    
                else:
                    header_line2 += '& '
                    header_line3 += '& '
                    #header_line5 += '& & & '
            
    latexheaderitems.append(header_line2)
    latexheaderitems.append(header_line3)
    latexheaderitems.append('\hline')
    #latexheaderitems.append(header_line5)
    
    latexfooteritems = ['\hline',
                        '\end{tabular}',
                        '}',
                        '\end{table}']
        
    txtheaderitems = 'Year'
    for r,region in enumerate(regions):
        if region == 'NW_EU2':
            region_name = 'NWEU'
        else:
            region_name = region
            
        if sectors == ['total']:
            txtheaderitems += f',{region_name}_inventory,{region_name}_inventory_uncert,{region_name}_InTEM,{region_name}_InTEM_uncert'
        else:
            for s,sector in enumerate(sectors):
                txtheaderitems += f',{region_name}_{sector}_inventory,{region_name}_{sector}_inventory_uncert,{region_name}_{sector}_InTEM,{region_name}_{sector}_InTEM_uncert'
        
    ### create latex and txt table lines containing inventory and intem output

    #if species == 'hfc4310mee':
    #    inv_str_chars = 3
    #    inv_uncert_str_chars = 3
    #    intem_str_chars = 3
    #    intem_uncert_str_chars = 3
    #else:
    inv_str_chars = 2
    inv_uncert_str_chars = 2
    intem_str_chars = 2
    intem_uncert_str_chars = 2
        
    print(f'\nIf the number of decimal places in the table is not correct, edit lines near {inspect.getframeinfo(inspect.currentframe()).lineno} in PARIS_inversion_results.py to add exception for this species.\n')

    print(f'\nIf the units table are not correct, edit the units_scaling variable in species_info.json to adjust this.\n')


    latexlines = []
    txtlines = []
    
    for m,model in enumerate(list(region_time.keys())):
        for t,data_time in enumerate (region_time[model][regions[-1]][sectors[0]]):
            dataline = str(data_time)
            dataline_txt = str(data_time)
            if inventory_time_all[region] is not None:
                if data_time in inventory_time_all[region][sectors[0]]:
                    if data_time == inventory_time_all[region][sectors[0]][t]:
                        for s,sector in enumerate(sectors):
                            for r,region in enumerate(regions):
                                if all(a == 0 for a in inventory_flux_uncert_all[region][sector]):
                                    dataline += f'& {inventory_flux_all[region][sector][t]:6.{inv_str_chars}f}'
                                    dataline_txt += f',{inventory_flux_all[region][sector][t]:6.{inv_str_chars}f},'
                                else:
                                    if include_uncert == True:
                                        dataline += f'& {inventory_flux_all[region][sector][t]:6.{inv_str_chars}f} ${{\pm}}$ {inventory_flux_uncert_all[region][sector][t]:5.{inv_uncert_str_chars}f}'
                                        dataline_txt += f',{inventory_flux_all[region][sector][t]:6.{inv_str_chars}f},{inventory_flux_uncert_all[region][sector][t]:5.{inv_uncert_str_chars}f}'
                                    else:
                                        dataline += f'& {inventory_flux_all[region][sector][t]:6.{inv_str_chars}f}'
                                        dataline_txt += f',{inventory_flux_all[region][sector][t]:6.{inv_str_chars}f}'
                                    
                                if include_uncert == True:
                                    dataline += f'& {region_flux[model][region][sector][t]:5.{intem_str_chars}f} ${{\pm}}$ {region_flux_uncert[model][region][sector][t]:5.{intem_uncert_str_chars}f}'
                                    dataline_txt += f',{region_flux[model][region][sector][t]:5.{intem_str_chars}f},{region_flux_uncert[model][region][sector][t]:5.{intem_uncert_str_chars}f}'
                                else:
                                    dataline += f'& {region_flux[model][region][sector][t]:5.{intem_str_chars}f} '
                                    dataline_txt += f',{region_flux[model][region][sector][t]:5.{intem_str_chars}f}'
                                
                                if r == len(regions)-1 and s == len(sectors)-1:
                                    dataline += r' \\'
                        latexlines.append(dataline)
                        txtlines.append(dataline_txt)
                    else:
                        print('Inventory and InTEM timestamps do not match, check read in of data.')
                else:
                    for r,region in enumerate(regions):
                        for s,sector in enumerate(sectors):
                            dataline += f'& '
                            dataline_txt += f',,'
                            if include_uncert == True:
                                dataline += f'& {region_flux[model][region][sector][t]:4.2f} ${{\pm}}$ {region_flux_uncert[model][region][sector][t]:4.2f}'
                                dataline_txt += f',{region_flux[model][region][sector][t]:4.2f},{region_flux_uncert[model][region][sector][t]:4.2f}'
                            else:
                                dataline += f'& {region_flux[model][region][sector][t]:4.2f}'
                                dataline_txt += f',{region_flux[model][region][sector][t]:4.2f}'
                                
                            if r == len(regions)-1 and s == len(sectors)-1:
                                dataline += r' \\'
                    latexlines.append(dataline)
                    txtlines.append(dataline_txt)

            else:
                for r,region in enumerate(regions):
                    for s,sector in enumerate(sectors):
                        dataline += f'& '
                        dataline_txt += f',,'
                        if include_uncert == True:
                            dataline += f'& {region_flux[model][region][sector][t]:4.2f} ${{\pm}}$ {region_flux_uncert[model][region][sector][t]:4.2f}'
                            dataline_txt += f',{region_flux[model][region][sector][t]:4.2f},{region_flux_uncert[model][region][sector][t]:4.2f}'
                        else:
                            dataline += f'& {region_flux[model][region][sector][t]:4.2f}'
                            dataline_txt += f',{region_flux[model][region][sector][t]:4.2f}'
                    
                        if r == len(regions)-1 and s == len(sectors)-1:
                            dataline += r' \\'
                latexlines.append(dataline)
                txtlines.append(dataline)
            
    ### save latex file
    if save == True:
        with open(output_path_table+'.tex','w') as f:
            for l in latexheaderitems:
                f.writelines(l+'\n')
            for l in latexlines:
                f.writelines(l+'\n')
            for l in latexfooteritems:
                f.writelines(l+'\n')
        print(f'\nTable saved to {output_path_table}.tex')
        
    ### save txt file
        with open(output_path_table+'.txt','w') as f:
            f.writelines(txtheaderitems+'\n')
            for l in txtlines:
                f.writelines(l+'\n')
        print(f'\nTable saved to {output_path_table}.txt')

    return latexheaderitems,latexlines

#####################################################################
def plot_spatial_flux_one_variable(ds_all,species,plot_area,s_data,m_data,var,
                                   cmap=None,cmap_diff=None,c_border=None,period_override=None,
                                    plot_site_locations=False,plot_point_markers=None,
                                    season=None,plot_inversion_grid_flux=False,
                                    scale_to_kgkm2yr=False,nid_style_plot=False,
                                    mask_sea_areas=False,include_threshold=False,sites_available=None):
    """
    Plots either posterior or prior fluxes, one plot per model.
    Function created just for the NISC_plots version of the notebook.
    This function only plots one set of results across one time period.
    If ds_all contains mulitple time periods for each model, the average 
    across all times will be plotted.
    Plots are formatted in the style of the NIR annex.
    
    Args:
        ds_all (dictionary of datasets):
            xarray datasets of fluxes, scaled and sliced between 
            chosen dates.
        species (str): 
            Gas species, e.g. 'ch4'.
        plot_area (str or list):
            Lat/lon region to plot, options for 'UK', 'FRANCE', 'GERMANY',
            'NWEU','CWEU','EUROPE'.
            A list with [min_lon, max_lon, min_lat, max_lat] can also be provided.
        s_data (dict of dict):
            Dictionary of species with information for plotting (read from json file).
        m_data (dict of dict):
            Dictionary of inversion runs with filename and plot label (read from json file).
        var (str):
            Variable to plot, either 'flux_total_posterior' or 'flux_total_prior'. 'inversion_grid'
            is appended to this if plot_inversion_grid_flux == True.
        cmap (str):
            Colour map for flux plots.
        cmap_diff (str):
            Colour map for difference plots.
        c_border (str):
            Colour for flux plot country borders.
        period_override (list of str, optional):
            Inversion periods to include, to override the standards in species_info.json.
            Must be the same length as models, e.g. ['monthly',None,'yearly']
        plot_site_locations (bool):
            If True, adds triangles with site locations to spatial plot.
        plot_point_markers (list of str or list of lat/lon):
            List of names of points to plot over larger point sources or lat/lon locations.
            See point_markers_dict for a list of options.
            e.g. ['paris','nw_england',[50.,5.]]
        season (string, default None):
            If specified, plot the seasonal mean (only valable for monthly data). 
            Options are 'DJF', 'MAM', 'JJA', 'SON'.
        plot_inversion_grid_flux (bool, default False):
            If True, plots fluxes at the spatial resolution of the inversion (using the 
            inversion_grid variable). If False, plots fluxes at the spatial resolution
            of the prior.
        scale_to_kgkm2yr (bool, default False):
            If True, outputs fluxes in kg/km2/year. If False, uses mol/m2/s.
        nid_style_plot (bool, default False):
            If True, plots sites and cities using NIR-style markers (triangles for sites and red circles for cities.)
            Simplifies plot title and colourbar title and converts zero-value fluxes to nans, to remove from plot.
        mask_sea_areas (bool, default False):
            If True, sets fluxes in grid cells labelled as 'Sea' to zero.
        include_threshold (bool, default None):
            If True, all values below threshold_scale * max(flux) are set to zero in the plotting. 
            threshold_scale is set in species_info.json.
            This scaling step is now applied after averaging.
        sites_available (dict of list of str, default None):
            List of 3-letter site codes with data between start_date and end_date, for each model.
            If None, site location will be plotting for all sites used across whole inversion period.
    Returns:
        fig (figure): 
            A plot of spatial flux posterior and prior mean/mode and a plot 
            of the absolute difference between these, for each model.
    """
    
    var_labels = {'flux_total_prior':'Prior mean',
                  'flux_total_posterior':'Posterior mean',
                  'posterior_prior_diff':'Posterior-prior'}
    
    period_all = {}
    
    for i,m in enumerate(ds_all.keys()):
        m0 = m.split('_')[0]
        if period_override is not None:
            if 'monthly' in period_override[i]:
                period_all[m] = 'datetime64[M]'
            elif period_override[i] == 'yearly':
                period_all[m] = 'datetime64[Y]'
        else:
            period_all[m] = s_data[species]["dt_units"][m0]
    
    if cmap == None:
        cmap = 'viridis' #'Blues'
    if cmap_diff == None:
        cmap_diff = 'coolwarm'
    if c_border == None:
        c_border = 'floralwhite'
    if cmap in ['green','blue','greyscale']:
        cmap = set_colormaps(cmap)
    
    n_cols = len(ds_all.keys())

    region_limits = {'UK':[-12,4,49,62],   #min_lon, max_lon, min_lat, max_lat
                    'FRANCE':[-6,9,42,52],
                    'GERMANY':[2,18,45,60],
                    'ITALY':[6,19,36,48],
                    'SWITZERLAND':[5.5,11,45,49],
                    'NETHERLANDS':[2.5,8,50,55],
                    'IRELAND':[-12,-4,51,56],
                    'HUNGARY':[15,24,44.5,50],
                    'NORWAY':[1,32,55,76],
                    'BENELUX':[1,9,48,55],
                    'NWEU':[-11,11,45,62],
                    'CWEU':[-12,27,37,66],
                    'EUROPE':[-98,40,10,80],
                    'AREUROPE':[-12,15,41,61]}
    
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    # Define variable specific settings
    if scale_to_kgkm2yr == True:
        lim = s_data[species]['fluxlim_kgkm2yr']
        if species == 'ch4':
            flux_units_scaling = s_data[species]['mwt'] / 1e6 * 1e6 * 365*24*60*60 
            cb_units = 't km$^{-2}$ yr$^{-1}$'
        else:
            flux_units_scaling = s_data[species]['mwt'] / 1000 * 1e6 * 365*24*60*60 
            cb_units = 'kg km$^{-2}$ yr$^{-1}$'
    else:
        lim = s_data[species]['fluxlim']
        flux_units_scaling = 1
        cb_units = 'mol m$^{-2}$ s$^{-1}$'

    # find site info in netcdf attrs. if none present, use site info from first model with this 
    # data available
    sites_info = {}
    if plot_site_locations == True:
        if sites_available == None:
            print('WARNING: sites_available not supplied, so plotting all sites listed in flux file attrs.')
            for i,m in enumerate(ds_all.keys()):
                try:
                    sites_test = ds_all[m].sites.replace("'","").replace(']','').replace('[','').replace(' ','').split(',')
                    sites_info[m] = extract_site_info(sites_test)
                except:
                    sites_info[m] = None
                    
            for i,m in enumerate(ds_all.keys()):
                if sites_info[m] == None:
                    for j,m2 in enumerate(sites_info.keys()):
                        if sites_info[m2] != None:
                            print(f'No sites data available in {m} attrs, so using site data from {m2}')
                            sites_info[m] = sites_info[m2]
                    break
        else:
            for i,m in enumerate(sites_available.keys()):
                sites_info[m] = extract_site_info(sites_available[m])

    fig,ax = plt.subplots(1,n_cols,figsize=(n_cols*4,3),
                   subplot_kw={'projection':cartopy.crs.PlateCarree()})

    for i in range(n_cols):
        border_color = c_border
        if n_cols > 1:
            ax_var = ax[i]
        else:
            ax_var = ax

        ax_var.add_feature(cartopy.feature.BORDERS,edgecolor=border_color,linewidth=1.)
        ax_var.coastlines(resolution='50m',color=border_color,linewidth=1.)
        if type(plot_area) == str:
            ax_var.set_extent(region_limits[plot_area])
        elif type(plot_area) == list:    
            ax_var.set_extent(plot_area)
            
        if nid_style_plot == True:
            ax_var.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,linewidth=1, color='gray', alpha=0.5, linestyle='-')

    for i,m in enumerate(ds_all.keys()):
        
        if n_cols > 1:
            ax_var = ax[i]
        else:
            ax_var = ax
        
        lon = ds_all[m].longitude.values + (ds_all[m].longitude.values[1]-ds_all[m].longitude.values[1])/2
        lat = ds_all[m].latitude.values + (ds_all[m].latitude.values[1]-ds_all[m].latitude.values[1])/2

        m0 = m.split('_')[0]

        if len(ds_all[m].time.values) == 1:
            time_out = to_datetime(ds_all[m].time.values[0].astype(s_data[species]["dt_units"][m0])).strftime('%d/%m/%Y')
        else:
            start_print = to_datetime(ds_all[m].time.values[0].astype(period_all[m])).strftime("%d/%m/%Y")
            if period_all[m] == 'datetime64[Y]':
                end_period = ds_all[m].time.values[-1].astype(period_all[m]) + np.timedelta64(1,'Y') - np.timedelta64(1,'D')                    
            elif period_all[m] == 'datetime64[M]':
                end_period = ds_all[m].time.values[-1].astype(period_all[m]) + np.timedelta64(1,'M') - np.timedelta64(1,'D')                    
            else:
                print('This currently only works for monthly or yearly inversion periods. Update the plotting code to print out '+
                        'correct dates for higher frequency inversions.')
            end_print = to_datetime(end_period).strftime("%d/%m/%Y")
            time_out = (f'{start_print} - {end_print}')
            
        if plot_inversion_grid_flux:
            var_append = '_inversion_grid'
        else:
            var_append = ''
        
        if season is None:
            try:
                var_plot = np.mean(ds_all[m][f'{var}{var_append}'][:,:,:],axis=0)
                #var_plot = np.sum(ds_all[m][f'{var}{var_append}'][:,:,:],axis=0)
                print(np.max(var_plot*flux_units_scaling))
            except:
                print(f'Cannot find inversion_grid variables for {m} so using standard flux output.')
                var_plot = np.mean(ds_all[m][f'{var}'][:,:,:],axis=0)          
        else:
            try:
                var_plot = ds_all[m][f'{var}{var_append}'].groupby("time.season").mean().sel(season=season).values
            except:
                print(f'Cannot find inversion_grid variables for {m} so using standard flux output.')
                var_plot = ds_all[m][f'{var}'].groupby("time.season").mean().sel(season=season).values
            time_out = f'{season} of {time_out}'
            
        if nid_style_plot == True and include_threshold is True:
            for m in ds_all.keys():
                threshold = s_data[species]['threshold_scale'] * np.max(var_plot.values)
                var_plot.values[np.where(var_plot.values == 0.)] = np.nan
                var_plot.values[np.where(var_plot.values < threshold)] = np.nan

        if mask_sea_areas == True:
            print(f'Masking sea areas...')
            
            if 'caz' in socket.gethostname():
                mask_path = '/data/users/intem_ghg/inversion/region_files/regions_EUextN_land_June2024_as_netcdf.nc'
            else:
                mask_path = '/project/InTEM_GHG/inversion/region_files/regions_EUextN_land_June2024_as_netcdf.nc'
            with xr.open_dataset(mask_path) as f:
                sea_code = f['region_code'].values[np.where(f['region_name'].values == 'Sea')][0]
                mask = f['region'].values
                mask_lat = f.lat.values
                mask_lon = f.lon.values
            
            # round used to fix rounding issues in emissions netcdf lat/lons
            for a,la in enumerate(np.round(lat,3)):
                for b,lo in enumerate(np.round(lon,3)):
                    if la in mask_lat and lo in mask_lon:
                        lat_id = np.where(mask_lat == la)[0]
                        lon_id = np.where(mask_lon == lo)[0]
                        if mask[lat_id,lon_id] == sea_code:
                            var_plot[a,b] = np.nan
  
        ax_var.pcolormesh(lon,lat,var_plot*flux_units_scaling,cmap=cmap,vmin=lim[0],vmax=lim[1],shading='nearest')

        if n_cols != 1: 
            ax_var.set_title(f'{m_data[m]["label"]} {var}')

        if plot_site_locations == True:
            if sites_info[m] is not None:
                for s in sites_info[m]:
                    if nid_style_plot == True:
                        ax_var.scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],facecolor='none',
                                        edgecolor='darkblue',marker='^',s=30,zorder=2)
                        
                    else:
                        ax_var.scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='white',
                                        edgecolor='none',marker='o',s=30,zorder=2,alpha=0.5)
                        ax_var.scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='none',
                                    edgecolor='black',marker='o',s=30,zorder=2)
            
        if plot_point_markers is not None:
            if nid_style_plot == True:
                marker_edge_color = 'purple'
                marker_fill_color = 'None'
                if plot_area == 'AREUROPE':
                    marker_s = 10
                else:
                    marker_s = 20
            else:
                marker_edge_color = 'black'
                marker_fill_color = 'black'
                marker_s = 2

            if i == 0:
                print(f'\nPlotting markers for: {plot_point_markers}')
                print(f'Edit lines below line {inspect.getframeinfo(inspect.currentframe()).lineno} to change marker colour and size')
            for p in plot_point_markers:
                if type(p) == list:
                    ax_var.scatter(p[0],p[1],color=marker_fill_color,edgecolor=marker_edge_color,marker='o',s=marker_s,zorder=2)
                elif type(p) == str:
                    if p not in point_source_dict.keys():
                        print(f'{p} is not specified in point_source_dict, edit this to add a lat/lon location.')
                    else:
                        ax_var.scatter(point_source_dict[p][0],point_source_dict[p][1],color=marker_fill_color,
                                       edgecolor=marker_edge_color,marker='o',s=marker_s,zorder=2)

    print('\nEdit flux_lim_kgkm2yr variable in species_info.json to adjust colourbar limits.'+
          'You will need to rerun the first cell of the notebook to apply the adjustment\n')
            
    #flux colorbar
    cbar = plt.cm.ScalarMappable(cmap=cmap)
    levels = np.linspace(lim[0],lim[1])
    cbar.set_array(levels)
    cbar.set_clim(lim)

    # Size of color bar
    f_height = 0.9
    f_bottom = (1-f_height)/2
    f_width = 0.04/n_cols #0.02
    if plot_area == 'AREUROPE':
        f_left = 0.98         #0.94
    else:
        f_left = 0.95        #0.94

    cbar_ax = fig.add_axes([f_left, f_bottom, f_width, f_height])
    color_bar = fig.colorbar(cbar,cax=cbar_ax,orientation='vertical',cmap=cmap,extend='max')
        
    color_bar.ax.tick_params(labelsize=10)
        
    if nid_style_plot == True:
        
        nbins = np.ceil(np.max(levels)-np.min(levels)).astype(int)
        if nbins > 11:
            nbins = 11
        if nbins < 5 :
            nbins = 5
        
        nbins = 6
        
        tick_locator = ticker.MaxNLocator(nbins=nbins)#,integer=True)
        color_bar.locator = tick_locator
        color_bar.update_ticks()
        cbar_ax.annotate(f'{cb_units}',xy=(3.7,0.3),xycoords='axes fraction',rotation=90)   #hardcode position of cb label, to fix plot dimensions
        #color_bar.set_label(f'{cb_units}',labelpad=0.05)
        
    else:
        color_bar.set_label(f'{var_labels[var]} {s_data[species]["species_print"]} ({cb_units})')
    
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.04, hspace=0.12)
    
    return fig

#####################################################################
def plot_spatial_flux_by_timestamp(ds_all,species,plot_area,s_data,m_data,var,
                                   cmap=None,cmap_diff=None,c_border=None,period_override=None,
                                    plot_site_locations=False,plot_point_markers=None,
                                    plot_inversion_grid_flux=False,
                                    scale_to_kgkm2yr=False,nid_style_plot=False,
                                    mask_sea_areas=False,include_threshold=False,sites_available=None):
    """
    Plots either posterior or prior fluxes, one plot per model per timestamp.
    Function created just for the NISC_plots version of the notebook.
    Plots are formatted in the style of the NIR annex.
    
    Args:
        ds_all (dictionary of datasets):
            xarray datasets of fluxes, scaled and sliced between 
            chosen dates.
        species (str): 
            Gas species, e.g. 'ch4'.
        plot_area (str or list):
            Lat/lon region to plot, options for 'UK', 'FRANCE', 'GERMANY',
            'NWEU','CWEU','EUROPE'.
            A list with [min_lon, max_lon, min_lat, max_lat] can also be provided.
        s_data (dict of dict):
            Dictionary of species with information for plotting (read from json file).
        m_data (dict of dict):
            Dictionary of inversion runs with filename and plot label (read from json file).
        var (str):
            Variable to plot, either 'flux_total_posterior' or 'flux_total_prior'. 'inversion_grid'
            is appended to this if plot_inversion_grid_flux == True.
        cmap (str):
            Colour map for flux plots.
        cmap_diff (str):
            Colour map for difference plots.
        c_border (str):
            Colour for flux plot country borders.
        period_override (list of str, optional):
            Inversion periods to include, to override the standards in species_info.json.
            Must be the same length as models, e.g. ['monthly',None,'yearly']
        plot_site_locations (bool):
            If True, adds triangles with site locations to spatial plot.
        plot_point_markers (list of str or list of lat/lon):
            List of names of points to plot over larger point sources or lat/lon locations.
            See point_markers_dict for a list of options.
            e.g. ['paris','nw_england',[50.,5.]]
            Options are 'DJF', 'MAM', 'JJA', 'SON'.
        plot_inversion_grid_flux (bool, default False):
            If True, plots fluxes at the spatial resolution of the inversion (using the 
            inversion_grid variable). If False, plots fluxes at the spatial resolution
            of the prior.
        scale_to_kgkm2yr (bool, default False):
            If True, outputs fluxes in kg/km2/year. If False, uses mol/m2/s.
        nid_style_plot (bool, default False):
            If True, plots sites and cities using NIR-style markers (triangles for sites and red circles for cities.)
            Simplifies plot title and colourbar title and converts zero-value fluxes to nans, to remove from plot.
        mask_sea_areas (bool, default False):
            If True, sets fluxes in grid cells labelled as 'Sea' to zero.
        include_threshold (bool, default None):
            If True, all values below threshold_scale * max(flux) are set to zero in the plotting. 
            threshold_scale is set in species_info.json.
            This scaling step is now applied after averaging.
        sites_available (dict of list of str, default None):
            List of 3-letter site codes with data between start_date and end_date, for each model.
            If None, site location will be plotting for all sites used across whole inversion period.
    Returns:
        fig (figure): 
            A plot of spatial flux posterior and prior mean/mode and a plot 
            of the absolute difference between these, for each model.
    """
    
    var_labels = {'flux_total_prior':'Prior mean',
                  'flux_total_posterior':'Posterior mean',
                  'posterior_prior_diff':'Posterior-prior'}
    
    print(f'\nPlotting one map per timestamp...')
    print(f'WARNING: This function will only plot data from the first model: {list(ds_all.keys())[0]}\n')
    m = list(ds_all.keys())[0]
    
    period_all = {}
    
    m0 = m.split('_')[0]
    if period_override is not None:
        if period_override[0] == 'monthly':
            period_all[m] = 'datetime64[M]'
        elif period_override[0] == 'yearly':
            period_all[m] = 'datetime64[Y]'
    else:
        period_all[m] = s_data[species]["dt_units"][m0]
    
    if cmap == None:
        cmap = 'viridis' #'Blues'
    if cmap_diff == None:
        cmap_diff = 'coolwarm'
    if c_border == None:
        c_border = 'floralwhite'
    if cmap in ['green','blue','greyscale']:
        cmap = set_colormaps(cmap)
    
    region_limits = {'UK':[-12,4,49,62],   #min_lon, max_lon, min_lat, max_lat
                    'FRANCE':[-6,9,42,52],
                    'GERMANY':[2,18,45,60],
                    'ITALY':[6,19,36,48],
                    'SWITZERLAND':[5.5,11,45,49],
                    'NETHERLANDS':[2.5,8,50,55],
                    'IRELAND':[-12,-4,51,56],
                    'HUNGARY':[15,24,44.5,50],
                    'NORWAY':[1,32,55,76],
                    'BENELUX':[1,9,48,55],
                    'NWEU':[-11,11,45,62],
                    'CWEU':[-12,27,37,66],
                    'EUROPE':[-98,40,10,80],
                    'AREUROPE':[-12,15,41,61]}

    # Define variable specific settings
    if scale_to_kgkm2yr == True:
        lim = s_data[species]['fluxlim_kgkm2yr']
        if species == 'ch4':
            flux_units_scaling = s_data[species]['mwt'] / 1e6 * 1e6 * 365*24*60*60 
            cb_units = 't km$^{-2}$ yr$^{-1}$'
        else:
            flux_units_scaling = s_data[species]['mwt'] / 1000 * 1e6 * 365*24*60*60 
            cb_units = 'kg km$^{-2}$ yr$^{-1}$'
    else:
        lim = s_data[species]['fluxlim']
        flux_units_scaling = 1
        cb_units = 'mol m$^{-2}$ s$^{-1}$'

    # find site info in netcdf attrs. if none present, use site info from first model with this 
    # data available
    sites_info = {}
    if plot_site_locations == True:
        if sites_available == None:
            print('WARNING: sites_available not supplied, so plotting all sites listed in flux file attrs.')
            try:
                sites_test = ds_all[m].sites.replace("'","").replace(']','').replace('[','').replace(' ','').split(',')
                sites_info[m] = extract_site_info(sites_test)
            except:
                sites_info[m] = None
                    
            if sites_info[m] == None:
                for j,m2 in enumerate(sites_info.keys()):
                    if sites_info[m2] != None:
                        print(f'No sites data available in {m} attrs, so using site data from {m2}')
                        sites_info[m] = sites_info[m2]
                
        else:
            sites_info[m] = extract_site_info(sites_available[m])
                
    lon = ds_all[m].longitude.values + (ds_all[m].longitude.values[1]-ds_all[m].longitude.values[1])/2
    lat = ds_all[m].latitude.values + (ds_all[m].latitude.values[1]-ds_all[m].latitude.values[1])/2

    m0 = m.split('_')[0]

    if len(ds_all[m].time.values) == 1:
        time_out = to_datetime(ds_all[m].time.values[0].astype(s_data[species]["dt_units"][m0])).strftime('%d/%m/%Y')
    else:
        start_print = to_datetime(ds_all[m].time.values[0].astype(period_all[m])).strftime("%d/%m/%Y")
        if period_all[m] == 'datetime64[Y]':
            end_period = ds_all[m].time.values[-1].astype(period_all[m]) + np.timedelta64(1,'Y') - np.timedelta64(1,'D')                    
        elif period_all[m] == 'datetime64[M]':
            end_period = ds_all[m].time.values[-1].astype(period_all[m]) + np.timedelta64(1,'M') - np.timedelta64(1,'D')                    
        else:
            print('This currently only works for monthly or yearly inversion periods. Update the plotting code to print out '+
                    'correct dates for higher frequency inversions.')
        end_print = to_datetime(end_period).strftime("%d/%m/%Y")
        time_out = (f'{start_print} - {end_print}')
        
    if plot_inversion_grid_flux:
        var_append = '_inversion_grid'
    else:
        var_append = ''
    
    try:
        var_plot = ds_all[m][f'{var}{var_append}'][:,:,:]
    except:
        print(f'Cannot find inversion_grid variables for {m} so using standard flux output.')
        var_plot = ds_all[m][f'{var}'][:,:,:]       
        
    if nid_style_plot == True and include_threshold is True:
        for m in ds_all.keys():
            threshold = s_data[species]['threshold_scale'] * np.max(var_plot.values)
            var_plot.values[np.where(var_plot.values == 0.)] = np.nan
            var_plot.values[np.where(var_plot.values < threshold)] = np.nan

    if mask_sea_areas == True:
        print(f'Masking sea areas...')
        
        if 'caz' in socket.gethostname():
            mask_path = '/data/users/intem_ghg/inversion/region_files/regions_EUextN_land_June2024_as_netcdf.nc'
        else:
            mask_path = '/project/InTEM_GHG/inversion/region_files/regions_EUextN_land_June2024_as_netcdf.nc'
        with xr.open_dataset(mask_path) as f:
            sea_code = f['region_code'].values[np.where(f['region_name'].values == 'Sea')][0]
            mask = f['region'].values
            mask_lat = f.lat.values
            mask_lon = f.lon.values
        
        # round used to fix rounding issues in emissions netcdf lat/lons
        for a,la in enumerate(np.round(lat,3)):
            for b,lo in enumerate(np.round(lon,3)):
                if la in mask_lat and lo in mask_lon:
                    lat_id = np.where(mask_lat == la)[0]
                    lon_id = np.where(mask_lon == lo)[0]
                    if mask[lat_id,lon_id] == sea_code:
                        var_plot[:,a,b] = np.nan
                        
    if var_plot.shape[0] == 1:
        n_cols = 1
        n_rows = 1
    elif var_plot.shape[0] <= 4:
        n_cols = var_plot.shape[0]
        n_rows = 1
    else:
        n_cols = 4
        n_rows = int(np.ceil(var_plot.shape[0]/4))
        
    row_count = 0
    col_count = 0

    fig,ax = plt.subplots(n_rows,n_cols,figsize=(n_cols*4,n_rows*3),
                   subplot_kw={'projection':cartopy.crs.PlateCarree()})
    
    for i in range(0,var_plot.shape[0],1):

        border_color = c_border
        if n_cols > 1 :
            if n_rows > 1:
                ax_var = ax[row_count,col_count]
            else:
                ax_var = ax[col_count]
        else:
            ax_var = ax

        ax_var.add_feature(cartopy.feature.BORDERS,edgecolor=border_color,linewidth=1.)
        ax_var.coastlines(resolution='50m',color=border_color,linewidth=1.)
        if type(plot_area) == str:
            ax_var.set_extent(region_limits[plot_area])
        elif type(plot_area) == list:    
            ax_var.set_extent(plot_area)
            
        if nid_style_plot == True:
            ax_var.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,linewidth=1, color='gray', alpha=0.5, linestyle='-')

        ax_var.pcolormesh(lon,lat,var_plot.values[i,:,:]*flux_units_scaling,cmap=cmap,vmin=lim[0],vmax=lim[1],shading='nearest')

        if period_all[m] == 'datetime64[M]' :
            print_time = f'{var_plot.time.values[i].astype(str)[5:7]}-{var_plot.time.values[i].astype(str)[0:4]}'
        elif period_all[m] == 'datetime64[Y]':
            print_time = f'{var_plot.time.values[i].astype(str)[0:4]}'
        
        ax_var.set_title(print_time)

        if plot_site_locations == True:
            if sites_info[m] is not None:
                for s in sites_info[m]:
                    if nid_style_plot == True:
                        ax_var.scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],facecolor='none',
                                        edgecolor='darkblue',marker='^',s=30,zorder=2)
                        ax_var.annotate(s,xy=(sites_info[m][s]['longitude']+0.5,sites_info[m][s]['latitude']-0.5),
                                        color='darkblue',weight='bold')
                    else:
                        ax_var.scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='white',
                                        edgecolor='none',marker='o',s=30,zorder=2,alpha=0.5)
                        ax_var.scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='none',
                                    edgecolor='black',marker='o',s=30,zorder=2)
            
        if plot_point_markers is not None:
            if nid_style_plot == True:
                marker_edge_color = 'purple'
                marker_fill_color = 'None'
                if plot_area == 'AREUROPE':
                    marker_s = 10
                else:
                    marker_s = 20
            else:
                marker_edge_color = 'black'
                marker_fill_color = 'black'
                marker_s = 2

            if i == 0:
                print(f'\nPlotting markers for: {plot_point_markers}')
                print(f'Edit lines below line {inspect.getframeinfo(inspect.currentframe()).lineno} to change marker colour and size')
            for p in plot_point_markers:
                if type(p) == list:
                    ax_var.scatter(p[0],p[1],color=marker_fill_color,edgecolor=marker_edge_color,marker='o',s=marker_s,zorder=2)
                elif type(p) == str:
                    if p not in point_source_dict.keys():
                        print(f'{p} is not specified in point_source_dict, edit this to add a lat/lon location.')
                    else:
                        ax_var.scatter(point_source_dict[p][0],point_source_dict[p][1],color=marker_fill_color,
                                       edgecolor=marker_edge_color,marker='o',s=marker_s,zorder=2)

        if (i+1) % 4 == 0:
            row_count += 1
            col_count = 0
        else:
            col_count += 1

    print('\nEdit flux_lim_kgkm2yr variable in species_info.json to adjust colourbar limits.'+
          'You will need to rerun the first cell of the notebook to apply the adjustment\n')
            
    #flux colorbar
    cbar = plt.cm.ScalarMappable(cmap=cmap)
    levels = np.linspace(lim[0],lim[1])
    cbar.set_array(levels)
    cbar.set_clim(lim)

    # Size of color bar
    f_height = 0.9
    f_bottom = (1-f_height)/2
    f_width = 0.02 #0.02
    if plot_area == 'AREUROPE':
        f_left = 0.98         #0.94
    else:
        f_left = 0.95        #0.94

    cbar_ax = fig.add_axes([f_left, f_bottom, f_width, f_height])
    color_bar = fig.colorbar(cbar,cax=cbar_ax,orientation='vertical',cmap=cmap,extend='max')
        
    color_bar.ax.tick_params(labelsize=10)
        
    if nid_style_plot == True:
        
        nbins = np.ceil(np.max(levels)-np.min(levels)).astype(int)
        if nbins > 11:
            nbins = 11
        if nbins < 5 :
            nbins = 5
        
        nbins = 6
        
        tick_locator = ticker.MaxNLocator(nbins=nbins)#,integer=True)
        color_bar.locator = tick_locator
        color_bar.update_ticks()
        cbar_ax.annotate(f'{cb_units}',xy=(3.7,0.3),xycoords='axes fraction',rotation=90)   #hardcode position of cb label, to fix plot dimensions
        #color_bar.set_label(f'{cb_units}',labelpad=0.05)
        
    else:
        color_bar.set_label(f'{var_labels[var]} {s_data[species]["species_print"]} ({cb_units})')
    
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.04, hspace=0.12)
    
    return fig

##############################################################################################################

def plot_obs_modelled_separate(ds_all,species,site,
                               model_colors,s_data,m_data,annotate_coords,ppt_mode=False,
                             include=['Yobs','Yapriori','Yapost'],
                             diff_include=['Yapriori','Yapost'],
                             add_unc=True,
                             y_lim=None):
    """
    Timeseries plots of observations and modelled mole fractions or 
    baselines from each model.
    Also includes a histogram for each model, showing the difference between
    the prior and posterior fit to the observations.
    
    Args:
        ds_all (dictionary of datasets):
            xarray datasets, scaled and sliced between chosen dates and for 
            chosen site.
        species (str): 
            Gas species, e.g. 'ch4'.
        site (str):
            Obs site, e.g. 'MHD'.
        model_colors (dict of str):
            Models and corresponding colours used to plot the model.
        s_data (dict of dict):
            Dictionary of species with information for plotting (read from json file).
        m_data (dict of dict):
            Dictionary of inversion runs with filename and plot label (read from json file).
        annotate_coords (dict of lists):
            Coordinates to annotate histogram.
        ppt_mode (logical) (optional):
            If True, adjust annotation position and xlabel rotation to accomodate bigger fonts.
        include (list of str):
            Variables included in the plot, options for 'Yobs', 'Yapriori',
            'Yapost', 'YaprioriBC', 'YapostBC'.
        diff_include (list of str):
            Variables included in the 'obs - variable' difference histogram, 
            same options as above.
        add_unc (bool):
            if True, plot uncertainty bar on Yobs and Yapost timeseries.
        y_lim (list of float, optional):
            Mix/max y axis limits to apply to all plots.
    Returns:
        fig (figure): 
            A timeseries and histogram plot for each model included.
    """
        
    var_labels = {'Yapriori':'prior mf',
                  'Yapost':'posterior mean mf',
                  'YaprioriBC':'prior baseline',
                  'YapostBC':'posterior mean baseline',
                  'Yapriori_bias':'prior bias',
                  'Yapost_bias':'posterior bias',
                  'YaprioriOUTER':'prior outer region mf',
                  'YapostOUTER':'posterior outer region mf',
                  'Yobs':'observed mf',
                  'uYobs_repeatability':'obs repeatability mf uncertainty',
                  'uYobs_variability':'obs variability mf uncertainty',
                  'uYmod':'model uncertainty',
                  'uYtotal':'total uncertainty'}
    var_colors = {'Yapriori':1,
                  'Yapost':0,
                  'YaprioriBC':1,
                  'YapostBC':0,
                  'Yapriori_bias':0,
                  'Yapost_bias':1,
                  'YaprioriOUTER':1,
                  'YapostOUTER':0,
                  'Yobs':0,
                  'uYobs_repeatability':0,
                  'uYobs_variability':0,
                  'uYmod':1,
                  'uYtotal':1}
        
    models = ds_all.keys()
    min_mf = []
    max_mf = []
    ax_all = []
    ax2_all = []
        
    fig = plt.figure(constrained_layout=True,figsize=(15,len(models)*3))
    gs = fig.add_gridspec(len(models),2,width_ratios=[0.8,0.2])
    
    for i,m in enumerate(models):
        
        m0 = m.split('_')[0]
        
        ax = fig.add_subplot(gs[i,0])
        ax2 = fig.add_subplot(gs[i,1])
        ax_all.append(ax)
        ax2_all.append(ax2)
        
        for var in include:

            if var == 'Yobs':
                if len(include) == 1:
                    ax.scatter(ds_all[m].time.values,
                               ds_all[m]['Yobs'].values,
                               color=model_colors[m][var_colors[var]],
                               label=f'Obs ({m_data[m]["label"]})',s=8,alpha=0.8,marker='s')
                    
                    if add_unc:
                        try:
                            ax.errorbar(ds_all[m].time.values,
                                        ds_all[m]['Yobs'].values,
                                        ds_all[m]['uYobs_repeatability'].values,
                                        color=model_colors[m][var_colors[var]],alpha=0.4,fmt='none')

                        except:
                            #handle old ncdf files
                            ax.errorbar(ds_all[m].time.values,
                                        ds_all[m]['Yobs'].values,
                                        ds_all[m]['uYobs'].values,
                                        color=model_colors[m][var_colors[var]],alpha=0.4,fmt='none')
                            print(f'WARNING: uYobs_repeatability is not present in {m}. uYobs is being plotted instead as error bars.')

                else:
                    
                    ax.scatter(ds_all[m].time.values,
                                ds_all[m]['Yobs'].values,
                                color='grey',label=f'Obs ({m_data[m]["label"]})',s=8,alpha=0.8,
                                marker='s')
                    
                    if add_unc:
                        try:
                            ax.errorbar(ds_all[m].time.values,
                                        ds_all[m]['Yobs'].values,
                                        ds_all[m]['uYobs_repeatability'].values,
                                        color='black',alpha=0.4,fmt='none')
                        except:
                            #handle old ncdf files
                            ax.errorbar(ds_all[m].time.values,
                                        ds_all[m]['Yobs'].values,
                                        ds_all[m]['uYobs'].values,
                                        color='black',alpha=0.4,fmt='none')
                            print(f'WARNING: uYobs_repeatability is not present in {m}. uYobs is being plotted instead as error bars.')

            else:
                try:
                    ax.plot(ds_all[m].time.values,
                            ds_all[m][var].values,
                            color=model_colors[m][var_colors[var]],alpha=0.8,
                            linewidth=2.,
                            label=f'{m_data[m]["label"]} {var_labels[var]}')
                
                except:
                    #handle old ncdf files
                    if var == 'uYmod':
                        uYmod = ds_all[m]['Yobs'].values - ds_all[m]['qYmod'].values[:,model_q_indices[m0][0]]
                        ax.plot(ds_all[m].time.values,
                                uYmod,
                                color=model_colors[m][var_colors[var]],alpha=0.8,
                                linewidth=2.,
                                label=f'{m_data[m]["label"]} {var_labels[var]}')
                        print(f'WARNING: uYmod is not present in {m}. This quantity is being computed from qYmod.')

                    elif var == 'uYobs_repeatability':
                        ax.plot(ds_all[m].time.values,
                                ds_all[m]['uYobs'].values,
                                color=model_colors[m][var_colors[var]],alpha=0.8,
                                linewidth=2.,
                                label=f'{m_data[m]["label"]} {var_labels[var]}')
                        print(f'WARNING: uYobs_repeatability is not present in {m}. uYobs is being plotted instead.')

                    else:
                        print(f'ERROR: variable {var} not found in {m} or deprecated!')

                if (var == 'Yapost') and add_unc:
                    ax.fill_between(ds_all[m].time.values,
                                    ds_all[m]['qYapost'].values[:,model_q_indices[m0][0]],
                                    ds_all[m]['qYapost'].values[:,model_q_indices[m0][1]],
                                    color=model_colors[m][var_colors[var]],alpha=0.2)
                #if var == 'YapostBC':
                #    ax.fill_between(ds_all[m].time.values,
                #                    ds_all[m]['qYapostBC'].values[:,model_q_indices[m][0]],
                #                    ds_all[m]['qYapostBC'].values[:,model_q_indices[m][1]],
                #                    color=model_colors[m][var_colors[var]],alpha=0.5)

        # Plot histogram
        if len(diff_include) == 0:
            make_diff   = False
            vars        = include
            legend_hist = 'Modelled mean'

        else:
            make_diff   = True
            vars        = diff_include
            legend_hist = 'Obs - modelled mean'

        for i,var in enumerate(vars):
            
            if make_diff:
                var_plot = ds_all[m]['Yobs'].values - ds_all[m][var].values
            else:
                try:
                    var_plot = ds_all[m][var].values
                except:
                    if var == 'uYmod':
                        var_plot = uYmod
                    elif var == 'uYobs_repeatability':
                        var_plot = ds_all[m]['uYobs'].values
                    else:
                        continue

            if np.nanmean(var_plot) <= 0.01:
                var_mean = np.round(np.nanmean(var_plot),5)
                var_sd = np.round(np.nanstd(var_plot),5)
            else:
                var_mean = np.round(np.nanmean(var_plot),2)
                var_sd = np.round(np.nanstd(var_plot),2)

            a,b,c = ax2.hist(var_plot,bins=30,color=model_colors[m][var_colors[var]],density=1)
            if make_diff:
                ax2.vlines(0,0,np.max(a),color='dimgrey',linewidth=3.)
            
            with np.printoptions(precision=2, suppress=True):

                ax2.annotate(r'$\mu$: '+str(var_mean)+'\n'+r'$\sigma$: '+str(var_sd),xy=annotate_coords[i],
                                xycoords='axes fraction',color=model_colors[m][var_colors[var]])

        # Write number of obs to plot
        n_obs = (~np.isnan(ds_all[m]['Yobs'].values)).sum()
        if (ppt_mode):
            pos_xy = [0.57,1.05]
        else:
            pos_xy = [0.65,1.05]
        ax2.annotate('\n$N_{obs}$: '+str(n_obs),xy=pos_xy,xycoords='axes fraction',color='k')

        ax2.set_xlabel(legend_hist)
    
        min_mf.append(ax.get_ylim()[0])
        max_mf.append(ax.get_ylim()[1])
        
        ax.set_title(m_data[m]["label"])
        ax.set_ylabel(f'{s_data[species]["species_print"]} {site} ({s_data[species]["mf_units_print"]})')
        leg = ax.legend(ncol=2,borderpad=.2,columnspacing=1.0)
        try:
            for l in leg.legend_handles:
                l.set_linewidth(5.0)
        except:
            for l in leg.legendHandles:
                l.set_linewidth(5.0)
                
        if int(ds_all[m].time.values[-1].astype('datetime64[M]')-ds_all[m].time.values[0].astype('datetime64[M]')) > 12:
            ax.xaxis.set_minor_locator(MonthLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.xaxis.set_major_locator(YearLocator())
        else:
            ax.xaxis.set_major_locator(MonthLocator())
            if (ppt_mode):
                ax.tick_params(axis='x', rotation=70)
                    
    if y_lim == None:    
        for i in range(len(models)):
            ax_all[i].set_ylim([min(min_mf)-(0.02*min(min_mf)),
                                max(max_mf)+(0.05*max(max_mf))])
    else:
        for i in range(len(models)):
            ax_all[i].set_ylim(y_lim)
            
    print('NOTE: If all the data is not within axis limits, adjust the set_ylim')
    print('NOTE: If annotations in the histograms are not displaying correctly, adjust annotate_coords.')
    
    return fig

#####################################################################

def plot_obs_modelled_together(ds_all,species,site,
                               model_colors,s_data,m_data,annotate_coords,ppt_mode=False,
                               include=['Yapost'],
                               diff_include=['Yapost'],
                               add_unc=True,
                               y_lim=None):
    """
    Timeseries plots of observations and modelled mole fractions or 
    baselines from each model, all on one plot.
    Also includes a histogram for each model, showing the difference between
    the prior and posterior fit to the observations.
    
    Args:
        ds_all (dictionary of datasets):
            xarray datasets, scaled and sliced between chosen dates and for 
            chosen site.
        species (str): 
            Gas species, e.g. 'ch4'.
        site (str):
            Obs site, e.g. 'MHD'.
        model_colors (dict of str):
            Models and corresponding colours used to plot the model.
        s_data (dict of dict):
            Dictionary of species with information for plotting (read from json file).
        m_data (dict of dict):
            Dictionary of inversion runs with filename and plot label (read from json file).
        annotate_coords (dict of lists):
            Coordinates to annotate histogram.
        ppt_mode (logical) (optional):
            If True, adjust xlabel rotation to accomodate bigger fonts.
        include (list of str):
            Variables included in the plot, options for 'Yobs', 'Yapriori',
            'Yapost', 'YaprioriBC', 'YapostBC'.
        diff_include (list of str):
            Variables included in the 'obs - variable' difference histogram, 
            same options as above.
        add_unc (bool):
            if True, plot uncertainty bar on Yobs and Yapost timeseries.
        y_lim (list of float, optional):
            Mix/max y axis limits to apply to all plots.
    Returns:
        fig (figure): 
            One timeseries and histogram plot containing data from all models.
    """

    var_labels = {'Yapriori':'prior mf',
                  'Yapost':'posterior mean mf',
                  'YaprioriBC':'prior baseline',
                  'YapostBC':'posterior mean baseline',
                  'Yapriori_bias':'prior bias',
                  'Yapost_bias':'posterior bias',
                  'YaprioriOUTER':'prior outer region mf',
                  'YapostOUTER':'posterior outer region mf',
                  'Yobs':'observed mf',
                  'uYobs_repeatability':'obs repeatability mf uncertainty',
                  'uYobs_variability':'obs variability mf uncertainty',
                  'uYmod':'model uncertainty',
                  'uYtotal':'total uncertainty'}
    var_colors = {'Yapriori':1,
                  'Yapost':0,
                  'YaprioriBC':1,
                  'YapostBC':0,
                  'Yapriori_bias':0,
                  'Yapost_bias':1,
                  'YaprioriOUTER':1,
                  'YapostOUTER':0,
                  'Yobs':0,
                  'uYobs_repeatability':0,
                  'uYobs_variability':0,
                  'uYmod':0,
                  'uYtotal':0}
        
    models = ds_all.keys()
    min_mf = []
    max_mf = []
        
    fig = plt.figure(constrained_layout=True,figsize=(15,7))
    gs = fig.add_gridspec(len(models),2,width_ratios=[0.8,0.2])

    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for i,m in enumerate(models):
        
        m0 = m.split('_')[0]
                
        for var in include:

            if var == 'Yobs':
                if len(include) == 1:
                    ax.scatter(ds_all[m].time.values,
                                ds_all[m]['Yobs'].values,
                                color=model_colors[m][var_colors[var]],label=f'Obs ({m_data[m]["label"]})',s=5,alpha=0.5)

                    if add_unc:
                        try:
                            ax.errorbar(ds_all[m].time.values,
                                        ds_all[m]['Yobs'].values,
                                        ds_all[m]['uYobs_repeatability'].values,
                                        color=model_colors[m][var_colors[var]],alpha=0.4,fmt='none')

                        except:
                            #handle old ncdf files
                            ax.errorbar(ds_all[m].time.values,
                                        ds_all[m]['Yobs'].values,
                                        ds_all[m]['uYobs'].values,
                                        color=model_colors[m][var_colors[var]],alpha=0.4,fmt='none')
                            print(f'WARNING: uYobs_repeatability is not present in {m}. uYobs is being plotted instead as error bars.')

                else:
                    ax.scatter(ds_all[m].time.values,
                                ds_all[m]['Yobs'].values,
                                color='dimgrey',label=f'Obs ({m_data[m]["label"]})',s=5,alpha=0.5)

            else:
                try:
                    ax.scatter(ds_all[m].time.values,
                            ds_all[m][var].values,
                            color=model_colors[m][var_colors[var]],alpha=0.5,
                            label=f'{m_data[m]["label"]} {var_labels[var]}',
                            linewidth=2,s=5)

                except:
                    # handle old ncdf files
                    if var == 'uYmod':
                        uYmod = ds_all[m]['Yobs'].values - ds_all[m]['qYmod'].values[:,model_q_indices[m0][0]]
                        ax.scatter(ds_all[m].time.values,
                                   uYmod,
                                   color=model_colors[m][var_colors[var]],
                                   label=f'{m_data[m]["label"]} {var_labels[var]}',
                                   linewidth=2,s=5,alpha=0.5)
                        print(f'WARNING: uYmod is not present in {m}. This quantity is being computed from qYmod.')

                    elif var == 'uYobs_repeatability':
                        ax.scatter(ds_all[m].time.values,
                                   ds_all[m]['uYobs'].values,
                                   color=model_colors[m][var_colors[var]],
                                   label=f'{m_data[m]["label"]} {var_labels[var]}',
                                   linewidth=2.,s=5,alpha=0.5)
                        print(f'WARNING: uYobs_repeatability is not present in {m}. uYobs is being plotted instead.')

                    else:
                        print(f'ERROR: variable {var} not found in {m} or deprecated!')

                if (var == 'Yapost') and add_unc:
                    ax.fill_between(ds_all[m].time.values,
                                    ds_all[m]['qYapost'].values[:,model_q_indices[m0][0]],
                                    ds_all[m]['qYapost'].values[:,model_q_indices[m0][1]],
                                    color=model_colors[m][var_colors[var]],alpha=0.3)
                #if var == 'YapostBC':
                #    ax.fill_between(ds_all[m].time.values,
                #                    ds_all[m]['qYapostBC'].values[:,model_q_indices[m][0]],
                #                    ds_all[m]['qYapostBC'].values[:,model_q_indices[m][1]],
                #                    color=model_colors[m][var_colors[var]],alpha=0.5)
        

        # Plot histogram
        if len(diff_include) == 0:
            make_diff   = False
            vars        = include
            legend_hist = 'Modelled mean'

        else:
            make_diff   = True
            vars        = diff_include
            legend_hist = 'Obs - modelled mean'

        for v,var in enumerate(vars):
            
            if make_diff:
                var_plot = ds_all[m]['Yobs'].values - ds_all[m][var].values
            else:
                try:
                    var_plot = ds_all[m][var].values
                except:
                    if var == 'uYmod':
                        var_plot = uYmod
                    elif var == 'uYobs_repeatability':
                        var_plot = ds_all[m]['uYobs'].values
                    else:
                        continue

            if np.nanmean(var_plot) <= 0.01:
                var_mean = np.round(np.nanmean(var_plot),5)
                var_sd = np.round(np.nanstd(var_plot),5)
            else:
                var_mean = np.round(np.nanmean(var_plot),2)
                var_sd = np.round(np.nanstd(var_plot),2)
            
            a,b,c = ax2.hist(var_plot,bins=30,color=model_colors[m][var_colors[var]],density=1,alpha=0.7)
            if make_diff:
                ax2.vlines(0,0,np.max(a),color='dimgrey',linewidth=3.)
            
            with np.printoptions(precision=2, suppress=True):

                ax2.annotate(r'$\mu$: '+str(var_mean)+'\n'+r'$\sigma$: '+str(var_sd),xy=annotate_coords[i],
                                xycoords='axes fraction',color=model_colors[m][var_colors[var]])
        
    ax2.set_xlabel(legend_hist)

    min_mf.append(ax.get_ylim()[0])
    max_mf.append(ax.get_ylim()[1])
    
    ax.set_title('All models')
    ax.set_ylabel(f'{s_data[species]["species_print"]} {site} ({s_data[species]["mf_units_print"]})')
    leg = ax.legend(ncol=2,borderpad=.2,columnspacing=1.0)
    try:
        for l in leg.legend_handles:
            l.set_linewidth(5.0)
    except:
        for l in leg.legendHandles:
            l.set_linewidth(5.0)
    
    if int(ds_all[m].time.values[-1].astype('datetime64[M]')-ds_all[m].time.values[0].astype('datetime64[M]')) > 12:
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_locator(YearLocator())
    else:
        ax.xaxis.set_major_locator(MonthLocator())
        if (ppt_mode):
            ax.tick_params(axis='x', rotation=70)
        
    if y_lim is None:
        ax.set_ylim([min(min_mf)-(0.02*min(min_mf)),
                                max(max_mf)+(0.05*max(max_mf))])
    else:
        ax.set_ylim(y_lim)
        
    print('NOTE: If all the data is not within axis limits, adjust the set_ylim')
    print('NOTE: If annotations in the histograms are not displaying correctly, adjust annotate_coords.')
    
    return fig

#####################################################################

def plot_obs_diff(ds_all,species,site,
                               model_colors,s_data,m_data,annotate_coords,ppt_mode=False,
                               include=['Yapost'],
                               diff_include=['Yapost'],
                               y_lim=None):
    """
    Plot of the absolute difference between variables from two models.
    Also includes a histogram for each model, showing the difference between
    the 'include' variables fit to the observations.
    
    If more than two models are included in ds_all, only the first two
    models will be plotted.
    
    Args:
        ds_all (dictionary of datasets):
            xarray datasets, scaled and sliced between chosen dates and for 
            chosen site.
        species (str): 
            Gas species, e.g. 'ch4'.
        site (str):
            Obs site, e.g. 'MHD'.
        model_colors (dict of str):
            Models and corresponding colours used to plot the model.
        s_data (dict of dict):
            Dictionary of species with information for plotting (read from json file).
        m_data (dict of dict):
            Dictionary of inversion runs with filename and plot label (read from json file).
        annotate_coords (dict of lists):
            Coordinates to annotate histogram.
        ppt_mode (logical) (optional):
            If True, adjust xlabel rotation to accomodate bigger fonts.
        include (list of str):
            Variables included in the plot, options for 'Yobs', 'Yapriori',
            'Yapost', 'YaprioriBC', 'YapostBC'.
        diff_include (list of str):
            Variables included in the 'obs - variable' difference histogram, 
            same options as above.
        y_lim (list of float, optional):
            Mix/max y axis limits to apply to all plots.
    Returns:
        fig (figure): 
            A timeseries and histogram plot for each model included.
    """

    var_labels = {'Yapriori':'prior mf',
                  'Yapost':'posterior mean mf',
                  'YaprioriBC':'prior baseline',
                  'YapostBC':'posterior mean baseline',
                  'Yapriori_bias':'prior bias',
                  'Yapost_bias':'posterior bias',
                  'YaprioriOUTER':'prior outer region mf',
                  'YapostOUTER':'posterior outer region mf',
                  'Yobs':'observed mf',
                  'uYobs_repeatability':'obs repeatability mf uncertainty',
                  'uYobs_variability':'obs variability mf uncertainty',
                  'uYmod':'model uncertainty',
                  'uYtotal':'total uncertainty'}
    var_colors = {'Yapriori':1,
                  'Yapost':0,
                  'YaprioriBC':1,
                  'YapostBC':0,
                  'Yapriori_bias':0,
                  'Yapost_bias':1,
                  'YaprioriOUTER':1,
                  'YapostOUTER':0,
                  'Yobs':0,
                  'uYobs_repeatability':0,
                  'uYobs_variability':0,
                  'uYmod':1,
                  'uYtotal':1}
        
    models = list(ds_all.keys())
    min_mf = []
    max_mf = []
        
    fig = plt.figure(constrained_layout=True,figsize=(15,7))
    gs = fig.add_gridspec(len(models),2,width_ratios=[0.8,0.2])

    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    both_times0 = np.isin(ds_all[models[0]].time.values,ds_all[models[1]].time.values)
    both_times1 = np.isin(ds_all[models[1]].time.values,ds_all[models[0]].time.values)
    
    ds_all[models[0]] = ds_all[models[0]].sel(time=both_times0)
    ds_all[models[1]] = ds_all[models[1]].sel(time=both_times1)
    
            
    for var in include:
        try:
            ax.scatter(ds_all[models[0]].time.values,
                       ds_all[models[0]][var].values - ds_all[models[1]][var].values,
                       color=model_colors[models[0]][var_colors[var]],alpha=0.5,
                       label=f'{m_data[models[0]]["label"]} - {m_data[models[1]]["label"]}\n{var_labels[var]}',
                       linewidth=2,s=8)

        except:
            # handle old ncdf files
            if var == 'uYmod':
                m00 = models[0].split('_')[0]
                m01 = models[1].split('_')[0]

                try:
                    uYmod0 = ds_all[models[0]]['Yobs'].values - ds_all[models[0]]['qYmod'].values[:,model_q_indices[m00][0]]
                    uYmod1 = ds_all[models[1]]['Yobs'].values - ds_all[models[1]]['qYmod'].values[:,model_q_indices[m01][0]]

                    ax.scatter(ds_all[models[0]].time.values,
                                uYmod0 - uYmod1,
                                color=model_colors[models[0]][var_colors[var]],alpha=0.5,
                                label=f'{m_data[models[0]]["label"]} - {m_data[models[1]]["label"]}\n{var_labels[var]}',
                                linewidth=2,s=8)
                    print(f'WARNING: uYmod is not present in both models. This quantity is being computed from qYmod.')

                except:
                    print(f'ERROR: {models[0]} and {models[1]} have different definitions of uYmod!')

            elif var == 'uYobs_repeatability':
                try:
                    ax.scatter(ds_all[models[0]].time.values,
                                ds_all[models[0]]['uYobs'].values - ds_all[models[1]]['uYobs'].values,
                                color=model_colors[models[0]][var_colors[var]],alpha=0.5,
                                label=f'{m_data[models[0]]["label"]} - {m_data[models[1]]["label"]}\n{var_labels[var]}',
                                linewidth=2,s=8)
                    print(f'WARNING: uYobs_repeatability is not present in both models. uYobs is being plotted instead.')

                except:
                    print(f'ERROR: {models[0]} and {models[1]} have different definitions of uYobs!')

            else:
                print(f'ERROR: variable {var} not found or deprecated in {models[0]} or {models[1]}!')

        #if var == 'Yapost':
        #    ax.fill_between(ds_all[m].time.values,
        #                    ds_all[m]['qYapost'].values[:,model_q_indices[m0][0]],
        #                    ds_all[m]['qYapost'].values[:,model_q_indices[m0][1]],
        #                    color=model_colors[m][var_colors[var]],alpha=0.3)
                #if var == 'YapostBC':
                #    ax.fill_between(ds_all[m].time.values,
                #                    ds_all[m]['qYapostBC'].values[:,model_q_indices[m][0]],
                #                    ds_all[m]['qYapostBC'].values[:,model_q_indices[m][1]],
                #                    color=model_colors[m][var_colors[var]],alpha=0.5)
        
    for i,m in enumerate(models):
        
        m0 = m.split('_')[0]

        # Plot histogram
        if len(diff_include) == 0:
            make_diff   = False
            vars        = include
            legend_hist = 'Modelled mean'

        else:
            make_diff   = True
            vars        = diff_include
            legend_hist = 'Obs - modelled mean'

        for v,var in enumerate(vars):

            if make_diff:
                var_plot = ds_all[m]['Yobs'].values - ds_all[m][var].values
            else:
                try:
                    var_plot = ds_all[m][var].values
                except:
                    if var == 'uYmod':
                        var_plot = ds_all[m]['Yobs'].values - ds_all[m]['qYmod'].values[:,model_q_indices[m0][0]]
                    elif var == 'uYobs_repeatability':
                        var_plot = ds_all[m]['uYobs'].values
                    else:
                        continue
            
            if np.nanmean(var_plot) <= 0.01:
                var_mean = np.round(np.nanmean(var_plot),5)
                var_sd = np.round(np.nanstd(var_plot),5)
            else:
                var_mean = np.round(np.nanmean(var_plot),2)
                var_sd = np.round(np.nanstd(var_plot),2)
            
            a,b,c = ax2.hist(var_plot,bins=30,color=model_colors[m][var_colors[var]],density=1,alpha=0.7)
            if make_diff:
                ax2.vlines(0,0,np.max(a),color='dimgrey',linewidth=3.)
            
            with np.printoptions(precision=2, suppress=True):

                ax2.annotate(r'$\mu$: '+str(var_mean)+'\n'+r'$\sigma$: '+str(var_sd),xy=annotate_coords[i],
                                xycoords='axes fraction',color=model_colors[m][var_colors[var]])
        
    ax2.set_xlabel(legend_hist)

    min_mf.append(ax.get_ylim()[0])
    max_mf.append(ax.get_ylim()[1])
    
    ax.set_title(f'{m_data[models[0]]["label"]} - {m_data[models[1]]["label"]}')
    ax.set_ylabel(f'{s_data[species]["species_print"]} {site} ({s_data[species]["mf_units_print"]})')
    leg = ax.legend(ncol=2,borderpad=.2,columnspacing=1.0)
    try:
        for l in leg.legend_handles:
            l.set_linewidth(5.0)
    except:
        for l in leg.legendHandles:
            l.set_linewidth(5.0)
    
    if int(ds_all[m].time.values[-1].astype('datetime64[M]')-ds_all[m].time.values[0].astype('datetime64[M]')) > 12:
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_locator(YearLocator())
    else:
        ax.xaxis.set_major_locator(MonthLocator())
        if (ppt_mode):
            ax.tick_params(axis='x', rotation=70)
        
    if y_lim is None:
        ax.set_ylim([min(min_mf)-(0.02*min(min_mf)),
                                max(max_mf)+(0.05*max(max_mf))])
    else:
        ax.set_ylim(y_lim)
        
    print('NOTE: If all the data is not within axis limits, adjust the set_ylim')
    print('NOTE: If annotations in the histograms are not displaying correctly, adjust annotate_coords.')
    
    return fig

#####################################################################

def plot_stats_mf(pearson,nrmse,species,
                  model_colors,s_data,m_data,
                  start_date=None,end_date=None):
    """
    Plots fit statistics for all sites, for all models.
    
    Args:
        pearson (dictionary of dictionaries):
            Pearson correlation coeffiecient, for each site and for each model.
        nrmse (dictionary of dictionaries):
            Normalised root mean square error, for each site and for each model.
        species (str): 
            Gas species, e.g. 'ch4'.
        model_colors (dict of str):
            Models and corresponding colours used to plot the model.
        s_data (dict of dict):
            Dictionary of species with information for plotting (read from json file).
        m_data (dict of dict):
            Dictionary of inversion runs with filename and plot label (read from json file).
        start_date (str) and end_date (str):
            Dates used to title the plot. 
    Returns:
        fig (figure): 
            Two plots showing each model's fit statistics, for each site.
    """
    
    x_val = []
    x_label = []

    model_colors_stats = {'intem':'dodgerblue',
                        'elris_name':'purple'}

    fig,ax = plt.subplots(2,1,figsize=(10,6),tight_layout=True)
    
    for i,site in enumerate(pearson.keys()):
        for m,model in enumerate(pearson[site]):
            model0 = model.split('_')[0]
            if i == 0:
                ax[0].scatter(i+m*0.2,pearson[site][model],color=model_colors[model][0],marker='x',s=150,label=m_data[model]["label"])
                ax[1].scatter(i+m*0.2,nrmse[site][model],color=model_colors[model][0],marker='x',s=150,label=m_data[model]["label"])
                #ax[2].scatter(i+m*0.2,std[site][model],color=model_colors_stats[model],marker='x',s=150,label=m_data[model]["label"])
                
            else:
                ax[0].scatter(i+m*0.2,pearson[site][model],color=model_colors[model][0],marker='x',s=150)
                ax[1].scatter(i+m*0.2,nrmse[site][model],color=model_colors[model][0],marker='x',s=150)
                #ax[2].scatter(i+m*0.2,std[site][model],color=model_colors_stats[model],marker='x',s=150)
                
        x_val.append(i)
        x_label.append(site)
        
    #y_lim0 = [ax[0].get_ylim()[0],ax[0].get_ylim()[1]]
    #y_lim1 = [ax[0].get_ylim()[0],ax[0].get_ylim()[1]]
    
    for i in range(2):
        ax[i].set_xticks(x_val);
        ax[i].set_xticklabels(x_label,rotation=45);
        ax[i].set_xlim(x_val[0]-0.2,x_val[-1]+0.4)
        #y_lim = [ax[i].get_ylim()[0],ax[i].get_ylim()[1]]
        #ax[i].set_ylim([y_lim[0]-0.1*y_lim[0],y_lim[1]+y_lim[1]*0.1])
                
    ax[0].invert_yaxis()
    ax[0].hlines(1,x_val[0]-0.2,x_val[-1]+0.4,linestyle='dotted',color='grey')        
    ax[1].hlines(0,x_val[0]-0.2,x_val[-1]+0.4,linestyle='dotted',color='grey')        
    
    ax[0].set_ylabel('Pearson\n correlation coefficient')
    ax[1].set_ylabel('Normalised RMSE')
    #ax[2].set_ylabel('Standard\ndeviation')

    leg = ax[0].legend(ncol=2,borderpad=.2,columnspacing=1.0)
    try:
        for l in leg.legend_handles:
            l.set_linewidth(5.0)
    except:
        for l in leg.legendHandles:
            l.set_linewidth(5.0)

    fig.suptitle((f'{s_data[species]["species_print"]} Modelled mole fraction statistical fit to obs')+
                 f' \n{start_date} to {end_date}')
    
    
    return fig

#####################################################################

def plot_sites_timeseries(ds_all,var,start_date,end_date,model_colors,m_data):
    """
    Plot the timeseries of data available for each site and model.
    
    Args:
        ds_all : 
            Dictionnary of xarray returned by read_mf
        var : 
            Var for which the timeseries should be plotted
        start_date (str): 
            Date to plot data from, e.g. '2021-01-01'
        end_date (str): 
            Date to plot data to, e.g. '2022-01-01' would include all
            data up to 2021-12-31.
        model_colors (dict of str):
            Models and corresponding colours used to plot the model.
        m_data (dict of dict):
            Dictionary of inversion runs with filename and plot label (read from json file).
    """
    siteList = np.sort(np.unique(np.concatenate([ds_all[m].sitenames.values.astype(str) for m in ds_all.keys()])))

    fig,ax = plt.subplots(1,1,figsize = (0.7*len(siteList),8))
    leg = []
    for iSite,site in enumerate(siteList):
        if iSite!=0:
            ax.plot([iSite-0.5,iSite-0.5],[np.datetime64(start_date),np.datetime64(end_date)],
                   c='gray',ls='-',lw=1)
        for i,m in enumerate(ds_all.keys()):
            try:
                if m not in leg:
                    site_index = np.where(ds_all[m]['sitenames'].astype(str) == site)[0][0]
                    data = ds_all[m].isel(nsite=site_index)[var].dropna(dim='time').time.values
                    ax.scatter(-2*np.ones(data.size),
                               data,c=model_colors[m][0],s=20,label=m_data[m]["label"])
                    leg.append(m)

                site_index = np.where(ds_all[m]['sitenames'].astype(str) == site)[0][0]
                data = ds_all[m].isel(nsite=site_index)[var].dropna(dim='time').time.values
                ax.scatter((iSite+0.2*(i-1))*np.ones(data.size),
                           data,c=model_colors[m][0],s=2)

            except:
                pass
    ax.set_ylim(np.datetime64(start_date)-np.timedelta64(1,'D'),
                np.datetime64(end_date)+np.timedelta64(1,'D'))
    
    
    ax.set_xticks(np.arange(siteList.size))
    ax.set_xticklabels(siteList)
    
    
    if int(np.datetime64(end_date).astype('datetime64[M]')-np.datetime64(start_date).astype('datetime64[M]')) > 12:
        ax.yaxis.set_minor_locator(MonthLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(YearLocator())
    else:
        ax.yaxis.set_major_locator(MonthLocator())
    ax.yaxis.grid(True, which='major')
    
    ax.set_xlim(-1,siteList.size)
        
    plt.legend(loc='upper right')
    
    return fig

#####################################################################

def plot_spatial_flux(ds_all,species,plot_area,s_data,m_data,cmap=None,
                      cmap_diff=None,c_border=None,period_override=None,
                      plot_site_locations=False,plot_point_markers=None,
                      season=None,plot_inversion_grid_flux=False,sites_available=None):
    """
    Plots posterior and prior fluxes and the difference between these
    for all models.
    
    If ds_all contains mulitple time periods for each model, the average 
    across all times will be plotted.
    
    Args:
        ds_all (dictionary of datasets):
            xarray datasets of fluxes, scaled and sliced between 
            chosen dates.
        species (str): 
            Gas species, e.g. 'ch4'.
        plot_area (str or list):
            Lat/lon region to plot, options for 'UK', 'FRANCE', 'GERMANY',
            'NWEU','CWEU','EUROPE'.
            A list with [min_lon, max_lon, min_lat, max_lat] can also be provided.
        s_data (dict of dict):
            Dictionary of species with information for plotting (read from json file).
        m_data (dict of dict):
            Dictionary of inversion runs with filename and plot label (read from json file).
        cmap (str):
            Colour map for flux plots.
        cmap_diff (str):
            Colour map for difference plots.
        c_border (str):
            Colour for flux plot country borders.
        period_override (list of str, optional):
            Inversion periods to include, to override the standards in species_info.json.
            Must be the same length as models, e.g. ['monthly',None,'yearly']
        plot_site_locations (bool):
            If True, adds triangles with site locations to spatial plot.
        plot_point_markers (list of str or list of lat/lon):
            List of names of points to plot over larger point sources or lat/lon locations.
            See point_markers_dict for a list of options.
            e.g. ['paris','nw_england',[50.,5.]]
        season (string, default None):
            If specified, plot the seasonal mean (only valable for monthly data). 
            Options are 'DJF', 'MAM', 'JJA', 'SON'.
        plot_inversion_grid_flux (bool, default False):
            If True, plots fluxes at the spatial resolution of the inversion (using the 
            inversion_grid variable). If False, plots fluxes at the spatial resolution
            of the prior.
        sites_available (dict of list of str, default None):
            List of 3-letter site codes with data between start_date and end_date, for each model.
            If None, site location will be plotting for all sites used across whole inversion period.
    Returns:
        fig (figure): 
            A plot of spatial flux posterior and prior mean/mode and a plot 
            of the absolute difference between these, for each model.
    """
    
    period_all = {}
    
    for i,m in enumerate(ds_all.keys()):
        m0 = m.split('_')[0]
        if period_override is not None:
            if 'monthly' in period_override[i]:
                period_all[m] = 'datetime64[M]'
            elif period_override[i] == 'yearly':
                period_all[m] = 'datetime64[Y]'
        else:
            period_all[m] = s_data[species]["dt_units"][m0]
    
    if cmap == None:
        cmap = 'viridis' #'Blues'
    if cmap_diff == None:
        cmap_diff = 'coolwarm'
    if c_border == None:
        c_border = 'floralwhite'
    if cmap in ['green','blue','greyscale']:
        cmap = set_colormaps(cmap)
    
    n_cols = len(ds_all.keys())

    region_limits = {'UK':[-12,4,49,62],   #min_lon, max_lon, min_lat, max_lat
                    'FRANCE':[-6,9,42,52],
                    'GERMANY':[2,18,45,60],
                    'ITALY':[6,19,36,48],
                    'SWITZERLAND':[5.5,11,45,49],
                    'NETHERLANDS':[2.5,8,50,55],
                    'IRELAND':[-12,-4,51,56],
                    'HUNGARY':[15,24,44.5,50],
                    'NORWAY':[1,32,55,76],
                    'BENELUX':[1,9,48,55],
                    'NWEU':[-11,11,45,62],
                    'CWEU':[-12,27,37,66],
                    'EUROPE':[-98,40,10,80]}
    
    
    # find site info in netcdf attrs. if none present, use site info from first model with this 
    # data available
    sites_info = {}
    if plot_site_locations == True:
        if sites_available == None:
            print('WARNING: sites_available not supplied, so plotting all sites listed in flux file attrs.')
            for i,m in enumerate(ds_all.keys()):
                try:
                    sites_test = ds_all[m].sites.replace("'","").replace(']','').replace('[','').replace(' ','').split(',')
                    sites_info[m] = extract_site_info(sites_test)
                except:
                    sites_info[m] = None
                    
            for i,m in enumerate(ds_all.keys()):
                if sites_info[m] == None:
                    for j,m2 in enumerate(sites_info.keys()):
                        if sites_info[m2] != None:
                            print(f'No sites data available in {m} attrs, so using site data from {m2}')
                            sites_info[m] = sites_info[m2]
                    break
        else:
            for i,m in enumerate(sites_available.keys()):
                sites_info[m] = extract_site_info(sites_available[m])

    fig,ax = plt.subplots(3,n_cols,constrained_layout=True,figsize=(n_cols*5,9),
                   subplot_kw={'projection':cartopy.crs.PlateCarree()})

    for i in range(3):
        for j in range(n_cols):
            if i == 2:
                border_color = 'dimgrey'
            else:
                border_color = c_border

            if n_cols == 1:
                ax_var = ax[i]
            else:
                ax_var = ax[i,j]

            ax_var.add_feature(cartopy.feature.BORDERS,edgecolor=border_color,linewidth=1.)
            ax_var.coastlines(resolution='50m',color=border_color,linewidth=1.)
            if type(plot_area) == str:
                ax_var.set_extent(region_limits[plot_area])
            elif type(plot_area) == list:    
                ax_var.set_extent(plot_area)

    for i,m in enumerate(ds_all.keys()):
        
        lon = ds_all[m].longitude.values + (ds_all[m].longitude.values[1]-ds_all[m].longitude.values[1])/2
        lat = ds_all[m].latitude.values + (ds_all[m].latitude.values[1]-ds_all[m].latitude.values[1])/2

        m0 = m.split('_')[0]
        
        try:
        
            if len(ds_all[m].time.values) == 1:
                time_out = to_datetime(ds_all[m].time.values[0].astype(s_data[species]["dt_units"][m0])).strftime('%d/%m/%Y')
            else:
                start_print = to_datetime(ds_all[m].time.values[0].astype(period_all[m])).strftime("%d/%m/%Y")
                if period_all[m] == 'datetime64[Y]':
                    end_period = ds_all[m].time.values[-1].astype(period_all[m]) + np.timedelta64(1,'Y') - np.timedelta64(1,'D')                    
                elif period_all[m] == 'datetime64[M]':
                    end_period = ds_all[m].time.values[-1].astype(period_all[m]) + np.timedelta64(1,'M') - np.timedelta64(1,'D')                    
                else:
                    print('This currently only works for monthly or yearly inversion periods. Update the plotting code to print out '+
                            'correct dates for higher frequency inversions.')
                end_print = to_datetime(end_period).strftime("%d/%m/%Y")
                time_out = (f'{start_print} - {end_print}')

            if n_cols == 1:
                ax0 = ax[0]
                ax1 = ax[1]
                ax2 = ax[2]
            else:
                ax0 = ax[0,i]
                ax1 = ax[1,i]
                ax2 = ax[2,i]
            
            if season is None:
                if plot_inversion_grid_flux == True:
                    plot_original = False
                    try:
                        ax0.pcolormesh(lon,lat,
                                    np.mean(ds_all[m]['flux_total_prior'][:,:,:],axis=0),
                                    cmap=cmap,vmin=s_data[species]['fluxlim'][0],vmax=s_data[species]['fluxlim'][1],shading='nearest')

                        ax1.pcolormesh(lon,lat,
                                    np.mean(ds_all[m]['flux_total_posterior_inversion_grid'][:,:,:],axis=0),
                                    cmap=cmap,vmin=s_data[species]['fluxlim'][0],vmax=s_data[species]['fluxlim'][1],shading='nearest')

                        flux_diff = np.mean(ds_all[m]['flux_total_posterior_inversion_grid'][:,:,:],axis=0)-np.mean(ds_all[m]['flux_total_prior'][:,:,:],axis=0)
                    except:
                        print(f'Cannot find inversion_grid variables for {m} so using standard flux output.')
                        plot_original = True
                else:
                    plot_original = True
                        
                if plot_original == True:
                    ax0.pcolormesh(lon,lat,
                                np.mean(ds_all[m]['flux_total_prior'][:,:,:],axis=0),
                                cmap=cmap,vmin=s_data[species]['fluxlim'][0],vmax=s_data[species]['fluxlim'][1],shading='nearest')

                    ax1.pcolormesh(lon,lat,
                                np.mean(ds_all[m]['flux_total_posterior'][:,:,:],axis=0),
                                cmap=cmap,vmin=s_data[species]['fluxlim'][0],vmax=s_data[species]['fluxlim'][1],shading='nearest')

                    flux_diff = np.mean(ds_all[m]['flux_total_posterior'][:,:,:],axis=0)-np.mean(ds_all[m]['flux_total_prior'][:,:,:],axis=0)
                        
                flux_diff[np.where(flux_diff) == np.nan] = 0.
                
            else :
                if plot_inversion_grid_flux == True:
                    plot_original = False
                try:
                    ax0.pcolormesh(lon,lat,
                                ds_all[m]['flux_total_prior'].groupby("time.season").mean().sel(season=season).values,
                                cmap=cmap,vmin=s_data[species]['fluxlim'][0],vmax=s_data[species]['fluxlim'][1],shading='nearest')

                    ax1.pcolormesh(lon,lat,
                                ds_all[m]['flux_total_posterior_inversion_grid'].groupby("time.season").mean().sel(season=season).values,
                                cmap=cmap,vmin=s_data[species]['fluxlim'][0],vmax=s_data[species]['fluxlim'][1],shading='nearest')
                    
                    flux_diff = ds_all[m]['flux_total_posterior_inversion_grid'].groupby("time.season").mean().sel(season=season).values \
                                -ds_all[m]['flux_total_prior'].groupby("time.season").mean().sel(season=season).values
                except:
                    print(f'Cannot find inversion_grid variables for {m} so using standard flux output.')
                    plot_original = True
                    
                if plot_original == True:
                    ax0.pcolormesh(lon,lat,
                                ds_all[m]['flux_total_prior'].groupby("time.season").mean().sel(season=season).values,
                                cmap=cmap,vmin=s_data[species]['fluxlim'][0],vmax=s_data[species]['fluxlim'][1],shading='nearest')

                    ax1.pcolormesh(lon,lat,
                                ds_all[m]['flux_total_posterior'].groupby("time.season").mean().sel(season=season).values,
                                cmap=cmap,vmin=s_data[species]['fluxlim'][0],vmax=s_data[species]['fluxlim'][1],shading='nearest')
                    
                    flux_diff = ds_all[m]['flux_total_posterior'].groupby("time.season").mean().sel(season=season).values \
                                -ds_all[m]['flux_total_prior'].groupby("time.season").mean().sel(season=season).values
                                
                flux_diff[np.where(flux_diff) == np.nan] = 0.
                
                time_out = f'{season} of {time_out}'
                
            
            ax0.set_title(f'{m_data[m]["label"]}: prior')
            ax1.set_title(f'{m_data[m]["label"]}: posterior')

            ax2.pcolormesh(lon,lat,
                            flux_diff,
                            cmap=cmap_diff,vmin=s_data[species]['difflim'][0],vmax=s_data[species]['difflim'][1],shading='nearest')

            ax2.set_title(f'{m_data[m]["label"]}: posterior - prior')

            if plot_site_locations == True:
                if sites_info[m] is not None:
                    for s in sites_info[m]:
                        ax0.scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='white',
                                    edgecolor='none',marker='o',s=30,zorder=2,alpha=0.5)
                        ax1.scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='white',
                                    edgecolor='none',marker='o',s=30,zorder=2,alpha=0.5)
                        ax2.scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='white',
                                    edgecolor='none',marker='o',s=30,zorder=2,alpha=0.5)
                        ax0.scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='none',
                                    edgecolor='black',marker='o',s=30,zorder=2)
                        ax1.scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='none',
                                    edgecolor='black',marker='o',s=30,zorder=2)
                        ax2.scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='none',
                                    edgecolor='black',marker='o',s=30,zorder=2)
                        
                
        except:
            print(f'ERROR: Either start and end dates are incorrect or there are missing data for model {m}.')
            print(f'Skipping plotting {m}.')
            
        if plot_point_markers is not None:

            if i == 0:
                print(f'\nPlotting markers for: {plot_point_markers}')
                print(f'Edit lines below line {inspect.getframeinfo(inspect.currentframe()).lineno} to change marker colour and size')
            for p in plot_point_markers:
                if type(p) == list:
                    ax0.scatter(p[0],p[1],color='black',marker='o',s=5,zorder=2)
                    ax1.scatter(p[0],p[1],color='black',marker='o',s=5,zorder=2)
                    ax2.scatter(p[0],p[1],color='black',marker='o',s=5,zorder=2)
                elif type(p) == str:
                    if p not in point_source_dict.keys():
                        print(f'{p} is not specified in point_source_dict, edit this to add a lat/lon location.')
                    else:
                        ax0.scatter(point_source_dict[p][0],point_source_dict[p][1],color='black',marker='o',s=5,zorder=2)
                        ax1.scatter(point_source_dict[p][0],point_source_dict[p][1],color='black',marker='o',s=5,zorder=2)
                        ax2.scatter(point_source_dict[p][0],point_source_dict[p][1],color='black',marker='o',s=5,zorder=2)
                        
    print('\nEdit flux_lim_kgkm2yr variable in species_info.json to adjust colourbar limits.'+
          'You will need to rerun the first cell of the notebook to apply the adjustment\n')
                        
    #flux colorbar
    levels = np.linspace(s_data[species]['fluxlim'][0],s_data[species]['fluxlim'][1])
    cbar = plt.cm.ScalarMappable(cmap=cmap)
    cbar.set_array(levels)
    cbar.set_clim(s_data[species]['fluxlim'])

    color_bar1 = fig.colorbar(cbar,orientation='vertical',cmap=cmap,extend='max',ax=ax[0,...],shrink=0.9,pad=0.005)
    color_bar1.set_label(f'Prior mean {s_data[species]["species_print"]}\n{time_out}\n(mol m$^{{-2}}$ s$^{{-1}}$)')

    color_bar2 = fig.colorbar(cbar,orientation='vertical',cmap=cmap,extend='max',ax=ax[1,...],shrink=0.9,pad=0.005)
    color_bar2.set_label(f'Posterior mean {s_data[species]["species_print"]}\n{time_out}\n(mol m$^{{-2}}$ s$^{{-1}}$)')

    #difference colorbar
    levels_diff = np.linspace(s_data[species]['difflim'][0],s_data[species]['difflim'][1])
    cbar_diff = plt.cm.ScalarMappable(cmap=cmap_diff)
    cbar_diff.set_array(levels_diff)
    cbar_diff.set_clim(s_data[species]['difflim'])

    color_bar3 = fig.colorbar(cbar_diff,orientation='vertical',extend='both',ax=ax[2,...],shrink=0.9,pad=0.005)
    color_bar3.set_label(f'Posterior - prior {s_data[species]["species_print"]}\n{time_out}\n(mol m$^{{-2}}$ s$^{{-1}}$)')
    
    return fig


def plot_spatial_flux_comparison(ds_all,species,plot_area,s_data,m_data,ppt_mode=False,
                                 cmap=None,cmap_diff=None,c_border=None,period_override=None,
                                 plot_site_locations=False,plot_point_markers=None,
                                plot_inversion_grid_flux=False):
    """
    Plots posterior fluxes and the difference between these
    for two models.
    Plots posterior and prior fluxes and the difference between these
    for all models.
    
    If ds_all contains more than two models, only the first two will
    be plotted.
    
    Args:
        ds_all (dictionary of datasets):
            xarray datasets of fluxes, scaled and sliced between 
            chosen dates.
        species (str): 
            Gas species, e.g. 'ch4'.
        plot_area (str):
            Lat/lon region to plot, options for 'UK', 'FRANCE', 'GERMANY',
            'NWEU','CWEU'.
        s_data (dict of dict):
            Dictionary of species with information for plotting (read from json file).
        m_data (dict of dict):
            Dictionary of inversion runs with filename and plot label (read from json file).
        ppt_mode (logical) (optional):
            If True, adjust label position to accomodate bigger fonts.
        cmap (str):
            Colour map for flux plots.
        cmap_diff (str):
            Colour map for difference plots.
        c_border (str):
            Colour for flux plot country borders.
        period_override (list of str, optional):
            Inversion periods to include, to override the standards in species_info.json.
            Must be the same length as models, e.g. ['monthly',None,'yearly']
        plot_site_locations (bool):
            If True, adds triangles with site locations to spatial plot.
        plot_point_markers (list of str or list of lat/lon):
            List of names of points to plot over larger point sources or lat/lon locations.
            See point_markers_dict for a list of options.
            e.g. ['paris','nw_england',[50.,5.]]
        plot_inversion_grid_flux (bool, default False):
            If True, plots fluxes at the spatial resolution of the inversion (using the 
            inversion_grid variable). If False, plots fluxes at the spatial resolution
            of the prior.
    Returns:
        fig (figure): 
            A plot of spatial flux posterior from two models a plot 
            of the absolute difference between these.
    """
    
    period_all = {}
    
    for i,m in enumerate(ds_all.keys()):
        m0 = m.split('_')[0]
        if period_override is not None:
            if 'monthly' in period_override[i]:
                period_all[m] = 'datetime64[M]'
            elif period_override[i] == 'yearly':
                period_all[m] = 'datetime64[Y]'
            else:
                period_all[m] = s_data[species]["dt_units"][m0]
        else:
            period_all[m] = s_data[species]["dt_units"][m0]
    
    if cmap == None:
        cmap = 'viridis' #'Blues'
    if cmap_diff == None:
        cmap_diff = 'coolwarm'
    if c_border == None:
        c_border = 'floralwhite'
    if cmap in ['green','blue','greyscale']:
        cmap = set_colormaps(cmap)
    
    n_cols = len(ds_all.keys())

    region_limits = {'UK':[-12,4,49,62],   #min_lon, max_lon, min_lat, max_lat
                    'FRANCE':[-6,9,42,52],
                    'GERMANY':[2,18,45,60],
                    'ITALY':[6,19,36,48],
                    'SWITZERLAND':[5.5,11,45,49],
                    'NETHERLANDS':[2.5,8,50,55],
                    'IRELAND':[-12,-4,51,56],
                    'HUNGARY':[15,24,44.5,50],
                    'NORWAY':[1,32,55,76],
                    'BENELUX':[1,9,48,55],
                    'NWEU':[-11,11,45,62],
                    'CWEU':[-12,27,37,66],
                    'EUROPE':[-98,40,10,80]}
    
    sites_info = {}
    if plot_site_locations == True:
        for i,m in enumerate(ds_all.keys()):
            try:
                sites_test = ds_all[m].sites.replace("'","").replace(']','').replace('[','').replace(' ','').split(',')
                sites_info[m] = extract_site_info(sites_test)
            except:
                sites_info[m] = None
                
        for i,m in enumerate(ds_all.keys()):
            if sites_info[m] == None:
                for j,m2 in enumerate(sites_info.keys()):
                    if sites_info[m2] != None:
                        print(f'No sites data available in {m} attrs, so using site data from {m2}')
                        sites_info[m] = sites_info[m2]
                    break

    fig,ax = plt.subplots(1,3,constrained_layout=True,figsize=(n_cols*5,9),
                   subplot_kw={'projection':cartopy.crs.PlateCarree()})

    for i in range(3):
        if i == 2:
            border_color = 'dimgrey'
        else:
            border_color = c_border
        ax[i].add_feature(cartopy.feature.BORDERS,edgecolor=border_color,linewidth=1.)
        ax[i].coastlines(resolution='50m',color=border_color,linewidth=1.)
        ax[i].set_extent(region_limits[plot_area])

    all_keys = []

    for i,m in enumerate(ds_all.keys()):
        
        lon = ds_all[m].longitude.values + (ds_all[m].longitude.values[1]-ds_all[m].longitude.values[1])/2
        lat = ds_all[m].latitude.values + (ds_all[m].latitude.values[1]-ds_all[m].latitude.values[1])/2
        
        all_keys.append(m)
        m0 = m.split('_')[0]
        
        if i == 0:
            if len(ds_all[m].time.values) == 1:
                time_out = to_datetime(ds_all[m].time.values[0].astype(period_all[m])).strftime('%d/%m/%Y')
            else:
                start_print = to_datetime(ds_all[m].time.values[0].astype(period_all[m])).strftime("%d/%m/%Y")
                if period_all[m] == 'datetime64[Y]':
                    end_period = ds_all[m].time.values[-1].astype(period_all[m]) + np.timedelta64(1,'Y') - np.timedelta64(1,'D')                    
                elif period_all[m] == 'datetime64[M]':
                    end_period = ds_all[m].time.values[-1].astype(period_all[m]) + np.timedelta64(1,'M') - np.timedelta64(1,'D')                    
                else:
                    print('This currently only works for monthly or yearly inversion periods. Update the plotting code to print out '+
                          'correct dates for higher frequency inversions.')
                end_print = to_datetime(end_period).strftime("%d/%m/%Y")
                time_out = (f'{start_print} - {end_print}')
        
            if plot_inversion_grid_flux == True:
                plot_original = False
                try:
                    ax[0].pcolormesh(lon,lat,
                                    np.mean(ds_all[m]['flux_total_posterior_inversion_grid'][:,:,:],axis=0),cmap=cmap,
                                    vmin=s_data[species]['fluxlim'][0],vmax=s_data[species]['fluxlim'][1],shading='nearest',
                                    )
                except:
                    print(f'Cannot find inversion_grid variables for {m} so using standard flux output.')
                    plot_original = True
            else:
                plot_original = True        
            
            if plot_original == True:
                ax[0].pcolormesh(lon,lat,
                                    np.mean(ds_all[m]['flux_total_posterior'][:,:,:],axis=0),cmap=cmap,
                                    vmin=s_data[species]['fluxlim'][0],vmax=s_data[species]['fluxlim'][1],shading='nearest',
                                    )

            ax[0].set_title(f'{m_data[m]["label"]}\nPosterior mean')
            
        elif i == 1:
            
            if plot_inversion_grid_flux == True:
                plot_original = False
                try:
                    ax[1].pcolormesh(lon,lat,
                                    np.mean(ds_all[m]['flux_total_posterior_inversion_grid'][:,:,:],axis=0),cmap=cmap,
                                    vmin=s_data[species]['fluxlim'][0],vmax=s_data[species]['fluxlim'][1],shading='nearest')
                except:
                    print(f'Cannot find inversion_grid variables for {m} so using standard flux output.')
                    plot_original = True
            else:
                plot_original = True
                    
            if plot_original == True:
                ax[1].pcolormesh(lon,lat,
                                    np.mean(ds_all[m]['flux_total_posterior'][:,:,:],axis=0),cmap=cmap,
                                    vmin=s_data[species]['fluxlim'][0],vmax=s_data[species]['fluxlim'][1],shading='nearest')  

            ax[1].set_title(f'{m_data[m]["label"]}\nPosterior mean')
            
        if plot_site_locations == True:
            if sites_info[m] is not None:
                for s in sites_info[m]:
                    ax[0].scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='white',
                                edgecolor='none',marker='o',s=30,zorder=2,alpha=0.5)
                    ax[1].scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='white',
                                edgecolor='none',marker='o',s=30,zorder=2,alpha=0.5)
                    ax[2].scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='white',
                                edgecolor='none',marker='o',s=30,zorder=2,alpha=0.5)
                    ax[0].scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='none',
                                edgecolor='black',marker='o',s=30,zorder=2)
                    ax[1].scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='none',
                                edgecolor='black',marker='o',s=30,zorder=2)
                    ax[2].scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='none',
                                edgecolor='black',marker='o',s=30,zorder=2)
        
    if plot_inversion_grid_flux == True:
        plot_original = False
        try:
            flux_diff = (np.mean(ds_all[all_keys[1]]['flux_total_posterior_inversion_grid'].values[:,:,:],axis=0)-
                        np.mean(ds_all[all_keys[0]]['flux_total_posterior_inversion_grid'].values[:,:,:],axis=0))
        except:
            plot_original = True
    else:
        plot_original = True
        
    if plot_original == True:
        flux_diff = (np.mean(ds_all[all_keys[1]]['flux_total_posterior'].values[:,:,:],axis=0)-
                        np.mean(ds_all[all_keys[0]]['flux_total_posterior'].values[:,:,:],axis=0))
        
    flux_diff[np.where(flux_diff) == np.nan] = 0.
    
    ax[2].pcolormesh(lon,lat,
                    flux_diff,
                    cmap=cmap_diff,vmin=s_data[species]['difflim'][0],vmax=s_data[species]['difflim'][1],shading='nearest')

    ax[2].set_title(f'{m_data[all_keys[1]]["label"]} - {m_data[all_keys[0]]["label"]}\nAbsolute difference')

    if plot_point_markers is not None:
        print(f'\nPlotting markers for: {plot_point_markers}')
        print(f'Edit lines below line {inspect.getframeinfo(inspect.currentframe()).lineno} to change marker colour')
        for p in plot_point_markers:
            if type(p) == list:
                ax[0].scatter(p[0],p[1],color='black',marker='o',s=5,zorder=2)
                ax[1].scatter(p[0],p[1],color='black',marker='o',s=5,zorder=2)
                ax[2].scatter(p[0],p[1],color='black',marker='o',s=5,zorder=2)
            elif type(p) == str:
                if p not in point_source_dict.keys():
                    print(f'{p} is not specified in point_source_dict, edit this to add a lat/lon location.')
                else:
                    ax[0].scatter(point_source_dict[p][0],point_source_dict[p][1],color='black',marker='o',s=5,zorder=2)
                    ax[1].scatter(point_source_dict[p][0],point_source_dict[p][1],color='black',marker='o',s=5,zorder=2)
                    ax[2].scatter(point_source_dict[p][0],point_source_dict[p][1],color='black',marker='o',s=5,zorder=2)
                        

    #flux colorbar
    levels = np.linspace(s_data[species]['fluxlim'][0],s_data[species]['fluxlim'][1])
    cbar = plt.cm.ScalarMappable(cmap=cmap)
    cbar.set_array(levels)
    cbar.set_clim(s_data[species]['fluxlim'])

    if (ppt_mode):
        labelpad_v = 20
    else:
        labelpad_v = 5

    color_bar2 = fig.colorbar(cbar,orientation='horizontal',cmap=cmap,extend='max',ax=ax[0],shrink=0.9,pad=0.01)
    color_bar2.set_label(f'{s_data[species]["species_print"]}\n{time_out}\n(mol m$^{{-2}}$ s$^{{-1}}$)', labelpad=labelpad_v)

    color_bar2 = fig.colorbar(cbar,orientation='horizontal',cmap=cmap,extend='max',ax=ax[1],shrink=0.9,pad=0.01)
    color_bar2.set_label(f'{s_data[species]["species_print"]}\n{time_out}\n(mol m$^{{-2}}$ s$^{{-1}}$)',labelpad=labelpad_v)

    #difference colorbar
    levels_diff = np.linspace(s_data[species]['difflim'][0],s_data[species]['difflim'][1])
    cbar_diff = plt.cm.ScalarMappable(cmap=cmap_diff)
    cbar_diff.set_array(levels_diff)
    cbar_diff.set_clim(s_data[species]['difflim'])

    color_bar3 = fig.colorbar(cbar_diff,orientation='horizontal',extend='both',ax=ax[2],shrink=0.9,pad=0.01)
    color_bar3.set_label(f'{s_data[species]["species_print"]}\n{time_out}\n(mol m$^{{-2}}$ s$^{{-1}}$)',labelpad=labelpad_v)
    
    return fig

def extract_country_sector_flux(models,ds_all_flux_scaled,species,resample,period_override,
                                s_data,resample_uncert_correlation,rolling_mean,
                                plot_resample_and_original,combine_to_one_timeseries,
                                models_priority,regions,sectors):
    """
    Runs functions to read in sector-level or total fluxes.
    Data is resampled/averaged etc. if needed, then returned for
    use by the rest of the plotting code or table-production code.
    """
    
    ### remove spatial flux variables to speed up later processing

    for m in models:
        ds_all_flux_scaled[m] = ds_all_flux_scaled[m].drop_vars(['flux_total_prior','percentile_flux_total_prior',
                                                                'flux_total_posterior','percentile_flux_total_posterior',
                                                                'flux_total_prior_out','percentile_flux_total_prior_out',
                                                                'flux_total_posterior_inversion_grid',
                                                                'percentile_flux_total_posterior_inversion_grid',
                                                                'country_fraction','outer_region_fraction'],errors='ignore')

    ### resample from monthly to yearly, if needed
    
    ds_all_p = resample_flux(ds_all_flux_scaled,species,resample,period_override,s_data,
                                    resample_uncert_correlation)
    
    rolling_mean_adjusted = []
    rolling_mean_count = 0
    
    for i,m in enumerate(models):
        
        rolling_mean_adjusted.append(rolling_mean[rolling_mean_count])
        
        if plot_resample_and_original == True:
            if resample[i] is not None:
                ds_all_p[f'{m}_original'] = ds_all_flux_scaled[m].copy()
                rolling_mean_adjusted.append(rolling_mean[rolling_mean_count])
        rolling_mean_count += 1
        
    ### calculate rolling means, if needed
            
    for i,m in enumerate(models):
        if rolling_mean_adjusted[i]:
            for var in list(ds_all_p[m].keys()):
                if var != 'countryname':
                    for c in range(ds_all_p[m]['countrynumber'].values.shape[0]):
                        if 'percentile' in var:
                            for p in range(2):
                                ds_all_p[m][var][:,p,c] = calc_rolling_mean(ds_all_p[m][var].values[:,p,c],rolling_mean_adjusted[i])
                        else:
                            ds_all_p[m][var][:,c] = calc_rolling_mean(ds_all_p[m][var].values[:,c],rolling_mean_adjusted[i])

       
    ### combine monthly and yearly model output into one timeseries, if needed
    
    ds_merged = {}
    
    if combine_to_one_timeseries == True:
        
        print('Merging results into one timeseries, with models used in priority order:')
        p_order = ''
        for p in models_priority:
            p_order += f'{models[p]}, '
        print(p_order)
        
        # set all timestamps to the start of the year, this is needed for combing monthly and yearly model output into one timeseries
        ds_merged_input = ds_all_p.copy()
        for i,m in enumerate(models):
            ds_merged_input[m]['time'] = ds_merged_input[m].time.values.astype('datetime64[Y]').astype('datetime64[ns]')

        if len(models) == 2:
            ds_merged[models[0]] = ds_merged_input[models[models_priority[0]]].combine_first(ds_merged_input[models[models_priority[1]]])
        elif len(models) == 1:
            ds_merged[models[0]] = ds_merged_input[models[0]].copy()
    
        # shift timestamps of averaged data forwards to centre of inversion period
        time_mid = np.array([]).astype('datetime64[ns]')
        time_diff_all = np.array([]).astype('timedelta64[ns]')
        
        for i,m in enumerate(list(ds_merged.keys())):
        
            for i,t in enumerate(ds_merged[m].time.values):
                if i < ds_merged[m].time.values.shape[0]-1:
                    time_diff = (ds_merged[m].time.values[i+1].astype('datetime64[ns]') - ds_merged[m].time.values[i].astype('datetime64[ns]'))/2
                    time_diff_all = np.hstack((time_diff_all,time_diff))
                    time_mid = np.hstack((time_mid,t+time_diff))
                else:
                    try:
                        av_diff = Counter(time_diff_all).most_common()[0][0]
                    except:
                        av_diff = np.mean(time_diff_all)
                    time_mid = np.hstack((time_mid,t+av_diff))
                    
            ds_merged[m]['time'] = time_mid
    
    else:
        ds_merged = ds_all_p.copy()
        
    ### extract region fluxes from merged, resampled and rolling-mean-ed datasets
    
    period_all = {}
    
    for m,model in enumerate(list(ds_merged.keys())):
         # Get inversion period
        if period_override is not None:
            if 'monthly' in period_override[m]:
                period_all[model] = 'monthly'
            elif period_override[m] == 'yearly':
                period_all[model] = 'yearly'
            else:
                period_all[model] = s_data[species]["period"]
        else:
            period_all[model] = s_data[species]["period"]

    region_time = {}
    region_flux = {}
    region_flux_prior = {}
    region_flux_lower = {}
    region_flux_upper = {}
    region_flux_prior_lower = {}
    region_flux_prior_upper = {}

    for m,model in enumerate(list(ds_merged.keys())):
        
        m0 = model.split('_')[0]
        
        region_time[model] = {}
        region_flux[model] = {}
        region_flux_prior[model] = {}
        region_flux_lower[model] = {}
        region_flux_upper[model] = {}
        region_flux_prior_lower[model] = {}
        region_flux_prior_upper[model] = {}
        
        for r,region in enumerate(regions):
            
            region_time[model][region] = {}
            region_flux[model][region] = {}
            region_flux_prior[model][region] = {}
            region_flux_lower[model][region] = {}
            region_flux_upper[model][region] = {}
            region_flux_prior_lower[model][region] = {}
            region_flux_prior_upper[model][region] = {}
            
            for sector in sectors:
            
                region_time[model][region][sector],region_flux[model][region][sector],region_flux_prior[model][region][sector],\
                region_flux_lower[model][region][sector],region_flux_upper[model][region][sector],\
                region_flux_prior_lower[model][region][sector],\
                region_flux_prior_upper[model][region][sector] = extract_region_flux(ds_merged,model,m0,region,sector=sector)
                        
                if ('region_time_years' in locals()) == False:
                    region_time_years = region_time[model][region][sector].astype('datetime64[Y]')
                else:
                    region_time_years = np.hstack((region_time_years,region_time[model][region][sector].astype('datetime64[Y]')))
    
    ### calculates monthly averages (e.g. average Jan results from all years)
    '''
    if seasonal_mean == True:
        
        region_time_month = {}
        region_flux_month = {}
        region_flux_lower_month = {}
        region_flux_upper_month = {}
        
        for m,model in enumerate(list(ds_merged.keys())):
            
            region_time_month[model] = {}
            region_flux_month[model] = {}
            region_flux_lower_month[model] = {}
            region_flux_upper_month[model] = {}
            
            for r,region in enumerate(regions):
                for j,month in enumerate(range(1,13,1)):
                    month_id = np.where(pd.DatetimeIndex(region_time[model][region]).month == month)
                    
                     #recalc posterior monthly uncertainty, assuming no correlation between all month e.g. all January fluxes
                    lower = region_flux[model][region][month_id] - region_flux_lower[model][region][month_id]   
                    upper = region_flux_upper[model][region][month_id] - region_flux[model][region][month_id]
                    lower_av = np.sqrt(np.sum(lower**2))/(lower.shape[0])
                    upper_av = np.sqrt(np.sum(upper**2))/(upper.shape[0])
                    lower_perc =  np.mean(region_flux[model][region][month_id]) - lower_av
                    upper_perc =  np.mean(region_flux[model][region][month_id]) + upper_av
                    
                    if j == 0:
                        region_time_month[model][region] = np.array([j])
                        region_flux_month[model][region] = np.mean(region_flux[model][region][month_id])
                        region_flux_lower_month[model][region] = lower_perc
                        region_flux_upper_month[model][region] = upper_perc
                    else:
                        region_time_month[model][region] = np.hstack((region_time_month[model][region],np.array([j])))
                        region_flux_month[model][region] = np.hstack((region_flux_month[model][region],np.mean(region_flux[model][region][month_id])))
                        region_flux_lower_month[model][region] = np.hstack((region_flux_lower_month[model][region],lower_perc))
                        region_flux_upper_month[model][region] = np.hstack((region_flux_upper_month[model][region],upper_perc))
                        
                region_time[model][region] = region_time_month[model][region].astype('timedelta64[M]') + np.datetime64('2018-01')
                region_flux[model][region] = region_flux_month[model][region]
                region_flux_lower[model][region] = region_flux_lower_month[model][region]
                region_flux_upper[model][region] = region_flux_upper_month[model][region]
    '''
    
    return (ds_all_p,ds_merged,period_all,
            region_time,region_flux,region_flux_lower,region_flux_upper,
            region_flux_prior_lower,region_flux_prior_upper)
        
#####################################################################
def plot_country_sector_flux(ds_all_flux_scaled,species,regions,
                      s_data,m_data,sectors,sector_colors,sector_labels,
                      start_date,end_date,
                      scale_co2eq=False,resample=None,
                      resample_uncert_correlation=False,
                      plot_resample_and_original=False,
                      period_override=None,fix_y_axes=False,
                      rolling_mean=[None],models_priority=None,
                      combine_to_one_timeseries=False,seasonal_mean=False):
    """
    Stacked bar chart of sector-level fluxes by timestamp.
    Currently only works for a single model.
    
    Args:
        ds_all_flux_scaled (dictionary of datasets):
            xarray datasets of fluxes, scaled and sliced between 
            chosen dates.
        species (str): 
            Gas species, e.g. 'ch4'.
        plot_regions (list of str):
            Country or regions to plot, e.g. ['UNITED KINGDOM','SWITZERLAND']
        s_data (dict of dict):
            Dictionary of species with information for plotting (read from json file).
        m_data (dict of dict):
            Dictionary of inversion runs with filename and plot label (read from json file).
        start_date (str) and end_date (str):
            Start and end dates of the data to plot.
            Used to slice inventory data.
        ppt_mode (logical) (optional):
            If True, adjust global legend position to accomodate bigger fonts.
        scale_co2eq (bool):
            If True, adapt y-axis label to CO2-eq.
        plot_inventory (bool):
            If True, plots inventory flux estimates as bars in each plot.
        inventory_years (list of str, optional):
            List of inventory data from different years to include. If None, only plots 
            the most recent inventory data.
        data_dir (str): 
            Path to top data directory, used to read inventory data files.
        fix_y_axes (bool):
            If True, uses a consistent y axis for all plots.
        resample (str):
            Option to be passed to resample built-in function of xarray Dataset. 
            For yearly average, 'YS' option should be used; 'QS-DEC' for seasonaly average.
            See http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
        resample_uncert_correlation (bool, default False):
            If True, calculates the resampled uncertainty as the mean from all averaged periods.
            If False, recalculates uncertainty assuming no correlation between all averaged periods,
            by taking the square root of the summed variances, divided by the number of averaging 
            periods.
        plot_resample_and_original (bool):
            If True, plots both the resampled data and the data as its original frequency.
            If False, only plots the resampled data.
        period_override (list of str, optional):
            Inversion periods to include, to override the standards in species_info.json.
            Must be the same length as models, e.g. ['monthly',None,'yearly']
        rolling_mean (int or list of int) (optional):
            If not None, calculates a rolling mean over this number of periods.
            e.g. if set to 3, will calculate the mean for each timestamp from values
            between timestamp-1 and timestamp+1.
    Returns:
        fig (figure): 
            A plot per country/region.
    """
    
    models = list(ds_all_flux_scaled.keys())
    
    if type(start_date) == list:
        start_date = str(min(start_date))
        end_date = str(max(end_date))
        
    if len(rolling_mean) != len(models):
        print('ERROR: rolling_mean must be the same length as models')
        return None

    if len(resample) != len(models):
        print('ERROR: resample must be the same length as models')
        return None
        
    if rolling_mean == None:
        rolling_mean = [None] * len(ds_all_flux_scaled.keys())
    print(f'\nApplying rolling means {dict(zip(ds_all_flux_scaled.keys(),rolling_mean))}.')
    
    print(f'\nApplying resampling: {dict(zip(ds_all_flux_scaled.keys(),resample))}.\n')

    ### read in all data, resample, apply rolling means and extract required regions
    
    ds_all_p,ds_merged,period_all,region_time,region_flux,region_flux_lower,region_flux_upper,\
    region_flux_prior_lower,\
    region_flux_prior_upper = extract_country_sector_flux(models,ds_all_flux_scaled,species,resample,period_override,
                                                    s_data,resample_uncert_correlation,rolling_mean,
                                                    plot_resample_and_original,combine_to_one_timeseries,
                                                    models_priority,regions,sectors)
    
    ### start creating plot

    if len(regions) == 4:
        n_cols = 2
        n_rows = 2
    elif len(regions) < 4:
        n_cols = len(regions)
        n_rows = 1
    elif len(regions) == 6:
        n_cols = 3
        n_rows = 2
    elif len(regions) > 4:
        n_cols = 4
        n_rows = math.ceil(len(regions)/4)
                
    fig = plt.figure(constrained_layout=True,figsize=(n_cols*6,n_rows*4))
    gs = fig.add_gridspec(n_rows,n_cols)
    
    # used to iterate through subplots
    count = 0
    total_s = 0

    for i,region in enumerate(regions):
        
        ax = fig.add_subplot(gs[count])
        
        for m,model in enumerate(list(ds_merged.keys())):
                    
            for s,sector in enumerate(sectors):
                
                xticks = np.arange(0,region_time[model][region][sector].shape[0],1)
                
                if s == 0:
                    ax.bar(xticks,region_flux[model][region][sector],
                           label=sector_labels[s],color=sector_colors[s],alpha=0.7,width=0.8)
                    total_s = region_flux[model][region][sector]
                    #ax.plot(region_time[model][region][sector],region_flux[model][region][sector],color=sector_colors[s])
                else:
                    ax.bar(xticks,region_flux[model][region][sector],
                           bottom=total_s,
                           label=sector_labels[s],color=sector_colors[s],alpha=0.7,width=0.8)
                    total_s =  total_s + region_flux[model][region][sector]
                
        #format each subplot
        if scale_co2eq:
            y_label_append = ' CO$_2$-eq'
            units_print = "T"
        else:
            y_label_append = ''
            units_print = s_data[species]["units_print"]
        
        ax.set_ylabel(f'{region} {s_data[species]["species_print"]} ({units_print}g y$^{{-1}}${y_label_append})')
        
        leg = ax.legend(ncol=1,borderpad=.4,columnspacing=1.0)
        for l in leg.legend_handles:
            l.set_linewidth(3.0)

        '''
        if seasonal_mean == True:
            ax.set_xticks((np.arange(np.min(min_x),np.max(max_x)+np.timedelta64(1,'M'),np.timedelta64(1,'M'))))
            ax.set_xticklabels(np.arange(1,13,1))    
            ax.set_xlabel('Month')   
        '''
        
        if any(period_all) == 'monthly' and any(resample) == 'year':
            
            xticklabels = (region_time[model][region][sector].astype('datetime64[Y]'))
            ax.set_xticks(xticks[::12])
            ax.set_xticklabels(xticklabels[::12],rotation=90)
        else:
            xticklabels = (region_time[model][region][sector].astype('datetime64[Y]'))
            ax.set_xticks(xticks[::2])
            ax.set_xticklabels(xticklabels[::2],rotation=90)
        
        #ax.set_xlim([np.datetime64(start_date)-np.timedelta64(200,'D'),
        #             np.datetime64(end_date)+np.timedelta64(200,'D')])
        
    # loop through plots again to fix min/max axis values
    
    for i,country in enumerate(regions):
        if fix_y_axes is not None:
            fig.axes[i].set_ylim(fix_y_axes)
    
    print('NOTE: If all the data is not within axis limits, adjust the set_ylim parameter')
    
    return fig

def plot_inventory_sector_flux(start_date,end_date,inv_sectors,inv_labels,inv_colors,species,
                               s_data,fix_y_axes=False,scale_co2eq=True):
    """
    """
    
    inv = pd.read_csv('/home/users/intem_ghg/UNFCCC_Inventory/NAEI/methane_emission_summary_10-04-2025.csv',
                  skiprows=1)   
    
    xticks = np.arange(int(start_date[:4]),int(end_date[:4]),1)
    
    empty_df = pd.DataFrame({'Sectors':inv['Sectors']})
    for d in xticks:
        empty_df[d] = [np.nan]*inv['Sectors'].shape[0]
        
    full_df = empty_df.copy()
        
    for d in xticks:
        if str(d) in inv.keys():
            full_df[d] = inv[str(d)]
    
    fig = plt.figure(constrained_layout=True,figsize=(6,4))
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[0])

    for s,sector in enumerate(inv_sectors):

        sector_id = [i for i,s in enumerate(full_df['Sectors'].values) if sector in s]
        
        sector_sum = full_df.iloc[sector_id].sum(axis=0,numeric_only=True).values        
        
        if scale_co2eq == True:
            gwp = s_data[species]["gwp"]
            if species == 'ch4':
                scale_factor = 1e3
            sector_sum = sector_sum / scale_factor * gwp
                        
        if s == 0:
            ax.bar(xticks,sector_sum,
                    label=inv_labels[s],color=inv_colors[s],alpha=0.7,width=0.8)
            total_s = sector_sum
            #ax.plot(region_time[model][region][sector],region_flux[model][region][sector],color=sector_colors[s])
        else:
            ax.bar(xticks,sector_sum,bottom=total_s,
                    label=inv_labels[s],color=inv_colors[s],alpha=0.7,width=0.8)
            total_s =  total_s + sector_sum
               
        #format each subplot
        if scale_co2eq:
            y_label_append = ' CO$_2$-eq'
            units_print = "T"
        else:
            y_label_append = ''
            units_print = s_data[species]["units_print"]
        
        if species == 'ch4':
            ax.set_ylabel(f'UK CH$_4$ ({units_print}g y$^{-1}$ {y_label_append})')
        
        leg = ax.legend(ncol=1,borderpad=.4,columnspacing=1.0)
        for l in leg.legend_handles:
            l.set_linewidth(3.0)

        '''
        if seasonal_mean == True:
            ax.set_xticks((np.arange(np.min(min_x),np.max(max_x)+np.timedelta64(1,'M'),np.timedelta64(1,'M'))))
            ax.set_xticklabels(np.arange(1,13,1))    
            ax.set_xlabel('Month')   
        '''
        
        ax.set_xticks(xticks[::2])
        ax.set_xticklabels(xticks[::2],rotation=90)
        
        #ax.set_xlim([np.datetime64(start_date)-np.timedelta64(200,'D'),
        #             np.datetime64(end_date)+np.timedelta64(200,'D')])
        
    # loop through plots again to fix min/max axis values
        if fix_y_axes is not None:
            ax.set_ylim(fix_y_axes)
    
    return fig
    