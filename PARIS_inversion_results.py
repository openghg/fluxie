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

model_q_indices = {'intem':[0,1],
                   'rhime':[0,1],
                   'elris':[0,1],
                   'flexinvert':[0,1]}

point_source_dict = {
                    'paris':[2.3404,48.8600],
                    'nw_england':[-2.7969,53.7748],
                    'london':[-0.1278, 51.5074],
                    'edinburgh':[-3.1883, 55.9533],
                    'cardiff':[-3.1791, 51.4816],
                    'belfast':[-5.9301, 54.5973],
                    'zurich': [8.5417, 47.3769],
                    'geneva': [6.1432, 46.2044],
                    'basel': [7.5886, 47.5596],
                    'lausanne': [6.6323, 46.5197],
                    'bern': [7.4474, 46.9481],
                    'berlin': [13.4050, 52.5200],
                    'hamburg': [9.9937, 53.5511],
                    'munich': [11.5820, 48.1351],
                    'koeln': [6.9603, 50.9375],
                    'frankfurt': [8.6821, 50.1109],
                    'essen': [7.0115, 51.4556],
                    'rome': [12.4964, 41.9028],
                    'milan': [9.1900, 45.4642],
                    'naples': [14.2681, 40.8518],
                    'turin': [7.6869, 45.0703],
                    'palermo': [13.3615, 38.1157],
                    'amsterdam': [4.9041, 52.3676],
                    'rotterdam': [4.4777, 51.9244],
                    'hague': [4.3007, 52.0705],
                    'utrecht': [5.1214, 52.0907],
                    'eindhoven': [5.4797, 51.4416],
                    'dublin': [-6.2603, 53.3498],
                    'cork': [-8.4727, 51.8985],
                    'limerick': [-8.6305, 52.6638],
                    'galway': [-9.0579, 53.2707],
                    'waterford': [-7.1119, 52.2593],
                    'budapest': [19.0402, 47.4979],
                    'debrecen': [21.6273, 47.5316],
                    'szeged': [20.1488, 46.2530],
                    'miskolc': [20.7852, 48.0993],
                    'pecs': [18.2324, 46.0784],
                    'oslo': [10.7522, 59.9139],
                    'bergen': [5.3241, 60.3929],
                    'sandnes': [58.8514, 58.8500],
                    'stavanger': [5.7382, 58.9690],
                    'drammen': [10.2045, 59.7438],
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
                     'NORWAY':'NOR'}

regions_dict = {'BELUX':'BEL-LUX',
                'BENELUX':'BEL-LUX-NLD',
                'CW_EU':'AUT-BEL-CHE-CZE-DEU-ESP-FRA-GBR-HRV-HUN-IRL-ITA-LUX-NLD-POL-PRT-SVK-SVN',
                'EU_GRP2':'AUT-BEL-CHE-DEU-DNK-FRA-GBR-IRL-ITA-LUX-NLD',
                'NW_EU':'BEL-DEU-DNK-FRA-GBR-IRL-LUX-NLD',
                'NW_EU2':'BEL-DEU-FRA-GBR-IRL-LUX-NLD',
                'NW_EU_CONTINENT':'BEL-DEU-FRA-LUX-NLD'}

regions_dict_old = {'CW_EU':'AUT-BEL-CHE-CZE-DEU-ESP-FRA-GBR-HRV-HUN-IRL-ITA-LUX-NLD-POL-PRT-SVK-SVK'}

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

    model_colors = {'intem':[['blue','dodgerblue'],
                             ['dodgerblue','skyblue']],
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
        plt.rc('font', size=11)
        plt.rc('axes', titlesize=11)
        plt.rc('axes', labelsize=10)
        plt.rc('xtick', labelsize=11)
        plt.rc('ytick', labelsize=11)
        plt.rc('legend', fontsize=10)

        annotate_coords = {0:[0.6,0.80],
                           1:[0.6,0.60],
                           2:[0.6,0.40]}

    return s_data,m_data,model_colors,annotate_coords

#####################################################################

def set_model_colors(models,model_colors):
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

#####################################################################

def slice_flux(ds_all,start_date,end_date,s_data,
               scale_units=True,scale_co2eq=False,convert_flux_units=False,species=None):
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
        convert_flux_units (bool): 
            If True, performs the conversion of molar flux to mass flux (default is False).
        species (str):
            Gas species, used to choose scaling units, e.g. 'ch4'.
    Returns:
        ds_all (dictionary of datasets):
            xarray datasets, scaled, converted, and sliced between chosen dates.
    
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
            if (scale_co2eq):
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
                    
        if convert_flux_units:
            M = s_data[species]["molar_mass"]
            ds_all[m] = convert_molar_to_mass_flux(ds_all[m], M)
        
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
                       'hfc227ea','hfc245fa','hfc32','hfc365mfc'] #,'hfc4310mee']
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
                    region_flux_total_prior_lower,region_flux_total_prior_upper = extract_region_flux(ds_in,model,m0,region,verbose=False)
                    
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
    
        for m in ds_all_original.keys():
            for v in ds_all_original[m].keys():
                if 'percentile_country' in v:
                    n_periods = ds_all_original[m][v].resample(time=rtime).count()[:,0,:]   #number of periods in each average
                    lower = (ds_all_original[m][v.replace('percentile_','')] - ds_all_original[m][v][:,0,:])    #recalculate upper and low standard deviations
                    upper = (ds_all_original[m][v][:,1,:] - ds_all_original[m][v.replace('percentile_','')])
                    lower_resampled = np.sqrt(((lower**2).resample(time=rtime).sum(dim="time")))/n_periods  #resample using sqrt of variances,divided by number of periods
                    upper_resampled = np.sqrt(((upper**2).resample(time=rtime).sum(dim="time")))/n_periods
                    lower_out = ds_all_p[m][v.replace('percentile_','')] - lower_resampled  #recalculated percentile upper and lower bounds
                    upper_out = ds_all_p[m][v.replace('percentile_','')] + upper_resampled
                    
                    ds_all_p[m][v] = xr.DataArray(np.concatenate((np.expand_dims(lower_out,axis=1),
                                                                    np.expand_dims(upper_out,axis=1)),axis=1),
                                                    dims=ds_all_p[m][v].dims)
        
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
            if 'flexinvert' in m:
                print('No scaling for flexinvert')
            else:
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

def stats_mf(ds_all):
    """
    Calculates the Pearson correlation coefficent and normalised root
    mean square error, of the fit between the posterior mean mf and the 
    observed mole fraction.
    
    Args:
        ds_all (dictionary of datasets):
            xarray datasets from slice_mf(), sliced between chosen dates
            but still containing all sites.
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
                                                            ds_all[m]['Yapost'].values[:,s][~np.isnan(ds_all[m]['Yobs'].values[:,s])])[0,1],3)
                    nrmse[site][m] = np.round(np.sqrt(np.mean((ds_all[m]['Yapost'].values[:,s][~np.isnan(ds_all[m]['Yobs'].values[:,s])]-
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
                                color='black',label=f'Obs ({m_data[m]["label"]})',s=8,alpha=0.8,
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

                ax2.annotate('$\mu$: '+str(var_mean)+'\n$\sigma$: '+str(var_sd),xy=annotate_coords[i],
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

                ax2.annotate('$\mu$: '+str(var_mean)+'\n$\sigma$: '+str(var_sd),xy=annotate_coords[i],
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

                ax2.annotate('$\mu$: '+str(var_mean)+'\n$\sigma$: '+str(var_sd),xy=annotate_coords[i],
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

def extract_region_flux(ds_all,m,m0,country,verbose=True):
    """
    Finds the index of a chosen region name and extracts the country flux
    variables for this region.
    Either extracts values directly from the dataset (if this region definition
    exists in the file) or calculates values by taking the sum of smaller regions
    (if this region definition does not exist in the file).
    """
    
    if m0 == 'intem':
        c_key = 'countrynumber'
    elif m0 == 'rhime':
        c_key = 'country'
    elif m0 == 'elris':
        c_key = 'country'
    elif m0 == 'flexinvert':
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
        region_flux_total_posterior = ds_all[m]['country_flux_total_posterior'].values[:,country_index]*r
        region_flux_total_prior = ds_all[m]['country_flux_total_prior'].values[:,country_index]*r
        
        if m0 == 'flexinvert':
            region_flux_total_posterior_lower = (ds_all[m]['country_flux_total_posterior'].values[:,country_index]
                                                 - ds_all[m]['country_flux_error_posterior'].values[:,country_index])*r
            region_flux_total_posterior_upper = (ds_all[m]['country_flux_total_posterior'].values[:,country_index]
                                                 + ds_all[m]['country_flux_error_posterior'].values[:,country_index])*r
            region_flux_total_prior_lower = (ds_all[m]['country_flux_total_prior'].values[:,country_index]
                                                 - ds_all[m]['country_flux_error_prior'].values[:,country_index])*r
            region_flux_total_prior_upper = (ds_all[m]['country_flux_total_prior'].values[:,country_index]
                                                 + ds_all[m]['country_flux_error_prior'].values[:,country_index])*r
        else:
            region_flux_total_posterior_lower = ds_all[m]['percentile_country_flux_total_posterior'].values[:,model_q_indices[m0][0],country_index]*r
            region_flux_total_posterior_upper = ds_all[m]['percentile_country_flux_total_posterior'].values[:,model_q_indices[m0][1],country_index]*r
            region_flux_total_prior_lower = ds_all[m]['percentile_country_flux_total_prior'].values[:,model_q_indices[m0][0],country_index]*r
            region_flux_total_prior_upper = ds_all[m]['percentile_country_flux_total_prior'].values[:,model_q_indices[m0][1],country_index]*r
        #print(region_time)
        #print(region_flux_total_posterior)
        
        region_flux_total_posterior_lower[region_flux_total_posterior_lower < 0.] = 0.
        region_flux_total_prior_lower[region_flux_total_prior_lower < 0.] = 0.

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
            elif m0 == 'flexinvert':
                c_key = 'country'

            country_index_vec = np.zeros(len(ds_all[m][c_key]))
            sigma2_region_flux_total_prior = 0
            region_flux_total_posterior = 0
            region_flux_total_prior = 0

            # Compute sum of prior/posterior emissions and prior uncertainty
            for var in country_list:
                try:
                    country_index = np.where(ds_all[m][c_key].values.astype(str) == var)[0][0]
                    country_index_vec[country_index] = 1

                    region_flux_total_posterior = region_flux_total_posterior + ds_all[m]['country_flux_total_posterior'].values[:,country_index]
                    region_flux_total_prior     = region_flux_total_prior + ds_all[m]['country_flux_total_prior'].values[:,country_index]

                    sigma_country_prior = ds_all[m]['country_flux_total_prior'].values[:,country_index] - ds_all[m]['percentile_country_flux_total_prior'].values[:,model_q_indices[m0][0],country_index]
                    sigma2_region_flux_total_prior = sigma2_region_flux_total_prior + sigma_country_prior**2

                except:
                    print(f'WARNING: {var} emissions are not present in {m}. This country will be neglected in {country} emissions.')
                    sigma2_region_flux_total_prior = np.zeros(ds_all[m].time.values.shape[0])
                    
            sigma_region_flux_total_prior = np.sqrt(sigma2_region_flux_total_prior)
        
            # Compute posterior uncertainty from covariance matrix
            try:
                sigma2 = np.zeros(np.shape(ds_all[m]['covariance_country_flux_total_posterior'])[0])

                for i in range(len(sigma2)):
                    sigma2[i] = country_index_vec.dot(ds_all[m]['covariance_country_flux_total_posterior'].values[i,:,:].dot(country_index_vec))

                sigma_region_flux_total_posterior = np.sqrt(sigma2)
            except:
                print(f'WARNING: Covariance matrix is not available for {m}. A posteriori uncertainty of {country} emissions will not be plotted.')
                sigma_region_flux_total_posterior = np.zeros(ds_all[m].time.values.shape[0])
                
            region_time = ds_all[m].time.values
            region_flux_total_posterior_lower = region_flux_total_posterior - sigma_region_flux_total_posterior
            region_flux_total_posterior_upper = region_flux_total_posterior + sigma_region_flux_total_posterior
            region_flux_total_prior_lower = region_flux_total_prior - sigma_region_flux_total_prior
            region_flux_total_prior_upper = region_flux_total_prior + sigma_region_flux_total_prior

            region_flux_total_posterior_lower[region_flux_total_posterior_lower < 0.] = 0.
            region_flux_total_prior_lower[region_flux_total_prior_lower < 0.] = 0.

        except:
            #print(f'ERROR: Could not find {country} emissions for {m}.')
            print(f'Skipping read in of {m}.')
            
            region_time = None
            region_flux_total_posterior,region_flux_total_prior = None,None
            region_flux_total_posterior_lower,region_flux_total_posterior_upper = None,None
            region_flux_total_prior_lower,region_flux_total_prior_upper = None,None
    
    return (region_time,region_flux_total_posterior,region_flux_total_prior,
            region_flux_total_posterior_lower,region_flux_total_posterior_upper,
            region_flux_total_prior_lower,region_flux_total_prior_upper)
    
#####################################################################

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
            region_search = regions_dict[country]
            country_list = region_search.split('-')

            inv_c_index = [0]*len(country_list)
            inv_c_value = np.zeros(len(inv_ds.time.values))

            print(f'No inventory data available for {country}. Considering sum of individual countries: {region_search}')

            for i,var in enumerate(country_list):
                try:
                    inv_key = [k for k, code in countrycodes_dict.items() if code == var]
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

#####################################################################

def plot_country_flux(ds_all,species,plot_regions,
                      s_data,m_data,model_colors,
                      start_date,end_date,ppt_mode=False,annex_mode=False,
                      scale_co2eq=False,
                      plot_inventory=True,inventory_years=None,
                      data_dir=None,fix_y_axes=False,fix_x_axes=None,
                      add_prior=True,add_prior_unc=False, set_global_leg=False,
                      country_codes_as_titles=None,plot_separate=True,
                      plot_combined=False,resample=None,
                      resample_uncert_correlation=False,
                      plot_resample_and_original=False,
                      period_override=None,
                      return_res=False,
                      rolling_mean=None
                     ):
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
        annex_mode (bool) (optional):
            If True, replace the labels with more concise versions for National Inventory Report Annexes.
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
        fix_x_axes (list, optional):
            Option to specify start and end years (e.g. fix_x_axes = ['2014','2024']).
        add_prior (bool):
            If True, plots prior as dashed lines.
        add_prior_unc (bool):
            If True, plots prior uncertainty as shaded area.
        set_global_leg (bool):
            If True, plots one single legend instead of one legend per subplot.
        country_codes_as_titles (bool)
            If True, uses list of country codes as titles, instead of the region names.
        plot_separate (list of bool or bool):
            If True, plots model result as separate line.
            List must be of same size as models, e.g. [True, False, False].
            If a single boolean is provided, the same flag is assumed for all models.
        plot_combined (list of bool or bool):
            If True, the model is included in combined average result to be plotted.
            List must be of same size as models, e.g. [False, True, True].
            If a single boolean is provided, the same flag is assumed for all models.
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
        return_res (bool, optional):
            Wheter or not including a dictionnary with the results as output
        rolling_mean (int, optional):
            If not None, calculate a rolling mean of the combined data using the input integer 
            to define the period.
    Returns:
        fig (figure): 
            A plot per country/region.
    """
    if return_res:
        res_dict = {country:dict() for country in plot_regions}
        print('WARNING : Only return the annual combined results for now, so work only if plot_combined=True')
    
    # Check input flags
    if type(plot_separate) == list:
        if len(plot_separate) != len(ds_all.keys()):
            print('ERROR: plot_separate must be a boolean or a list of booleans of the same length as models.')
            return None
    else:
        plot_separate = [plot_separate]*len(ds_all.keys())

    if type(plot_combined) == list:
        if len(plot_combined) != len(ds_all.keys()):
            print('ERROR: plot_combined must be a boolean or a list of booleans of the same length as models.')
            return None
    else:
        plot_combined = [plot_combined]*len(ds_all.keys())

    # Create annual mean xarrays if needed
    if resample is not None:

        # Check resample option
        if (resample == 'year'):
            rtime = 'YS'
        elif (resample == 'season'):
            rtime = 'QS-DEC'
        else:
            print(f'ERROR: Option resample=\'{resample}\' is not available. Try \'year\' or \'season\'.')
            return None

        ds_all_original = {m:ds_all[m].copy() for m in ds_all.keys()}
                
        if period_override is not None: 
            for m in ds_all.keys():
                if 'elris' in m:
                    del ds_all_original[m]['covariance_country_flux_total_posterior']
            ds_all_p = {m:ds_all_original[m].resample(time=rtime).mean(dim="time") if period_override[i] == 'monthly' else ds_all_original[m] for i,m in enumerate(ds_all.keys())}
            for i,m in enumerate(ds_all.keys()):
                if 'elris' in m and period_override[i] == 'monthly':
                    ds_all_p[m]['country'] = ds_all_p[m]['country'].isel(time=0).drop('time')
                    ds_all_p[m]['country_fraction'] = ds_all_p[m]['country_fraction'].isel(time=0).drop('time')
                    ds_all_p[m] = ds_all_p[m].assign({'covariance_country_flux_total_posterior':
                                                      ds_all[m]['covariance_country_flux_total_posterior'].resample(time=rtime).mean(dim="time")})
                    
        elif s_data[species]["period"]=='monthly':
            for m in ds_all.keys():
                if 'elris' in m:
                    del ds_all_original[m]['covariance_country_flux_total_posterior']
            ds_all_p = {m:ds_all_original[m].resample(time=rtime).mean(dim="time") for m in ds_all.keys()}
            for m in ds_all.keys():
                if 'elris' in m:
                    ds_all_p[m]['country'] = ds_all_p[m]['country'].isel(time=0).drop('time')
                    ds_all_p[m]['country_fraction'] = ds_all_p[m]['country_fraction'].isel(time=0).drop('time')
                    ds_all_p[m] = ds_all_p[m].assign({'covariance_country_flux_total_posterior':
                                                      ds_all[m]['covariance_country_flux_total_posterior'].resample(time=rtime).mean(dim="time")})
        
        else:
            ds_all_p = ds_all.copy()
            
        ds_all_p = calculate_resample_uncertainty(ds_all_original,ds_all_p,rtime,
                                                  resample_uncert_correlation)
            
        del ds_all_original
        
        # shift timestamps of averaged data forwards to centre of inversion period
        for m in ds_all.keys():
                
            time_mid = np.array([]).astype('datetime64[ns]')
            time_diff_all = np.array([]).astype('timedelta64[ns]')
            
            for i,t in enumerate(ds_all_p[m].time.values):
                if i < ds_all_p[m].time.values.shape[0]-1:
                    time_diff = (ds_all_p[m].time.values[i+1].astype('datetime64[ns]') - ds_all_p[m].time.values[i].astype('datetime64[ns]'))/2
                    time_diff_all = np.hstack((time_diff_all,time_diff))
                    time_mid = np.hstack((time_mid,t+time_diff))
                else:
                    try:
                        av_diff = collections.Counter(time_diff_all).most_common()[0][0]
                    except:
                        av_diff = np.mean(time_diff_all)
                    time_mid = np.hstack((time_mid,t+av_diff))
                    
            ds_all_p[m]['time'] = time_mid
            
    else:
        ds_all_p = ds_all.copy()

    # Initialize variables
    max_cf = np.zeros(len(plot_regions))
    start_year = [9999]*len(plot_regions)
    end_year = [0]*len(plot_regions)
    min_x = []
    max_x = []
    period_all = {}
    
    # Create figure
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
        
    fig = plt.figure(constrained_layout=True,figsize=(n_cols*6,n_rows*4))
    gs = fig.add_gridspec(n_rows,n_cols)
    
    # used to iterate through subplots
    count = 0

    for i,country in enumerate(plot_regions):
        
        ax = fig.add_subplot(gs[count])
        
        if plot_inventory == True:
            
            inv_colours = ['grey','black']
            
            if inventory_years == None:
                search_years = sorted(glob.glob(os.path.join(data_dir,'inventory',f'UNFCCC_inventory_{species}_*.nc')))
                inventory_years = [search_years[-1][-7:-3]]
            
            for y,i_year in enumerate(inventory_years):
            
                inventory_flux,inventory_time = extract_region_inventory_flux(country,data_dir,species,s_data,scale_co2eq,
                                                                              start_date,end_date,
                                                                              inventory_year=i_year)
                
                if inventory_flux is not None:
                    ax.bar(inventory_time,inventory_flux,
                                np.timedelta64(340, 'D'),color='white',edgecolor=inv_colours[y],align='edge',
                                label=f'Inventory {i_year}',zorder=0)
                    if return_res:
                        res_dict[country]['inventory']= {'time':inventory_time,
                                                         'value':inventory_flux}
        
        ds_count = 0
        
        if plot_resample_and_original == True:
            all_datasets = [ds_all_p,ds_all]
        else:
            all_datasets = [ds_all_p]
        
        for ds in all_datasets:
        
            post_pdfs = {}
            k = 0
            
            for j,m in enumerate(ds.keys()):
                
                m0 = m.split('_')[0]

                # Get inversion period
                if period_override is not None:
                    if period_override[j] == 'monthly':
                        period_all[m] = 'monthly'
                    elif period_override[j] == 'yearly':
                        period_all[m] = 'yearly'
                    else:
                        period_all[m] = s_data[species]["period"]
                else:
                    period_all[m] = s_data[species]["period"]
                    
                region_time,region_flux_total_posterior,region_flux_total_prior,\
                region_flux_total_posterior_lower,region_flux_total_posterior_upper,\
                region_flux_total_prior_lower,region_flux_total_prior_upper = extract_region_flux(ds,m,m0,country)
                
                if region_time is not None:
            
                    if plot_combined[j] == True:
                
                        if k == 0:
                            all_region_flux_total_posterior = region_flux_total_posterior
                            all_region_flux_total_prior = region_flux_total_prior
                            all_region_flux_total_lower = region_flux_total_posterior_lower
                            all_region_flux_total_upper = region_flux_total_posterior_upper
                            k+=1
                        else:
                            all_region_flux_total_posterior = np.vstack((all_region_flux_total_posterior,
                                                                        region_flux_total_posterior))
                            all_region_flux_total_prior = np.vstack((all_region_flux_total_prior,
                                                                    region_flux_total_prior))
                            all_region_flux_total_lower = np.vstack((all_region_flux_total_lower,
                                                                    region_flux_total_posterior_lower))
                            all_region_flux_total_upper = np.vstack((all_region_flux_total_upper,
                                                                    region_flux_total_posterior_upper))
                            
                        post_pdfs[m] = np.array([np.random.default_rng().normal(loc=region_flux_total_posterior[t],
                                                                                scale=np.mean(np.array([region_flux_total_posterior[t]-region_flux_total_posterior_lower[t],
                                                                                                        region_flux_total_posterior_upper[t]-region_flux_total_posterior[t]])),
                                                                                size=1000) for t in range(region_time.shape[0])])
                    else:
                        post_pdfs[m] = None
                                
                    if plot_separate[j] == True:
                        if ds_count == 0:
                            if annex_mode:
                                include_label = m_data[m]["label"].split()[0]
                                include_label_prior = f'{include_label} prior'
                                lw = 1
                                alpha = 0.7
                            else:
                                include_label = m_data[m]["label"]
                                include_label_prior = f'{m_data[m]["label"]} prior'
                                lw = 1.5
                                alpha = 1
                        else:
                            include_label = None
                            include_label_prior = None
                            
                        ax.plot(region_time,
                                    region_flux_total_posterior,
                                    label=include_label,color=model_colors[m][0])
                        
                        if not(plot_combined[j]):
                            ax.fill_between(region_time,
                                                region_flux_total_posterior_lower,
                                                region_flux_total_posterior_upper,
                                                alpha=0.3,color=model_colors[m][0])

                            if add_prior:
                                ax.plot(region_time,
                                            region_flux_total_prior,
                                            label=include_label_prior,color=model_colors[m][0],linestyle='dashed',linewidth=lw,alpha=alpha)
                                max_cf[i] = np.max((max_cf[i],np.nanmax(region_flux_total_prior)))

                                if add_prior_unc:
                                    ax.fill_between(region_time,
                                                        region_flux_total_prior_lower,
                                                        region_flux_total_prior_upper,
                                                        alpha=0.1,color=model_colors[m][0])
                                    max_cf[i] = np.max((max_cf[i],np.nanmax(region_flux_total_prior_upper)))

                    
                    min_x.append(np.min(region_time).astype('datetime64[M]'))
                    max_x.append(np.max(region_time).astype('datetime64[M]'))
                    max_cf[i] = np.max((max_cf[i],np.nanmax(region_flux_total_posterior_upper)))

                    if plot_inventory == True:
                        if inventory_flux is not None:
                            max_cf[i] = np.nanmax((max_cf[i],np.nanmax(inventory_flux[np.logical_and(inventory_time >= np.min(region_time),
                                                                                        inventory_time <= np.max(region_time))])))

                    y0 = ((region_time[0]).astype('datetime64[Y]')).astype(int)+1970
                    y1 = ((region_time[-1]).astype('datetime64[Y]')).astype(int)+1970
                    if (type(fix_x_axes) == list) == True:
                        start_year[i] = min(start_year[i],y0,int(fix_x_axes[0]))
                        end_year[i] = max(end_year[i],y1,int(fix_x_axes[1]))
                    else:
                        start_year[i] = min(start_year[i],y0)
                        end_year[i] = max(end_year[i],y1)

            if sum(plot_combined) != 0:
                
                #if i == 0:
                    #print('\nNOTE: This currently assumes that posterior PDFs are Gaussian. The average percentile is used '+
                    #    'to estimate an approximate standard deviation.\n')
                # Define labels
                if (annex_mode):
                    labels_combined = {'prior':'PARIS prior',
                                      'posterior':'PARIS mean',
                                      'unc':'_nolegend_'}
                else:
                    labels_combined = {'prior':'Mean prior',
                                      'posterior':'Mean posterior',
                                      'unc':'Min/max of post uncertainty'}
                
                if rolling_mean:
                    mean_country_flux_total_posterior = np.mean(calc_rolling_mean(all_region_flux_total_posterior),axis=0)
                    mean_country_flux_total_prior = np.mean(calc_rolling_mean(all_region_flux_total_prior),axis=0)
                    mean_country_flux_total_lower = np.mean(calc_rolling_mean(all_region_flux_total_lower),axis=0)
                    mean_country_flux_total_upper = np.mean(calc_rolling_mean(all_region_flux_total_upper),axis=0)
                    min_country_flux_total_lower = np.min(calc_rolling_mean(all_region_flux_total_lower),axis=0)
                    max_country_flux_total_upper = np.max(calc_rolling_mean(all_region_flux_total_upper),axis=0)
                else:
                    mean_country_flux_total_posterior = np.mean(all_region_flux_total_posterior,axis=0)
                    mean_country_flux_total_prior = np.mean(all_region_flux_total_prior,axis=0)
                    mean_country_flux_total_lower = np.mean(all_region_flux_total_lower,axis=0)
                    mean_country_flux_total_upper = np.mean(all_region_flux_total_upper,axis=0)
                    min_country_flux_total_lower = np.min(all_region_flux_total_lower,axis=0)
                    max_country_flux_total_upper = np.max(all_region_flux_total_upper,axis=0)
                
                #print(return_res,region_time.astype('datetime64[Y]'))
                if return_res :#and ((region_time.astype('datetime64[Y]')[1:]-region_time.astype('datetime64[Y]')[:-1]).astype(int)>=1).all():
                    res_dict[country]['combined']= {'time':region_time.astype('datetime64[ns]'),
                                                    'mean':mean_country_flux_total_posterior,
                                                    'min':min_country_flux_total_lower,
                                                    'max':max_country_flux_total_upper}
                '''
                # NOTE: This section of code is not prepared for type(plot_combined) == list
                #       When plot_combined[j] == False, post_pdfs[m] = None
                #       Plese revise the implementation before uncommenting.

                for j,m in enumerate(ds.keys()):
                    if j == 0:
                        pdf_all = np.array([np.random.choice(post_pdfs[m][t,:],500) for t in range(post_pdfs[m].shape[0])])
                    else:
                        pdf_all = np.hstack((pdf_all,
                                            np.array([np.random.choice(post_pdfs[m][t,:],500) for t in range(post_pdfs[m].shape[0])])))
                
                pdf_mean = np.mean(pdf_all,axis=1)
                pdf_std = np.std(pdf_all,axis=1)
                '''     
                
                ax.plot(region_time.astype('datetime64[ns]'),
                        mean_country_flux_total_posterior,
                        label=labels_combined['posterior'],color='black',linewidth=3.5)
                
                if add_prior:
                    ax.plot(region_time.astype('datetime64[ns]'),
                            mean_country_flux_total_prior,
                            label=labels_combined['prior'],color='black',linestyle='dashed',linewidth=lw,alpha=alpha)
                
                ax.fill_between(region_time.astype('datetime64[ns]'),
                                min_country_flux_total_lower,
                                max_country_flux_total_upper,
                                alpha=0.3,color='grey',label=labels_combined['unc'])
                '''
                ax.plot(region_time.astype('datetime64[ns]'),
                                    pdf_mean,
                                    label='Mean of sampled post PDFs',color='dodgerblue')
                
                ax.fill_between(region_time.astype('datetime64[ns]'),
                                                pdf_mean-pdf_std,
                                                pdf_mean+pdf_std,
                                                alpha=0.3,color='dodgerblue',label='Std dev of sampled post PDFs')
                
                ax.fill_between(region_time.astype('datetime64[ns]'),
                                                mean_country_flux_total_lower,
                                                mean_country_flux_total_upper,
                                                alpha=0.3,color='yellow',label='Mean of post uncertainty')
                '''                            
            
            ds_count += 1
            
        #format each subplot
        units_print = s_data[species]["units_print"]
        if 'all' in species:
            y_label_append = ' CO$_2$-eq'
        elif scale_co2eq:
            y_label_append = ' CO$_2$-eq'
            units_print = "T"
        else:
            y_label_append = ''
        
        ax.set_ylabel(f'{s_data[species]["species_print"]} ({units_print}g{y_label_append} yr$^{{-1}}$)')
        
        if (type(fix_x_axes) == list) == True:
            ax.set_xlim([np.datetime64(fix_x_axes[0])-np.timedelta64(1,'M'),
                            np.datetime64(fix_x_axes[1])+np.timedelta64(1,'M')])
        elif period_all[list(ds.keys())[0]] == 'monthly' and resample != 'year':
            ax.set_xlim([np.min(min_x)-np.timedelta64(1,'M'),
                            np.max(max_x)+np.timedelta64(1,'M')])
        else: #period_all[list(ds.keys())[0]] == 'yearly':
            ax.set_xlim([np.min(min_x)-np.timedelta64(7,'M'),
                            np.max(max_x)+np.timedelta64(7,'M')])
        
        ncol = 2
        if annex_mode: ncol = 3
        if set_global_leg == False:
            leg = ax.legend(ncol=ncol,borderpad=.4,columnspacing=1.0)
            if plot_inventory == True:
                for l in leg.legendHandles[:-1]:
                    l.set_linewidth(3.0)
            else:
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
        
        if country_codes_as_titles != None:
            if country_codes_as_titles == True:
                try:
                    ax.set_title(f'{print_country}\n{regions_dict[country]}')
                except:
                    ax.set_title(f'{print_country}')
            else:
                ax.set_title(f'{print_country}')
            
        ax.grid(visible=True,which='major',alpha=0.4)

        if (end_year[i]-start_year[i]) > 8:
            years_list = list(range(start_year[i],end_year[i]+2))
            region_time = np.array([np.datetime64(str(year), 'Y') for year in years_list])
            ax.set_xticks(region_time[::2])
            ax.set_xticklabels(region_time[::2].astype('datetime64[Y]'))
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
                for l in leg.legendHandles:
                    l.set_linewidth(3.0)
            else:
                for l in leg.legendHandles:
                    l.set_linewidth(3.0)

    # loop through plots again to fix min/max axis values
    
    fac = 1.1
    if not(set_global_leg): fac = 1.2
    for i,country in enumerate(plot_regions):
        if fix_y_axes == True:
            fig.axes[i].set_ylim([0,np.nanmax(max_cf)*fac])  
        elif (type(fix_y_axes) == list) == True:
            fig.axes[i].set_ylim(fix_y_axes)
        elif fix_y_axes == False:
            fig.axes[i].set_ylim([0,max_cf[i]*fac])
    
    print('NOTE: If all the data is not within axis limits, adjust the set_ylim parameter')
    
    if return_res:
        return fig,res_dict
    else:
        return fig

#####################################################################

def plot_spatial_flux(ds_all,species,plot_area,s_data,m_data,cmap=None,
                      cmap_diff=None,c_border=None,period_override=None,
                      plot_site_locations=False,plot_point_markers=None,
                      season=None,set_fluxlim='default',set_fluxlim_percentile=None,
                      plot_inversion_grid_flux=False):
    """
    Plots posterior and prior fluxes and the difference between these
    for all models.
    
    If ds_all contains multiple time periods for each model, the average 
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
        set_fluxlim (str or list/tuple, optional): 
            If provided, set the colorbar limits based on the selected options.
            Options are 'default', 'auto', a list or tuple with two elements (min, max).
            The default option is 'default', which is based on species_info values.
        set_fluxlim_percentile (float, optional):
            If provided, set the percentile to use when setting the colorbar limits with 'auto' option.
        plot_inversion_grid_flux (bool, default False):
            If True, plots fluxes at the spatial resolution of the inversion (using the 
            inversion_grid variable). If False, plots fluxes at the spatial resolution
            of the prior.
            
    Returns:
        fig (figure): 
            A plot of spatial flux posterior and prior mean/mode and a plot 
            of the absolute difference between these, for each model.
    """
    
    period_all = {}
    
    for i,m in enumerate(ds_all.keys()):
        m0 = m.split('_')[0]
        if period_override is not None:
            if period_override[i] == 'monthly':
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
    
    n_cols = len(ds_all.keys())

    region_limits = {'UK':[-12,4,49,62],   #min_lon, max_lon, min_lat, max_lat
                    'FRANCE':[-6,9,42,52],
                    'GERMANY':[2,18,45,60],
                    'ITALY':[6,19,36,48],
                    'SWITZERLAND':[5.5,11,45,49],
                    'NETHERLANDS':[2.5,8,50,55],
                    'IRELAND':[-12,-4,51,56],
                    'HUNGARY':[15.5,23.5,44.5,50],
                    'NORWAY':[1,32,52,76],
                    'BENELUX':[1,9,48,55],
                    'NWEU':[-11,11,45,62],
                    'CWEU':[-12,27,37,66],
                    'EUROPE':[-98,40,10,80]}
    
    # find site info in netcdf attrs. if none present, use site info from first model with this 
    # data available
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
                    
    # Determine flux limits based on 'flux_total_posterior'
    fluxlim = set_flux_limits(ds_all, 'flux_total_posterior', region_limits[plot_area], s_data[species], option=set_fluxlim, custom_percentile=set_fluxlim_percentile)
    difflim = tuple([-fluxlim[1],fluxlim[1]])
        
    # Find units info in netcdf attrs
    first_key = list(ds_all.keys())[0]
    flux_units = ds_all[first_key]['flux_total_posterior'].attrs.get('units')
    flux_units = flux_units.replace("-2", "$^{-2}$").replace("-1", "$^{-1}$")    

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
                            cmap=cmap_diff,vmin=difflim[0],vmax=difflim[1],shading='nearest')

            ax2.set_title(f'{m_data[m]["label"]}: posterior - prior')

            if plot_site_locations == True:
                if sites_info[m] is not None:
                    for s in sites_info[m]:
                        ax0.scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],facecolor='none',
                                    edgecolor='red',marker='o',s=30,zorder=2)
                        ax1.scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],facecolor='none',
                                    edgecolor='red',marker='o',s=30,zorder=2)
                        ax2.scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],facecolor='none',
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
                    ax0.scatter(p[0],p[1],facecolors='none',edgecolors='red',marker='^',s=30,zorder=2)
                    ax1.scatter(p[0],p[1],facecolors='none',edgecolors='red',marker='^',s=30,zorder=2)
                    ax2.scatter(p[0],p[1],facecolors='none',edgecolors='black',marker='^',s=30,zorder=2)
                elif type(p) == str:
                    if p not in point_source_dict.keys():
                        print(f'{p} is not specified in point_source_dict, edit this to add a lat/lon location.')
                    else:
                        ax0.scatter(point_source_dict[p][0],point_source_dict[p][1], facecolors='none', 
                                    edgecolors='red',marker='^',s=30,zorder=2)
                        ax1.scatter(point_source_dict[p][0],point_source_dict[p][1], facecolors='none', 
                                    edgecolors='red',marker='^',s=30,zorder=2)
                        ax2.scatter(point_source_dict[p][0],point_source_dict[p][1], facecolors='none', 
                                    edgecolors='black',marker='^',s=30,zorder=2)
                        
    #flux colorbar
    levels = np.linspace(fluxlim[0],fluxlim[1])
    cbar = plt.cm.ScalarMappable(cmap=cmap)
    cbar.set_array(levels)
    cbar.set_clim(fluxlim)

    color_bar1 = fig.colorbar(cbar,orientation='vertical',cmap=cmap,extend='max',ax=ax[0,...],shrink=0.9,pad=0.005)
    color_bar1.set_label(f'Prior mean {s_data[species]["species_print"]}\n{time_out}\n({flux_units})')

    color_bar2 = fig.colorbar(cbar,orientation='vertical',cmap=cmap,extend='max',ax=ax[1,...],shrink=0.9,pad=0.005)
    color_bar2.set_label(f'Posterior mean {s_data[species]["species_print"]}\n{time_out}\n({flux_units})')

    #difference colorbar
    levels_diff = np.linspace(difflim[0],difflim[1])
    cbar_diff = plt.cm.ScalarMappable(cmap=cmap_diff)
    cbar_diff.set_array(levels_diff)
    cbar_diff.set_clim(difflim)

    color_bar3 = fig.colorbar(cbar_diff,orientation='vertical',extend='both',ax=ax[2,...],shrink=0.9,pad=0.005)
    color_bar3.set_label(f'Posterior - prior {s_data[species]["species_print"]}\n{time_out}\n({flux_units})')
    
    return fig

#####################################################################

def plot_spatial_flux_comparison(ds_all,species,plot_area,s_data,m_data,ppt_mode=False,
                                 cmap=None,cmap_diff=None,c_border=None,period_override=None,
                                 plot_site_locations=False,plot_point_markers=None,set_fluxlim='default',
                                 set_fluxlim_percentile=None,plot_inversion_grid_flux=False):
    """
    Plots posterior fluxes and the difference between these
    for two models.
    
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
        set_fluxlim (str or list/tuple, optional): 
            If provided, set the colorbar limits based on the selected options.
            Options are 'default', 'auto', a list or tuple with two elements (min, max).
            The default option is 'default', which is based on species_info values.
        set_fluxlim_percentile (float, optional):
            If provided, set the percentile to use when setting the colorbar limits with 'auto' option.
    Returns:
        fig (figure): 
            A plot of spatial flux posterior from two models a plot 
            of the absolute difference between these.
    """
    
    period_all = {}
    
    for i,m in enumerate(ds_all.keys()):
        m0 = m.split('_')[0]
        if period_override is not None:
            if period_override[i] == 'monthly':
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
    
    n_cols = len(ds_all.keys())

    region_limits = {'UK':[-12,4,49,62],   #min_lon, max_lon, min_lat, max_lat
                    'FRANCE':[-6,9,42,52],
                    'GERMANY':[2,18,45,60],
                    'ITALY':[6,19,36,48],
                    'SWITZERLAND':[5.5,11,45,49],
                    'NETHERLANDS':[2.5,8,50,55],
                    'IRELAND':[-12,-4,51,56],
                    'HUNGARY':[15.5,23.5,44.5,50],
                    'NORWAY':[1,32,52,76],
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
                    
    # Determine flux limits based on 'flux_total_posterior'
    fluxlim = set_flux_limits(ds_all, 'flux_total_posterior', region_limits[plot_area], s_data[species], option=set_fluxlim, custom_percentile=set_fluxlim_percentile)
    difflim = tuple([-fluxlim[1],fluxlim[1]])
        
    # Find units info in netcdf attrs
    first_key = list(ds_all.keys())[0]
    flux_units = ds_all[first_key]['flux_total_posterior'].attrs.get('units')
    flux_units = flux_units.replace("-2", "$^{-2}$").replace("-1", "$^{-1}$")

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
                    ax[0].scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],facecolor='none',
                                edgecolor='red',marker='o',s=30,zorder=2)
                    ax[1].scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],facecolor='none',
                                edgecolor='red',marker='o',s=30,zorder=2)
                    ax[2].scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],facecolor='none',
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
                    cmap=cmap_diff,vmin=difflim[0],vmax=difflim[1],shading='nearest')

    ax[2].set_title(f'{m_data[all_keys[1]]["label"]} - {m_data[all_keys[0]]["label"]}\nAbsolute difference')

    if plot_point_markers is not None:
        print(f'\nPlotting markers for: {plot_point_markers}')
        print(f'Edit lines below line {inspect.getframeinfo(inspect.currentframe()).lineno} to change marker colour')
        for p in plot_point_markers:
            if type(p) == list:
                ax[0].scatter(p[0],p[1],facecolor='none',edgecolor='red',marker='^',s=30,zorder=2)
                ax[1].scatter(p[0],p[1],facecolor='none',edgecolor='red',marker='^',s=30,zorder=2)
                ax[2].scatter(p[0],p[1],facecolor='none',edgecolor='black',marker='^',s=30,zorder=2)
            elif type(p) == str:
                if p not in point_source_dict.keys():
                    print(f'{p} is not specified in point_source_dict, edit this to add a lat/lon location.')
                else:
                    ax[0].scatter(point_source_dict[p][0],point_source_dict[p][1],
                                  facecolor='none',edgecolor='red',marker='^',s=30,zorder=2)
                    ax[1].scatter(point_source_dict[p][0],point_source_dict[p][1],
                                  facecolor='none',edgecolor='red',marker='^',s=30,zorder=2)
                    ax[2].scatter(point_source_dict[p][0],point_source_dict[p][1],
                                  facecolor='none',edgecolor='black',marker='^',s=30,zorder=2)
                        

    #flux colorbar
    levels = np.linspace(fluxlim[0],fluxlim[1])
    cbar = plt.cm.ScalarMappable(cmap=cmap)
    cbar.set_array(levels)
    cbar.set_clim(fluxlim)

    if (ppt_mode):
        labelpad_v = 20
    else:
        labelpad_v = 5

    color_bar2 = fig.colorbar(cbar,orientation='horizontal',cmap=cmap,extend='max',ax=ax[0],shrink=0.9,pad=0.01)
    color_bar2.set_label(f'{s_data[species]["species_print"]}\n{time_out}\n({flux_units})', labelpad=labelpad_v)

    color_bar2 = fig.colorbar(cbar,orientation='horizontal',cmap=cmap,extend='max',ax=ax[1],shrink=0.9,pad=0.01)
    color_bar2.set_label(f'{s_data[species]["species_print"]}\n{time_out}\n({flux_units})',labelpad=labelpad_v)

    #difference colorbar
    levels_diff = np.linspace(difflim[0],difflim[1])
    cbar_diff = plt.cm.ScalarMappable(cmap=cmap_diff)
    cbar_diff.set_array(levels_diff)
    cbar_diff.set_clim(difflim)

    color_bar3 = fig.colorbar(cbar_diff,orientation='horizontal',extend='both',ax=ax[2],shrink=0.9,pad=0.01)
    color_bar3.set_label(f'{s_data[species]["species_print"]}\n{time_out}\n({flux_units})',labelpad=labelpad_v)
    
    return fig

#####################################################################

def plot_spatial_flux_per_timestamp(ds_all,species,plot_area,end_date,s_data,m_data,
                                    cmap='viridis',c_border='floralwhite',
                                    var='flux_total_posterior', plot_combined=False, annex_mode=False,
                                    chop_by='year',dt=1,period_override=None,
                                    plot_site_locations=False,plot_point_markers=False,set_fluxlim='default',
                                    set_fluxlim_percentile=None,plot_inversion_grid_flux=False):
    """
    Plots posterior fluxes, prior fluxes or difference between these
    for all models and specific time intervals.

    Args:
        ds_all (dictionary of datasets):
            xarray datasets of fluxes, scaled and sliced between
            chosen dates.
        species (str):
            Gas species, e.g. 'ch4'.
        plot_area (str):
            Lat/lon region to plot, options for 'UK', 'FRANCE', 'GERMANY',
            'ITALY','SWITZERLAND','NWEU','CWEU','EUROPE'.
        end_date (str):
            End date of sliced data, e.g. '2022-01-01' would include all
            data up to 2021-12-31.
        s_data (dict of dict):
            Dictionary of species with information for plotting (read from json file).
        m_data (dict of dict):
            Dictionary of inversion runs with filename and plot label (read from json file).
        cmap (str):
            Colour map for flux plots.
        c_border (str):
            Colour for flux plot country borders.
        var (str):
            Variable to be plotted; options for 'flux_total_prior',
            'flux_total_posterior', 'posterior_prior_diff'
        plot_combined (bool):
            If True, plots the mean over all models at each time step.
        annex_mode (bool) (optional):
            If True, replace the labels with more concise versions for National Inventory Report Annexes.
        chop_by (str or list):
            Time units to perform the average, options for 'year', 'month' and 'season'.
            Option 'season' will perform the average over specific months/seasons (e.g. Jan-Jun, Jul-Dec).
            Alternatively, a list of starting dates can be provided. The respective
            end dates are assumed equal to the start date of the following averaging period.
        dt (int or list of lists):
            if chop_by = 'year' or 'month': dt is the number of time steps (in chop_by units) to use in the averaging;
            if chop_by = 'season': dt is a list where each element is a list of months to consider in the averaging (e.g. dt=[[1,2],[10,11]], will average over Jan-Feb and Oct-Nov);
            if chop_by is a list, dt is set to None
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
        set_fluxlim (str or list/tuple, optional): 
            If provided, set the colorbar limits based on the selected options.
            Options are 'default', 'auto', a list or tuple with two elements (min, max).
            The default option is 'default', which is based on species_info values.
        set_fluxlim_percentile (float, optional):
            If provided, set the percentile to use when setting the colorbar limits with 'auto' option.
    Returns:
        fig (figure):
            A plot of spatial flux of the variable specified in var
            averaged over the number of time steps specified in dt.
    """
    dt_units_all = {}
    period_all = {}
    
    for i,m in enumerate(ds_all.keys()):
        m0 = m.split('_')[0]
        if period_override is not None:
            if period_override[i] == 'monthly':
                period_all[m] = 'monthly'
                dt_units_all[m] = 'datetime64[M]'
            elif period_override[i] == 'yearly':
                period_all[m] = 'yearly'
                dt_units_all[m] = 'datetime64[Y]'
            else:
                period_all[m] = s_data[species]["period"]
                dt_units_all[m] = s_data[species]["dt_units"][m0]
        else:
            period_all[m] = s_data[species]["period"]
            dt_units_all[m] = s_data[species]["dt_units"][m0]

    if (annex_mode) and (plot_combined):
        var_labels = {'flux_total_prior':'PARIS prior',
                      'flux_total_posterior':'PARIS mean',
                      'posterior_prior_diff':'Posterior-prior',
                      'posterior_mean_diff':'Deviation'}
    else:
        var_labels = {'flux_total_prior':'Prior mean',
                      'flux_total_posterior':'Posterior mean',
                      'posterior_prior_diff':'Posterior-prior',
                      'posterior_mean_diff':'Posterior-mean'}
        
    region_limits = {'UK':[-12,4,49,62],   #min_lon, max_lon, min_lat, max_lat
                    'FRANCE':[-6,9,42,52],
                    'GERMANY':[2,18,45,60],
                    'ITALY':[6,19,36,48],
                    'SWITZERLAND':[5.5,11,45,49],
                    'NETHERLANDS':[2.5,8,50,55],
                    'IRELAND':[-12,-4,51,56],
                    'HUNGARY':[15.5,23.5,44.5,50],
                    'NORWAY':[1,32,52,76],
                    'BENELUX':[1,9,48,55],
                    'NWEU':[-11,11,45,62],
                    'CWEU':[-12,27,37,66],
                    'EUROPE':[-98,40,10,80]}

    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
    # Determine flux limits and define variable extend setting
    lim = set_flux_limits(ds_all, var, region_limits[plot_area], s_data[species], option=set_fluxlim, custom_percentile=set_fluxlim_percentile)
    if var in ['posterior_prior_diff','posterior_mean_diff']:
        extend ='both'
    else:
        extend = 'max'
        
    # Find units info in netcdf attrs
    first_key = list(ds_all.keys())[0]
    flux_units = ds_all[first_key]['flux_total_posterior'].attrs.get('units')
    flux_units = flux_units.replace("-2", "$^{-2}$").replace("-1", "$^{-1}$")

    # Figure size and averaging period
    n_lines = len(ds_all.keys())
    if plot_combined: n_lines = 1
    t0_date = {}
    t1_date = {}
    start_print = {}
    end_print = {}
    indexes = {}

    if type(chop_by) == list:
        n_cols = len(chop_by)
        dt = None
        for m in ds_all.keys():
            # Get dates of start/end time stamps
            t0_date[m] = chop_by
            t1_date[m] = chop_by[1:] + [end_date]

            # Get start/end time stamps for caption
            if annex_mode:
                fmt = '%Y'
            else:
                fmt = '%d/%m/%Y'

            start_print[m] = to_datetime(t0_date[m]).strftime(fmt)
            end_print[m]   = (to_datetime(t1_date[m]) - np.timedelta64(1,'D')).strftime(fmt)

    else:
        # NOTE: It will only work properly if the data is complete between start_date and end_date
        nt = np.zeros(len(ds_all.keys()))
        for i,m in enumerate(ds_all.keys()):
            total_times = len(ds_all[m].time)

            if ((period_all[m]=='yearly') and (chop_by=='year')) or ((period_all[m]=='monthly') and (chop_by=='month')):
                nt[i] = total_times//dt
                # Get indexes of start/end time stamps
                t0 = [k for k in range(0,total_times,dt)]
                t1 = [k for k in range(dt,total_times,dt)]

                # Get dates of start/end time stamps
                t0_date[m] = [to_datetime(ds_all[m].time.values[tt].astype(s_data[species]["dt_units"][m0])).strftime('%Y-%m-%d') for tt in t0]
                t1_date[m] = [to_datetime(ds_all[m].time.values[tt].astype(s_data[species]["dt_units"][m0])).strftime('%Y-%m-%d') for tt in t1]
                t1_date[m] = t1_date[m] + [end_date]

                # Get start/end time stamps for caption
                if period_all[m]=='yearly':
                    start_print[m] = [to_datetime(ds_all[m].time.values[tt].astype(s_data[species]["dt_units"][m0])).strftime('%Y') for tt in t0]
                    end_print[m]   = [to_datetime(ds_all[m].time.values[tt-1].astype(s_data[species]["dt_units"][m0])).strftime('%Y') for tt in t1]
                    end_print[m]   = end_print[m] + [to_datetime(ds_all[m].time.values[-1].astype(s_data[species]["dt_units"][m0])).strftime('%Y')]
                else:
                    start_print[m] = [to_datetime(ds_all[m].time.values[tt].astype(s_data[species]["dt_units"][m0])).strftime('%m/%Y') for tt in t0]
                    end_print[m]   = [to_datetime(ds_all[m].time.values[tt-1].astype(s_data[species]["dt_units"][m0])).strftime('%m/%Y') for tt in t1]
                    end_print[m]   = end_print[m] + [to_datetime(ds_all[m].time.values[-1].astype(s_data[species]["dt_units"][m0])).strftime('%m/%Y')]

            elif period_all[m] == 'monthly':
                if chop_by == 'year':
                    nt[i] = total_times//(dt*12)
                    # Get indexes of start/end time stamps
                    t0 = [k for k in range(0,total_times,dt*12)]
                    t1 = [k for k in range(dt*12,total_times,dt*12)]

                    # Get dates of start/end time stamps
                    t0_date[m] = [to_datetime(ds_all[m].time.values[tt].astype(s_data[species]["dt_units"][m0])).strftime('%Y-%m-%d') for tt in t0]
                    t1_date[m] = [to_datetime(ds_all[m].time.values[tt].astype(s_data[species]["dt_units"][m0])).strftime('%Y-%m-%d') for tt in t1]
                    t1_date[m] = t1_date[m] + [end_date]

                    # Get start/end time stamps for caption
                    start_print[m] = [to_datetime(ds_all[m].time.values[tt].astype(s_data[species]["dt_units"][m0])).strftime('%Y') for tt in t0]
                    end_print[m]   = [to_datetime(ds_all[m].time.values[tt-1].astype(s_data[species]["dt_units"][m0])).strftime('%Y') for tt in t1]
                    end_print[m]   = end_print[m] + [to_datetime(ds_all[m].time.values[-1].astype(s_data[species]["dt_units"][m0])).strftime('%Y')]

                elif chop_by == 'season':
                    n_seasons = len(dt)
                    nt[i] = n_seasons
                    indexes[m] = []

                    # Get indexes of interest
                    for k in range(n_seasons):
                        indexes[m].append(list())
                        for ind,tt in enumerate(ds_all[m].time.values):
                            mm = int(to_datetime(tt.astype(s_data[species]["dt_units"][m0])).strftime('%m'))
                            if mm in dt[k]:
                                indexes[m][k].extend([ind])

                    # Get start/end time stamps for caption
                    ind_start = [dt[k][0] for k in range(n_seasons)]
                    ind_end   = [dt[k][-1] for k in range(n_seasons)]
                    start_print[m] = [month_names[ii-1] for ii in ind_start]
                    end_print[m]   = [month_names[ii-1] for ii in ind_end]

                else:
                    print(f'ERROR: option {chop_by} for chop_by not implemented. Options are year, month, season or a list of starting dates.')

            else:
                print(f'ERROR: inversion period of {m} is yearly. Set chop_by equal to year or to a list of starting dates.')

        n_cols = int(np.min(nt)) # only time intervals that are common to all models

        if n_cols == 0:
            print('ERROR: dt is greater than the number of timestamps for at least one of the models.')

    # Get sites info
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
    
    # Create figure
    if n_lines*n_cols== 4:
        # Re-organize the data for a nicer display
        fig,ax_tmp = plt.subplots(2,2,figsize=(2*4.2,2*3), #3.25
                                  subplot_kw={'projection':cartopy.crs.PlateCarree()})
        ax = ax_tmp.flatten()
    else:
        fig,ax = plt.subplots(n_lines,n_cols,figsize=(n_cols*4,n_lines*3), #3.25
                       subplot_kw={'projection':cartopy.crs.PlateCarree()})

    # Add map
    for i in range(n_lines):
        for j in range(n_cols):

            if n_cols == 1 and n_lines == 1:
                ax.add_feature(cartopy.feature.BORDERS,edgecolor=c_border,linewidth=1.)
                ax.coastlines(resolution='50m',color=c_border,linewidth=1.)
                ax.set_extent(region_limits[plot_area])

            else:
                if n_cols == 1:
                    ax_var = ax[i]
                elif n_lines == 1:
                    ax_var = ax[j]
                else:
                    ax_var = ax[i,j]

                ax_var.add_feature(cartopy.feature.BORDERS,edgecolor=c_border,linewidth=1.)
                ax_var.coastlines(resolution='50m',color=c_border,linewidth=1.)
                ax_var.set_extent(region_limits[plot_area])

    if plot_inversion_grid_flux:
        var_append = '_inversion_grid'
    else:
        var_append = ''

    # Plot fields
    for i in range(n_cols):
        for j,m in enumerate(ds_all.keys()):

            lon = ds_all[m].longitude.values
            lat = ds_all[m].latitude.values

            m0 = m.split('_')[0]

            # Compute averaged quantities
            if chop_by == 'season':
                
                if var == 'posterior_prior_diff':
                    try:
                        var_plot = np.mean(ds_all[m][f'flux_total_posterior{var_append}'][indexes[m][i],:,:],axis=0) - np.mean(ds_all[m]['flux_total_prior'][indexes[m][i],:,:],axis=0)
                    except:
                        print(f'Cannot find inversion_grid variables for {m} so using standard flux output.')
                        var_plot = np.mean(ds_all[m]['flux_total_posterior'][indexes[m][i],:,:],axis=0) - np.mean(ds_all[m]['flux_total_prior'][indexes[m][i],:,:],axis=0)
                    var_plot[np.where(var_plot) == np.nan] = 0.
                elif var == 'posterior_mean_diff':
                    try:
                        var_plot = np.mean(ds_all[m][f'flux_total_posterior{var_append}'][indexes[m][i],:,:],axis=0) - np.mean(ds_all[m][f'flux_total_posterior{var_append}'],axis=0)
                    except:
                        print(f'Cannot find inversion_grid variables for {m} so using standard flux output.')
                        var_plot = np.mean(ds_all[m][f'flux_total_posterior'][indexes[m][i],:,:],axis=0) - np.mean(ds_all[m][f'flux_total_posterior'],axis=0)
                    var_plot[np.where(var_plot) == np.nan] = 0.
                else:
                    try:
                        var_plot = np.mean(ds_all[m][f'{var}{var_append}'][indexes[m][i],:,:],axis=0)
                    except:
                        print(f'Cannot find inversion_grid variables for {m} so using standard flux output.')
                        var_plot = np.mean(ds_all[m][var][indexes[m][i],:,:],axis=0)
                        
                # Define string for caption
                if len(dt[i]) == 1:
                    time_out = (f'{start_print[m][i]}')
                else:
                    time_out = (f'{start_print[m][i]} - {end_print[m][i]}')

            else:
                if var == 'posterior_prior_diff':
                    try:
                        slice_apost   = ds_all[m][f'flux_total_posterior{var_append}'].sel(time=slice(t0_date[m][i],t1_date[m][i]))
                        slice_apriori = ds_all[m][flux_total_prior].sel(time=slice(t0_date[m][i],t1_date[m][i]))
                    except:
                        print(f'Cannot find inversion_grid variables for {m} so using standard flux output.')
                        slice_apost   = ds_all[m]['flux_total_posterior'].sel(time=slice(t0_date[m][i],t1_date[m][i]))
                        slice_apriori = ds_all[m]['flux_total_prior'].sel(time=slice(t0_date[m][i],t1_date[m][i]))
                    var_plot      = np.mean(slice_apost,axis=0) - np.mean(slice_apriori,axis=0)
                    var_plot[np.where(var_plot) == np.nan] = 0.
                elif var == 'posterior_mean_diff':
                    try:
                        mean_slice_apost = np.mean(ds_all[m]['flux_total_posterior'].sel(time=slice(t0_date[m][i],t1_date[m][i])),axis=0)
                        mean_apost       = np.mean(ds_all[m]['flux_total_posterior'],axis=0)
                    except:
                        print(f'Cannot find inversion_grid variables for {m} so using standard flux output.')
                        mean_slice_apost = np.mean(ds_all[m][f'flux_total_posterior{var_append}'].sel(time=slice(t0_date[m][i],t1_date[m][i])),axis=0)
                        mean_apost       = np.mean(ds_all[m][f'flux_total_posterior{var_append}'],axis=0)
                    var_plot         = mean_slice_apost - mean_apost
                    var_plot[np.where(var_plot) == np.nan] = 0.
                else:
                    try:
                        var_plot = np.mean(ds_all[m][f'{var}{var_append}'].sel(time=slice(t0_date[m][i],t1_date[m][i])),axis=0)
                    except:
                        print(f'Cannot find inversion_grid variables for {m} so using standard flux output.')
                        var_plot = np.mean(ds_all[m][var].sel(time=slice(t0_date[m][i],t1_date[m][i])),axis=0)

                # Define string for caption
                if dt == 1:
                    time_out = (f'{start_print[m][i]}')
                else:
                    time_out = (f'{start_print[m][i]} - {end_print[m][i]}')

            # Concatenate all models 2D-array
            if (plot_combined):
                if j == 0:
                    all_var_plot = np.expand_dims(var_plot, axis=0)
                else:
                    all_var_plot = np.concatenate((all_var_plot, np.expand_dims(var_plot, axis=0)),axis=0)

            # Make separate plots
            if not(plot_combined):
                if n_cols == 1 and n_lines == 1:
                    ax.pcolormesh(lon,lat,var_plot,cmap=cmap,vmin=lim[0],vmax=lim[1],
                                  shading='nearest')
                    ax.set_title(f'{m_data[m]["label"]}\n{time_out}')
                    ax_var = ax
                else:
                    if n_lines == 1:
                        ax_var = ax[i] 
                    elif n_cols == 1:
                        ax_var = ax[j]
                    else:
                        ax_var = ax[j,i]

                    ax_var.pcolormesh(lon,lat,var_plot,cmap=cmap,vmin=lim[0],vmax=lim[1],
                                      shading='nearest')
                    ax_var.set_title(f'{time_out}')
                    if i == 0:
                        if '\n' in m_data[m]["label"]:
                            ax_var.text(-0.14, 0.25, f'{m_data[m]["label"]}', transform=ax_var.transAxes, rotation=90)
                        else:
                            ax_var.text(-0.07, 0.25, f'{m_data[m]["label"]}', transform=ax_var.transAxes, rotation=90)
                    
                # Add site location
                if plot_site_locations == True:
                    if sites_info[m] is not None:
                        for s in sites_info[m]:
                            ax_var.scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],facecolor='none',
                                            edgecolor='red',marker='o',s=30,zorder=2)
                    
                # Add markers at specific locations
                if plot_point_markers is not None:
                    if i == 0:
                        print(f'\nPlotting markers for: {plot_point_markers}')
                        print(f'Edit lines below line {inspect.getframeinfo(inspect.currentframe()).lineno} to change marker colour')
                    for p in plot_point_markers:
                        if type(p) == list:
                            ax_var.scatter(p[0],p[1],facecolor='none',edgecolor='red',marker='^',s=30,zorder=2)
                        elif type(p) == str:
                            if p not in point_source_dict.keys():
                                print(f'{p} is not specified in point_source_dict, edit this to add a lat/lon location.')
                            else:
                                ax_var.scatter(point_source_dict[p][0],point_source_dict[p][1],facecolor='none',edgecolor='red',marker='^',s=30,zorder=2)

            #except:
            #    print(f'ERROR: Either start and end dates are incorrect or there is no model output from {m}.')
            #    print(f'Skipping plotting {m}.')

        # Plot combined
        if plot_combined:
            mean_var_plot = np.mean(all_var_plot,axis=0)

            if n_cols == 1:
                ax.pcolormesh(lon,lat,mean_var_plot,cmap=cmap,vmin=lim[0],vmax=lim[1],
                              shading='nearest')
                ax_var = ax
            else:
                ax_var = ax[i]
                ax_var.pcolormesh(lon,lat,mean_var_plot,cmap=cmap,vmin=lim[0],vmax=lim[1],
                                  shading='nearest')

            ax_var.set_title(f'{time_out}')
                
            # Add site location
            if plot_site_locations == True:
                if sites_info[m] is not None:
                    for s in sites_info[m]:
                        ax_var.scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],facecolor='none',
                                        edgecolor='red',marker='o',s=30,zorder=2)
                
            # Add markers at specific locations
            if plot_point_markers is not None:
                if i == 0:
                    print(f'\nPlotting markers for: {plot_point_markers}')
                    print(f'Edit lines below line {inspect.getframeinfo(inspect.currentframe()).lineno} to change marker colour')
                for p in plot_point_markers:
                    if type(p) == list:
                        ax_var.scatter(p[0],p[1],facecolor='none',edgecolor='red',marker='^',s=30,zorder=2)
                    elif type(p) == str:
                        if p not in point_source_dict.keys():
                            print(f'{p} is not specified in point_source_dict, edit this to add a lat/lon location.')
                        else:
                            ax_var.scatter(point_source_dict[p][0],point_source_dict[p][1],facecolor='none',edgecolor='red',marker='^',s=30,zorder=2)

    #flux colorbar
    cbar = plt.cm.ScalarMappable(cmap=cmap)
    levels = np.linspace(lim[0],lim[1])
    cbar.set_array(levels)
    cbar.set_clim(lim)

    # Size of color bar
    f_height = 0.9
    f_bottom = (1-f_height)/2
    f_width = 0.04/n_cols #0.02
    f_left = 0.95         #0.94

    if n_cols == 1 and n_lines == 1:
        cbar_ax = fig.add_axes([1, f_bottom, f_width, f_height])
        color_bar = fig.colorbar(cbar,cax=cbar_ax,orientation='vertical',cmap=cmap,extend=extend)
    elif n_cols*n_lines == 4:
        cbar_ax = fig.add_axes([f_left, f_bottom, 0.02, f_height])
        color_bar = fig.colorbar(cbar,cax=cbar_ax,orientation='vertical',cmap=cmap,extend=extend)       
    elif n_lines == 1:
        cbar_ax = fig.add_axes([f_left, f_bottom, f_width, f_height])
        color_bar = fig.colorbar(cbar,cax=cbar_ax,orientation='vertical',cmap=cmap,extend=extend)
    else:
        # Size of color bar
        f_height = 0.95*2/n_lines
        f_bottom = (1-f_height)/2
        if n_cols == 1: f_left = 1

        cbar_ax = fig.add_axes([f_left, f_bottom, f_width, f_height])
        color_bar = fig.colorbar(cbar,cax=cbar_ax,orientation='vertical',cmap=cmap,extend=extend)

    color_bar.set_label(f'{var_labels[var]} {s_data[species]["species_print"]} ({flux_units})')
    fig.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05, wspace=0.04, hspace=0.12)

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

def convert_molar_to_mass_flux(ds, M):
    """
    Converts spatial flux variables in the flux dataset from mol/m²/s to kg/km²/year.
    
    Args:
        ds (xarray dataset): The xarray dataset containing flux variables.
        M (float): The molar mass in g/mol.
    """
    
    # Convert the molar mass to kg/mol
    M_kg = M * 0.001  # Convert grams to kilograms
    
    # List to store names of converted variables
    converted_vars = []
    
    for var_name, variable in ds.items():
        if 'units' in variable.attrs and variable.attrs['units'] == 'mol m-2 s-1':
            # Convert the variable data
            # Multiply by molar mass, seconds per year, and convert m² to km²
            ds[var_name] = variable * M_kg * 31.536e6 # kg/km²/year
            ds[var_name].attrs['units'] = 'kg km-2 yr-1' # Update the units
            
            # Add the variable name to the list of converted variables
            converted_vars.append(var_name)

    print("Converting molar flux variables to mass flux (kg km-2 yr-1)")
    
    return ds

#####################################################################

def set_flux_limits(ds_all, var, region_plot, species_info, option='default', custom_percentile=None):
    """
    Set flux limits based on the option provided:
    
    1. `default` - use default values from species_info.
    2. List or tuple with two values (e.g., [0, 1]) - use specified limits.
    3. 'auto' - auto-calculate limits based on data percentiles.
    
    Args:
        ds_all (dict): A dictionary of datasets containing the flux variables.
        var (str): The variable name to compute limits for.
        region_plot (list/tuple): Coordinates [lon_min, lon_max, lat_min, lat_max].
        species_info (dict): Contains default limit values.
        option (None, str or list/tuple): The option for setting limits.
            - `default` for default values.
            - A list or tuple with two elements (min, max) for specified limits.
            - 'auto' for auto-calculated values using the 99th percentile (if custom_percentile is None).
        custom_percentile (float, optional): The percentile to use as the upper limit if option is 'auto'.
    
    Returns:
        tuple: A tuple containing the flux limits (min, max).
        
    Raises:
        ValueError: If `fluxlim[0] >= fluxlim[1]` or an invalid option is provided.
    """
   
    if option == 'default':  # Case 1: Use default values from species_info
        if var in ['posterior_prior_diff', 'posterior_mean_diff']:
            if 'difflim' in species_info:
                fluxlim = tuple(species_info['difflim'])
            else:
                raise KeyError("Key 'difflim' not found in species_info.")
        else:
            if 'fluxlim' in species_info:
                fluxlim = tuple(species_info['fluxlim'])
            else:
                raise KeyError("Key 'fluxlim' not found in species_info.")    
                
    elif isinstance(option, (list, tuple)) and len(option) == 2:  # Case 2: Use specified values [min, max]
        fluxlim = (option[0], option[1])
        
    elif option == 'auto':  # Case 3: Auto-calculate limits based on percentiles
        all_var = []
        for j, m in enumerate(ds_all.keys()):
            if var == 'posterior_prior_diff':
                var_j = ds_all[m]['flux_total_posterior'] - ds_all[m]['flux_total_prior']
            elif var == 'posterior_mean_diff':
                var_j = ds_all[m]['flux_total_posterior'] - np.mean(ds_all[m]['flux_total_posterior'], axis=0)            
            else:
                var_j = ds_all[m][var]
                
            # Filter based on longitude and latitude of region_plot
            mask_region = ((ds_all[m].longitude > region_plot[0]) &
                           (ds_all[m].longitude < region_plot[1]) &
                           (ds_all[m].latitude > region_plot[2]) &
                           (ds_all[m].latitude < region_plot[3]))
            var_j = var_j.where(mask_region).dropna(dim='longitude', how='all').dropna(dim='latitude', how='all')
            all_var.append(var_j.values)
            
        all_var = np.concatenate(all_var, axis=0)  # Concatenate along the time axis (axis=0)
        
        if custom_percentile is None:
            upper_lim = np.quantile(all_var, .99) # Calculate the 99th percentile
        else:
            upper_lim = np.quantile(all_var, custom_percentile)
        
        if var in ['posterior_prior_diff', 'posterior_mean_diff']:
            fluxlim = (-upper_lim, upper_lim)
        else:
            fluxlim = (0, upper_lim)
            
    else:
        # Raise an error if the option is invalid
        raise ValueError(f"Invalid option '{option}'. Use 'default', a [min, max] list/tuple, or 'auto'.")
        
    # Validation: Check that fluxlim[0] is smaller than fluxlim[1]
    if fluxlim[0] >= fluxlim[1]:
        raise ValueError(f"The lower flux limit {fluxlim[0]} must be less than the upper flux limit {fluxlim[1]}.") 
        
    return fluxlim

def calc_rolling_mean_v0(list_data,rolling_mean=2):
    """
    Calculate rolling mean of a list of numpy array using numpy.convolv
    (see https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy).
    
    Args:
        list_data : list of numpy array of dtype float or numpy.datetime64
        rolling_mean : rolling_mean period
        
    Return
        averaged_data : list of numpy array containing the averaged data.
    """
    if rolling_mean is None :
        return list_data
    
    elif rolling_mean>=list_data[0].size:
        raise ValueError(f'rolling_mean value ({rolling_mean}) should be inferior to the size of the data ({list_data[0].size}).')
        
    else :
        averaged_data = list()
        for data in list_data:
            if np.issubdtype(data.dtype, np.datetime64):
                averaged_data.append((np.convolve(data.astype(int), np.ones(rolling_mean), 'valid') / rolling_mean
                                     ).astype(data.dtype))
            else : 
                averaged_data.append(np.convolve(data, np.ones(rolling_mean), 'valid') / rolling_mean)
                
        return averaged_data
 
def calc_3yr_rolling_mean_v1(list_data):
    """
    Calculate rolling mean of a list of numpy array using numpy.convolv
    (see https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy).
    
    Args:
        list_data : list of numpy array of dtype float or numpy.datetime64
        
    Return
        averaged_data : list of numpy array containing the averaged data.
    """
    rolling_mean = 3
    averaged_data = list()
    for data in list_data:
        if np.issubdtype(data.dtype, np.datetime64):
            tmp = (np.convolve(data.astype(int), np.ones(rolling_mean), 'valid') / rolling_mean
                  ).astype(data.dtype)
            tmp = np.concatenate([[data[0],],tmp,[data[-1],]])
        else : 
            tmp = np.convolve(data, np.ones(rolling_mean), 'valid') / rolling_mean
            tmp = np.concatenate([[np.mean(data[:2]),],tmp,[np.mean(data[-2:]),]])
        averaged_data.append(tmp)
    return averaged_data
 
def calc_3yr_rolling_mean_v2(list_data):
    """
    Calculate rolling mean of a list of numpy array using numpy.convolv
    (see https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy).
    
    Args:
        list_data : list of numpy array of dtype float or numpy.datetime64
        
    Return
        averaged_data : list of numpy array containing the averaged data.
    """
    rolling_mean = 3
    averaged_data = list()
    for data in list_data:
        print(data,type(data))
        if np.issubdtype(data.dtype, np.datetime64):
            tmp = (np.convolve(data.astype(int), np.ones(rolling_mean), 'valid') / rolling_mean
                  ).astype(data.dtype)
            tmp = np.concatenate([[data[0],],tmp,[data[-1],]])
        else : 
            tmp = np.convolve(data, np.ones(rolling_mean), 'valid') / rolling_mean
            tmp = np.concatenate([[data[0],],tmp,[data[-1],]])
        averaged_data.append(tmp)
    return averaged_data


def plt_diff(mean, unc,title):
    plt.plot(mean[0],mean[1],ls='-',c='black')
    plt.fill_between(unc[0],unc[1],unc[2],color='gray',alpha=0.5)
    
    for i,f in enumerate([calc_rolling_mean,
                          calc_3yr_rolling_mean_v1,
                          # calc_3yr_rolling_mean_v2
                         ]):
        mean1 = f(mean)
        unc1 = f(unc)
        plt.plot(mean1[0],mean1[1],ls='-',c=f'C0{i}')
        plt.fill_between(unc1[0],unc1[1],unc1[2],color=f'C0{i}',alpha=0.05)
    plt.grid(True)
    plt.title(title)
    plt.show()

calc_rolling_mean = calc_3yr_rolling_mean_v1