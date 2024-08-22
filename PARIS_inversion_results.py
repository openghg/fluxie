import xarray as xr
import numpy as np
from pandas import to_datetime
import matplotlib.pyplot as plt
import os
import glob
import math
from matplotlib.dates import YearLocator, MonthLocator, DayLocator
from matplotlib.ticker import NullFormatter
import pprint
import cartopy
from json import load
import matplotlib.dates as mdates
import inspect

model_colors = {'intem':[['darkslateblue','dodgerblue'],
                         ['black','grey']],
                'rhime':[['darkgreen','green']],
                'elris':[['purple','mediumpurple']]}

model_q_indices = {'intem':[0,1],
                   'rhime':[0,1],
                   'elris':[0,1],
                   'mcmc':[0,1]}

point_source_dict = {'paris':[2.340430,48.860050],
                     'london':[-0.127799,51.507593],
                 'nw_england':[-2.796870,53.774820]}

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

annotate_coords = {0:[0.7,0.80],
                   1:[0.7,0.60],
                   2:[0.7,0.40],
                   3:[0.7,0.20],}

# population from 2018 to 2023 (at Jan 1 each year)
bel_pop = np.array([11.399,11.455,11.522,11.555,11.618,11.723])
lux_pop = np.array([0.602,0.614,0.626,0.635,0.645,0.661])
bel_pop_r = np.round(np.mean(bel_pop/(bel_pop+lux_pop)),3)

font = {'size':12}
plt.rc('font', **font)

### read in species info file

filename = os.path.join(os.getcwd(),'species_info.json')

if os.path.exists(filename) == False:
    print('ERROR: Cannot find species_info.json file. Check that this exists in the same directory as your notebook.')

with open(filename, "r") as f:
    s_data = load(f)
    
print('NOTE: If plotting units or scales look odd, edit species_info.json to fix this.')

#####################################################################

def set_model_colors(models):
    cList = [['darkslateblue','dodgerblue'],
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

def read_flux(data_dir,species,models,model_filenames,period_override=None):
    """
    Extracts flux and country flux timeseries from each model.
    
    Args:
        data_dir (str): 
            Path to top data directory.
        species (str): 
            Gas species, e.g. 'ch4'.
        models (list of str): 
            Keys specifying model names, e.g. ['intem','elris']
        model_filenames (dict of str): 
            Paired models and filenames, e.g. {'intem':'InTEM_NAME_EUROPE',
                                               'elris':'ELRIS_NAME_EUROPE_baselinetest'}
        period_override (list of str) (optional):
            Inversion periods to include, to override the standards in species_info.json.
            Must be the same length as models, e.g. ['monthly',None,'yearly']
                                       
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
        print(f'\nAttempting to read data from {m}')
        
        m0 = m.split('_')[0]
        
        model_dir = model_filenames[m].split('_')[0]

        try:
            filepath = glob.glob(os.path.join(data_dir,model_dir,species,
                                              f'{model_filenames[m]}_{s_data[species]["model_species"][m0]}_{period_all[m]}.nc'))
            print(f'Reading data from: {filepath[0]}')
            with xr.open_dataset(filepath[0]) as in_ds:
                ds_all[m] = in_ds
                print('Done!')
        except:
            try:
                if (model_filenames[m].split('_')[-1] == 'std*'):
                    alternative_filename = f'{model_filenames[m][0:-5]}_{m0}_obs_{m0}_baseline_optimized'
                    filepath = glob.glob(os.path.join(data_dir,model_dir,species,f'{alternative_filename}_{s_data[species]["model_species"][m0]}_{period_all[m]}.nc'))
                    print(f'Cannot find {m} file for {species}. Reading data from: {filepath[0]}')
                    with xr.open_dataset(filepath[0]) as in_ds:
                        ds_all[m] = in_ds
                    print('Done!')
                else:
                    print(f'Failed!')
                    print(f'Cannot find {m} file for {species}. This model will not be plotted')
            except:
                print(f'Failed!')
                print(f'Cannot find {m} file for {species}. This model will not be plotted')
    
    return ds_all

#####################################################################

def slice_flux(ds_all,start_date,end_date,
               scale_units=True,species=None):
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
        scale_units (bool): 
            If True, scales country fluxes to Tg or Gy per year.
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
                'covariance_country_flux_total_posterior']

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
            print(f'Scaling {m} units by {s_data[species]["units_scaling"][m0]}')
            if ds_all[m] is not None:
                var_names = [k for k in ds_all[m].keys() if k not in skip_var]
                for v in var_names:
                    ds_all[m][v].values = ds_all[m][v].values/s_data[species]["units_scaling"][m0]

                cov_var = 'covariance_country_flux_total_posterior'
                if cov_var in ds_all[m].keys():
                    ds_all[m][cov_var].values = ds_all[m][cov_var].values/s_data[species]["units_scaling"][m0]**2
                    print(f'Scaling covariance units in {m} by {s_data[species]["units_scaling"][m0]**2}')
        
    return ds_all

#####################################################################

def read_mf(data_dir,species,models,model_filenames,period_override=None):
    """
    Extracts mole fraction timeseries data from each model.
    Args:
        data_dir (str): 
            Path to top data directory.
        species (str): 
            Gas species, e.g. 'ch4'.
        models (list of str): 
            Keys specifying model names, e.g. ['intem','elris']
        model_filenames (dict of str): 
            Paired models and filenames, e.g. {'intem':'InTEM_NAME_EUROPE',
                                               'elris':'ELRIS_NAME_EUROPE_baselinetest'}
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
        model_dir = model_filenames[m].split('_')[0]
        
        print(f'\nAttempting to read data from {m}')
        try:
            filepath = glob.glob(os.path.join(data_dir,model_dir,species,f'{model_filenames[m]}_{s_data[species]["model_species"][m0]}_{period_all[m]}_concentrations.nc'))
            print(f'Reading data from: {filepath[0]}')
            with xr.open_dataset(filepath[0]) as in_ds:
                ds_all[m] = in_ds
            print('Done!')
        except:
            try:
                if (model_filenames[m].split('_')[-1] == 'std*'):
                    alternative_filename = f'{model_filenames[m][0:-5]}_{m0}_obs_{m0}_baseline_optimized'
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

def slice_mf(ds_all,start_date=None,end_date=None,site=None,
             baseline_site=None,data_dir=None,
             scale_units=False,
             species=None):
    """
    Slices down the mole fraction timeseries data, to within the
    given time limits, and/or for the chosen site.
    
    Args:
        ds_all (dictionary of datasets): 
            xarray datasets read directly from each model's flux netCDF.
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
        
        m0 = m.split('_')[0]
        
        if 'mcmc' in m0:
            s = f'_{species}'
        else:
            s = ''
        
        print(f'\nMasking data from {m}')
        
        if 'Yav' in ds_all[m].keys():
            offset = int(np.mean(ds_all[m]['Yav'].values))
        else:
            offset = (ds_all[m][f'time{s}'].values[1].astype('datetime64[h]') - ds_all[m][f'time{s}'].values[0].astype('datetime64[h]')).astype(int)

        # fix to move elris timestamps back to the middle of av period - to be removed once fixed in .nc files
        if 'elris_old' in m:
            ds_all[m]['time'] = ds_all[m]['time'] - np.timedelta64(offset,'h')/2

        # round seconds to integer (correction for elris)
        if 'elris' in m:
            ds_all[m]['time'] = ds_all[m]['time'].dt.round('s')

        if site is not None:
            try:
                site_index = np.where(ds_all[m][f'sitenames{s}'].astype(str) == site)[0][0]
                ds_all[m] = ds_all[m].sel(**{f'time{s}':slice(start_date,end_date),
                                            f'nsite{s}':site_index})
            except:
                ds_all[m] = None
                print(f'No {m} obs found for {site} between {start_date} and {end_date}')
        else:
            try:
                ds_all[m] = ds_all[m].sel(**{f'time{s}':slice(start_date,end_date)})
            except:
                ds_all[m] = None
                print(f'No {m} obs found between {start_date} and {end_date}')
                
        if scale_units == True:
            print(f'Scaling {m} units by {s_data[species]["mf_units_scaling"]}')
            if ds_all[m] is not None:
                var_names = [k for k in ds_all[m].keys() if k not in ['Yav','sitenames','median_poll_uncert_flag']]
                print(ds_all[m].species)

                if type(ds_all[m].species) is not list:
                    species_test = [ds_all[m].species]
                else:
                    species_test = ds_all[m].species
                for s in species_test:
                    if f'sitenames_{s}' in var_names:
                        var_names.remove(f'sitenames_{s}')
                for v in var_names:
                    ds_all[m][v] = ds_all[m][v]/s_data[species]["mf_units_scaling"]
      
        if baseline_site is not None:
            print('Masking timeseries to only include baseline times')

            try:
                                        
                #average baseline mask over obs averaging period
                b = baseline.resample(**{f'time{s}':f'{offset}H'}).mean()
                #adjust baseline mask time back to centre of av period (resample removes this)
                b[f'time{s}'] = b[f'time{s}'] + np.timedelta64(offset,'h')/2
                                    
                #mask baseline mask again, to only include timestamps where every period in the averaging period is classified as baseline
                b_masked = b.sel(**{f'time{s}':b[f'time{s}'].values[np.where(b['baseline'] == 1.)]})
                                
                #mask dataset using only baseline times
                both_times = np.isin(ds_all[m][f'time{s}'].values,b_masked[f'time{s}'].values)
                                
                ds_all[m] = ds_all[m].sel(**{f'time{s}':both_times})
                    
            except:
                print('Failed to mask {m} data by baseline times')
    
    check_keys = list(ds_all.keys())
    for m in check_keys:
        if ds_all[m] is None:
            ds_all.pop(m)
                
    return ds_all

#####################################################################

def stats_mf(ds_all,species):
    """
    Calculates the Pearson correlation coefficent and normalised root
    mean square error, of the fit between the posterior mean mf and the 
    observed mole fraction.
    
    Args:
        ds_all (dictionary of datasets):
            xarray datasets from slice_mf(), sliced between chosen dates
            but still containing all sites.
        species (str):
            Used to extract data for correct species when using multi gas MCMC model.
    Returns:
        pearson (dictionary of dictionaries):
            Pearson correlation coeffiecient, for each site and for each model.
        nrmse (dictionary of dictionaries):
            Normalised root mean square error, for each site and for each model.
    """
    
    sites_all = np.array([])

    for i,m in enumerate(ds_all.keys()):
        m0 = m.split('_')[0]
        
        if 'mcmc' in m0:
            s = f'_{species}'
        else:
            s = ''
        
        sites_all = np.hstack((sites_all,ds_all[m][f'sitenames{s}'].values.astype(str)))
    
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
            
            m0 = m.split('_')[0]
        
            if 'mcmc' in m0:
                sp = f'_{species}'
            else:
                sp = ''
            
            if site in ds_all[m][f'sitenames{sp}'].values.astype('str'):
                s = np.where(ds_all[m][f'sitenames{sp}'].values.astype('str') == site)[0][0]
                if ds_all[m][f'Yobs{sp}'].values[:,s][~np.isnan(ds_all[m][f'Yobs{sp}'].values[:,s])].shape[0] != 0:
                    pearson[site][m] = np.round(np.corrcoef(ds_all[m][f'Yobs{sp}'].values[:,s][~np.isnan(ds_all[m][f'Yobs{sp}'].values[:,s])],
                                                            ds_all[m][f'Yapost{sp}'].values[:,s][~np.isnan(ds_all[m][f'Yobs{sp}'].values[:,s])])[0,1],3)
                    nrmse[site][m] = np.round(np.sqrt(np.mean((ds_all[m][f'Yapost{sp}'].values[:,s][~np.isnan(ds_all[m][f'Yobs{sp}'].values[:,s])]-
                                                    ds_all[m][f'Yobs{sp}'].values[:,s][~np.isnan(ds_all[m][f'Yobs{sp}'].values[:,s])])**2))/np.mean(ds_all[m][f'Yobs{sp}'].values[:,s][~np.isnan(ds_all[m][f'Yobs{sp}'].values[:,s])]),3)
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

def plot_obs_modelled_separate(ds_all,species,site,model_labels,
                               model_colors,
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
        model_labels (dict of str):
            Models and corresponding strings used to describe the model in the 
            plot legend.
        model_colors (dict of str):
            Models and corresponding colours used to plot the model.
        include (list of str):
            Variables included in the plot, options for 'Yobs', 'Yapriori',
            'Yapost', 'YaprioriBC', 'YapostBC'.
        diff_include (list of str):
            Variables included in the 'obs - variable' difference histogram, 
            same options as above.
        add_unc (bool):
            if True, plot uncertainty bar on Yobs and Yapost timeseries.
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
        
        if m0 == 'mcmc':
            s = f'_{species}'
        else:
            s = ''
            
        for var in include:

            if var == 'Yobs':
                if len(include) == 1:
                    ax.scatter(ds_all[m][f'time{s}'].values,
                               ds_all[m][f'Yobs{s}'].values,
                               color=model_colors[m][var_colors[var]],
                               label=f'Obs ({model_labels[m]})',s=8,alpha=0.8,marker='s')
                    
                    if add_unc:
                        try:
                            ax.errorbar(ds_all[m][f'time{s}'].values,
                                        ds_all[m][f'Yobs{s}'].values,
                                        ds_all[m][f'uYobs_repeatability{s}'].values,
                                        color=model_colors[m][var_colors[var]],alpha=0.4,fmt='none')
                        except:
                            ax.errorbar(ds_all[m][f'time{s}'].values,
                                        ds_all[m][f'Yobs{s}'].values,
                                        ds_all[m][f'uYobs{s}'].values,
                                        color=model_colors[m][var_colors[var]],alpha=0.4,fmt='none')
                            print(f'WARNING: uYobs_repeatability is not present in {m}. uYobs is being plotted instead as error bars.')
                            
                else:
                    
                    ax.scatter(ds_all[m][f'time{s}'].values,
                                ds_all[m][f'Yobs{s}'].values,
                                color='black',label=f'Obs ({model_labels[m]})',s=8,alpha=0.8,
                                marker='s')
                    
                    if add_unc:
                        try:
                            ax.errorbar(ds_all[m][f'time{s}'].values,
                                        ds_all[m][f'Yobs{s}'].values,
                                        ds_all[m][f'uYobs_repeatability{s}'].values,
                                            color='black',alpha=0.4,fmt='none')
                        except:
                            ax.errorbar(ds_all[m][f'time{s}'].values,
                                        ds_all[m][f'Yobs{s}'].values,
                                        ds_all[m][f'uYobs{s}'].values,
                                            color='black',alpha=0.4,fmt='none')
                            print(f'WARNING: uYobs_repeatability is not present in {m}. uYobs is being plotted instead as error bars.')

            else:
                try:
                    ax.plot(ds_all[m][f'time{s}'].values,
                            ds_all[m][f'{var}{s}'].values,
                            color=model_colors[m][var_colors[var]],alpha=0.8,
                            linewidth=2.,
                            label=f'{model_labels[m]} {var_labels[var]}')
                
                except:
                    #handle old ncdf files
                    if var == 'uYmod':
                        uYmod = ds_all[m][f'Yobs{s}'].values - ds_all[m]['qYmod'].values[:,model_q_indices[m0][0]]
                        ax.plot(ds_all[m][f'time{s}'].values,
                                uYmod,
                                color=model_colors[m][var_colors[var]],alpha=0.8,
                                linewidth=2.,
                                label=f'{model_labels[m]} {var_labels[var]}')
                        print(f'WARNING: uYmod is not present in {m}. This quantity is being computed from qYmod.')

                    elif var == 'uYobs_repeatability':
                        ax.plot(ds_all[m][f'time{s}'].values,
                                ds_all[m][f'uYobs{s}'].values,
                                color=model_colors[m][var_colors[var]],alpha=0.8,
                                linewidth=2.,
                                label=f'{model_labels[m]} {var_labels[var]}')
                        print(f'WARNING: uYobs_repeatability is not present in {m}. uYobs is being plotted instead.')

                    else:
                        print(f'ERROR: variable {var} not found in {m} or deprecated!')

                if (var == 'Yapost') and add_unc:
                    ax.fill_between(ds_all[m][f'time{s}'].values,
                                    ds_all[m][f'qYapost{s}'].values[:,model_q_indices[m0][0]],
                                    ds_all[m][f'qYapost{s}'].values[:,model_q_indices[m0][1]],
                                    color=model_colors[m][var_colors[var]],alpha=0.2)

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
                var_plot = ds_all[m][f'Yobs{s}'].values - ds_all[m][f'{var}{s}'].values
            else:
                try:
                    var_plot = ds_all[m][f'{var}{s}'].values
                except:
                    if var == 'uYmod':
                        var_plot = f'uYmod{s}'
                    elif var == 'uYobs_repeatability':
                        var_plot = ds_all[m][f'uYobs{s}'].values
                    else:
                        continue

            if np.abs(np.nanmean(var_plot)) <= 0.01:
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
        n_obs = (~np.isnan(ds_all[m][f'Yobs{s}'].values)).sum()
        ax2.annotate('\n$N_{obs}$: '+str(n_obs),xy=[0.65,1.05],xycoords='axes fraction',color='k')

        ax2.set_xlabel(legend_hist)
    
        min_mf.append(ax.get_ylim()[0])
        max_mf.append(ax.get_ylim()[1])
        
        ax.set_title(model_labels[m])
        ax.set_ylabel(f'{s_data[species]["species_print"]} {site} ({s_data[species]["mf_units_print"]})')
        leg = ax.legend(ncol=2,borderpad=.2,columnspacing=1.0)
        try:
            for l in leg.legend_handles:
                l.set_linewidth(5.0)
        except:
            for l in leg.legendHandles:
                l.set_linewidth(5.0)
        '''        
        if int(ds_all[m][f'time{s}'].values[-1].astype('datetime64[M]')-ds_all[m][f'time{s}'].values[0].astype('datetime64[M]')) > 12:
            ax.xaxis.set_minor_locator(MonthLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.xaxis.set_major_locator(YearLocator())
        else:
            ax.xaxis.set_major_locator(MonthLocator())
            ax.xaxis.set_minor_locator(DayLocator())
        '''
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        
                    
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

def plot_obs_modelled_together(ds_all,species,site,model_labels,
                               model_colors,
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
        model_labels (dict of str):
            Models and corresponding strings used to describe the model in the 
            plot legend.
        model_colors (dict of str):
            Models and corresponding colours used to plot the model.
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
        
        if m0 == 'mcmc':
            s = f'_{species}'
        else:
            s = ''
                
        for var in include:

            if var == 'Yobs':
                if len(include) == 1:
                    ax.scatter(ds_all[m][f'time{s}'].values,
                                ds_all[m][f'Yobs{s}'].values,
                                color=model_colors[m][var_colors[var]],label=f'Obs ({model_labels[m]})',s=5,alpha=0.5)

                    if add_unc:
                        try:
                            ax.errorbar(ds_all[m][f'time{s}'].values,
                                        ds_all[m][f'Yobs{s}'].values,
                                        ds_all[m][f'uYobs_repeatability{s}'].values,
                                        color=model_colors[m][var_colors[var]],alpha=0.4,fmt='none')

                        except:
                            #handle old ncdf files
                            ax.errorbar(ds_all[m][f'time{s}'].values,
                                        ds_all[m][f'Yobs{s}'].values,
                                        ds_all[m][f'uYobs{s}'].values,
                                        color=model_colors[m][var_colors[var]],alpha=0.4,fmt='none')
                            print(f'WARNING: uYobs_repeatability is not present in {m}. uYobs is being plotted instead as error bars.')

                else:
                    ax.scatter(ds_all[m][f'time{s}'].values,
                                ds_all[m][f'Yobs{s}'].values,
                                color='dimgrey',label=f'Obs ({model_labels[m]})',s=5,alpha=0.5)

            else:
                try:
                    ax.scatter(ds_all[m][f'time{s}'].values,
                            ds_all[m][f'{var}{s}'].values,
                            color=model_colors[m][var_colors[var]],alpha=0.5,
                            label=f'{model_labels[m]} {var_labels[var]}',
                            linewidth=2,s=5)

                except:
                    # handle old ncdf files
                    if var == 'uYmod':
                        uYmod = ds_all[m][f'Yobs{s}'].values - ds_all[m][f'qYmod{s}'].values[:,model_q_indices[m0][0]]
                        ax.scatter(ds_all[m][f'time{s}'].values,
                                   uYmod,
                                   color=model_colors[m][var_colors[var]],
                                   label=f'{model_labels[m]} {var_labels[var]}',
                                   linewidth=2,s=5,alpha=0.5)
                        print(f'WARNING: uYmod is not present in {m}. This quantity is being computed from qYmod.')

                    elif var == 'uYobs_repeatability':
                        ax.scatter(ds_all[m][f'time{s}'].values,
                                   ds_all[m][f'uYobs{s}'].values,
                                   color=model_colors[m][var_colors[var]],
                                   label=f'{model_labels[m]} {var_labels[var]}',
                                   linewidth=2.,s=5,alpha=0.5)
                        print(f'WARNING: uYobs_repeatability is not present in {m}. uYobs is being plotted instead.')

                    else:
                        print(f'ERROR: variable {var} not found in {m} or deprecated!')

                if (var == 'Yapost') and add_unc:
                    ax.fill_between(ds_all[m][f'time{s}'].values,
                                    ds_all[m][f'qYapost{s}'].values[:,model_q_indices[m0][0]],
                                    ds_all[m][f'qYapost{s}'].values[:,model_q_indices[m0][1]],
                                    color=model_colors[m][var_colors[var]],alpha=0.3)
        

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
                var_plot = ds_all[m][f'Yobs{s}'].values - ds_all[m][f'{var}{s}'].values
            else:
                try:
                    var_plot = ds_all[m][f'{var}{s}'].values
                except:
                    if var == 'uYmod':
                        var_plot = f'uYmod{s}'
                    elif var == 'uYobs_repeatability':
                        var_plot = ds_all[m][f'uYobs{s}'].values
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
    '''
    if int(ds_all[m].time.values[-1].astype('datetime64[M]')-ds_all[m].time.values[0].astype('datetime64[M]')) > 12:
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_locator(YearLocator())
    else:
        ax.xaxis.set_major_locator(MonthLocator())
    '''
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
        
    if y_lim is None:
        ax.set_ylim([min(min_mf)-(0.02*min(min_mf)),
                                max(max_mf)+(0.05*max(max_mf))])
    else:
        ax.set_ylim(y_lim)
        
    print('NOTE: If all the data is not within axis limits, adjust the set_ylim')
    print('NOTE: If annotations in the histograms are not displaying correctly, adjust annotate_coords.')
    
    return fig

#####################################################################

def plot_obs_diff(ds_all,species,site,model_labels,
                               model_colors,
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
        model_labels (dict of str):
            Models and corresponding strings used to describe the model in the 
            plot legend.
        model_colors (dict of str):
            Models and corresponding colours used to plot the model.
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
    
    m00 = models[0].split('_')[0]
    m01 = models[1].split('_')[0]
    
    if m00 == 'mcmc':
        s0 = f'_{species}'
        time_name0 = f'time_{species}'
    else:
        s0 = ''
        time_name0 = 'time'
        
    if m01 == 'mcmc':
        s1 = f'_{species}'
        time_name1 = f'time_{species}'
    else:
        s1 = ''
        time_name1 = 'time'

    both_times0 = np.isin(ds_all[models[0]][f'time{s0}'].values,ds_all[models[1]][f'time{s1}'].values)
    both_times1 = np.isin(ds_all[models[1]][f'time{s1}'].values,ds_all[models[0]][f'time{s0}'].values)
    
    ds_all[models[0]] = ds_all[models[0]].sel(**{time_name0:both_times0})
    ds_all[models[1]] = ds_all[models[1]].sel(**{time_name1:both_times1})
    
    #ds_all[models[1]] = ds_all[models[1]].sel(time=both_times1)
            
    for var in include:
        try:
            ax.scatter(ds_all[models[0]].time.values,
                       ds_all[models[0]][f'{var}{s0}'].values - ds_all[models[1]][f'{var}{s1}'].values,
                       color=model_colors[models[0]][var_colors[var]],alpha=0.5,
                       label=f'{model_labels[models[0]]} - {model_labels[models[1]]}\n{var_labels[var]}',
                       linewidth=2,s=8)

        except:
            # handle old ncdf files
            if var == 'uYmod':

                try:
                    uYmod0 = ds_all[models[0]][f'Yobs{s0}'].values - ds_all[models[0]][f'qYmod{s0}'].values[:,model_q_indices[m00][0]]
                    uYmod1 = ds_all[models[1]][f'Yobs{s1}'].values - ds_all[models[1]][f'qYmod{s1}'].values[:,model_q_indices[m01][0]]

                    ax.scatter(ds_all[models[0]].time.values,
                                uYmod0 - uYmod1,
                                color=model_colors[models[0]][var_colors[var]],alpha=0.5,
                                label=f'{model_labels[models[0]]} - {model_labels[models[1]]}\n{var_labels[var]}',
                                linewidth=2,s=8)
                    print(f'WARNING: uYmod is not present in both models. This quantity is being computed from qYmod.')

                except:
                    print(f'ERROR: {models[0]} and {models[1]} have different definitions of uYmod!')

            elif var == 'uYobs_repeatability':
                try:
                    ax.scatter(ds_all[models[0]].time.values,
                                ds_all[models[0]][f'uYobs{s0}'].values - ds_all[models[1]][f'uYobs{s1}'].values,
                                color=model_colors[models[0]][var_colors[var]],alpha=0.5,
                                label=f'{model_labels[models[0]]} - {model_labels[models[1]]}\n{var_labels[var]}',
                                linewidth=2,s=8)
                    print(f'WARNING: uYobs_repeatability is not present in both models. uYobs is being plotted instead.')

                except:
                    print(f'ERROR: {models[0]} and {models[1]} have different definitions of uYobs!')

            else:
                print(f'ERROR: variable {var} not found or deprecated in {models[0]} or {models[1]}!')

    for i,m in enumerate(models):
        
        m0 = m.split('_')[0]
        
        if m0 == 'mcmc':
            s = f'_{species}'
        else:
            s = ''

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
                var_plot = ds_all[m][f'Yobs{s}'].values - ds_all[m][f'{var}{s}'].values
            else:
                try:
                    var_plot = ds_all[m][var].values
                except:
                    if var == 'uYmod':
                        var_plot = ds_all[m][f'Yobs{s}'].values - ds_all[m][f'qYmod{s}'].values[:,model_q_indices[m0][0]]
                    elif var == 'uYobs_repeatability':
                        var_plot = ds_all[m][f'uYobs{s}'].values
                    else:
                        continue
            
            if np.abs(np.nanmean(var_plot)) <= 0.01:
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
    
    ax.set_title(model_labels[m])
    ax.set_ylabel(f'{s_data[species]["species_print"]} {site} ({s_data[species]["mf_units_print"]})')
    leg = ax.legend(ncol=2,borderpad=.2,columnspacing=1.0)
    try:
        for l in leg.legend_handles:
            l.set_linewidth(5.0)
    except:
        for l in leg.legendHandles:
            l.set_linewidth(5.0)
    '''
    if int(ds_all[m][time_name1].values[-1].astype('datetime64[M]')-ds_all[m][time_name1].values[0].astype('datetime64[M]')) > 12:
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_locator(YearLocator())
    else:
        ax.xaxis.set_major_locator(MonthLocator())
    '''
    
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    
    if y_lim is None:
        ax.set_ylim([min(min_mf)-(0.02*min(min_mf)),
                                max(max_mf)+(0.05*max(max_mf))])
    else:
        ax.set_ylim(y_lim)
        
    print('NOTE: If all the data is not within axis limits, adjust the set_ylim')
    print('NOTE: If annotations in the histograms are not displaying correctly, adjust annotate_coords.')
    
    return fig

#####################################################################

def plot_stats_mf(pearson,nrmse,species,model_labels,
                  model_colors,
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
        model_labels (dict of str):
            Models and corresponding strings used to describe the model in the 
            plot legend.
        model_colors (dict of str):
            Models and corresponding colours used to plot the model.
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
                ax[0].scatter(i+m*0.2,pearson[site][model],color=model_colors[model][0],marker='x',s=150,label=model_labels[model])
                ax[1].scatter(i+m*0.2,nrmse[site][model],color=model_colors[model][0],marker='x',s=150,label=model_labels[model])
                #ax[2].scatter(i+m*0.2,std[site][model],color=model_colors_stats[model],marker='x',s=150,label=model_labels[model])
                
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
def extract_region_flux(ds_all,m,m0,country,sector='total'):
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
    elif m0 == 'mcmc':
        c_key = 'countrynumber'
        
    #search for existing region names
    try:
        try:
            try:
                if m0 == 'intem' and country == 'BELGIUM':
                    country_search = 'BEL-LUX'
                    print(f'\nNOTE: InTEM does not estimate separate BELGIUM emissions.')
                    print(f'So a population ratio of {bel_pop_r} is being used to scale InTEM\'s {sector} BELGIUM+LUXEMBOURG estimate.\n')
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
        region_flux_sector_posterior = ds_all[m][f'country_flux_{sector}_posterior'].values[:,country_index]*r
        region_flux_sector_prior = ds_all[m][f'country_flux_{sector}_prior'].values[:,country_index]*r
        region_flux_sector_posterior_lower = ds_all[m][f'percentile_country_flux_{sector}_posterior'].values[:,model_q_indices[m0][0],country_index]*r
        region_flux_sector_posterior_upper = ds_all[m][f'percentile_country_flux_{sector}_posterior'].values[:,model_q_indices[m0][1],country_index]*r
        try:
            region_flux_sector_prior_lower = ds_all[m][f'percentile_country_flux_{sector}_prior'].values[:,model_q_indices[m0][0],country_index]*r
            region_flux_sector_prior_upper = ds_all[m][f'percentile_country_flux_{sector}_prior'].values[:,model_q_indices[m0][1],country_index]*r
        except:
            region_flux_sector_prior_lower = None
            region_flux_sector_prior_upper = None
    
    #calculate values for region names that don't exist in the file
    except:
        
        try:
            region_search = regions_dict[country]
            print(f'{country} emissions are not present in {m}. Considering covariance matrix and sum of individual countries: {region_search}.')

            country_list = region_search.split('-')

            if m0 == 'intem':
                c_key = 'countrynumber'
            elif m0 == 'rhime':
                c_key = 'country'
            elif m0 == 'elris':
                c_key = 'country'
            elif m0 == 'mcmc':
                c_key = 'countrynumber'

            country_index_vec = np.zeros(len(ds_all[m][c_key]))
            sigma2_region_flux_sector_prior = 0
            region_flux_sector_posterior = 0
            region_flux_sector_prior = 0

            # Compute sum of prior/posterior emissions and prior uncertainty
            for var in country_list:
                try:
                    country_index = np.where(ds_all[m][c_key].values.astype(str) == var)[0][0]
                    country_index_vec[country_index] = 1

                    region_flux_sector_posterior = region_flux_sector_posterior + ds_all[m][f'country_flux_{sector}_posterior'].values[:,country_index]
                    region_flux_sector_prior     = region_flux_sector_prior + ds_all[m][f'country_flux_{sector}_prior'].values[:,country_index]

                    sigma_country_prior = ds_all[m][f'country_flux_{sector}_prior'].values[:,country_index] - ds_all[m][f'percentile_country_flux_{sector}_prior'].values[:,model_q_indices[m0][0],country_index]
                    sigma2_region_flux_sector_prior = sigma2_region_flux_sector_prior + sigma_country_prior**2

                except:
                    print(f'WARNING: {var} emissions are not present in {m}. This country will be neglected in {country} emissions.')
                    sigma2_region_flux_sector_prior = np.zeros(ds_all[m].time.values.shape[0])
                    
            sigma_region_flux_sector_prior = np.sqrt(sigma2_region_flux_sector_prior)
        
            # Compute posterior uncertainty from covariance matrix
            try:
                sigma2 = np.zeros(np.shape(ds_all[m][f'covariance_country_flux_{sector}_posterior'])[0])

                for i in range(len(sigma2)):
                    sigma2[i] = country_index_vec.dot(ds_all[m][f'covariance_country_flux_{sector}_posterior'].values[i,:,:].dot(country_index_vec))

                sigma_region_flux_sector_posterior = np.sqrt(sigma2)
            except:
                print(f'WARNING: Covariance matrix is not available for {m}. A posteriori uncertainty of {country} emissions will not be plotted.')
                sigma_region_flux_sector_posterior = np.zeros(ds_all[m].time.values.shape[0])
                
            region_time = ds_all[m].time.values
            region_flux_sector_posterior_lower = region_flux_sector_posterior - sigma_region_flux_sector_posterior
            region_flux_sector_posterior_upper = region_flux_sector_posterior + sigma_region_flux_sector_posterior
            region_flux_sector_prior_lower = region_flux_sector_prior - sigma_region_flux_sector_prior
            region_flux_sector_prior_upper = region_flux_sector_prior + sigma_region_flux_sector_prior

        except:
            print(f'ERROR: Either start and end dates are incorrect or there is no {country} {sector} emissions in {m}.')
            print(f'Skipping plotting {m}.')
            
            region_time = None
            region_flux_sector_posterior,region_flux_sector_prior = None,None
            region_flux_sector_posterior_lower,region_flux_sector_posterior_upper = None,None
            region_flux_sector_prior_lower,region_flux_sector_prior_upper = None,None
    
    return (region_time,region_flux_sector_posterior,region_flux_sector_prior,
            region_flux_sector_posterior_lower,region_flux_sector_posterior_upper,
            region_flux_sector_prior_lower,region_flux_sector_prior_upper)
    
#####################################################################
def extract_region_inventory_flux(country,data_dir,species,
                                  inventory_year=None):
    """
    Extracts inventory flux values for regions that exists,
    or calculates total inventory flux values for aggregated regions.
    """
    
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
        inv_c_index = np.where(inv_ds['country'].values == country)[0][0]
        inventory_flux = inv_ds['inventory'].values[:,inv_c_index]/s_data[species]["units_scaling"]["intem"]
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
                    if np.any(np.isnan(inv_c_temp) == True):
                        inv_c_temp = np.zeros(len(inv_ds.time.values))
                        print(f'WARNING: Inventory data for {inv_key[0]} is NaN. Inventory value for {country} will not include {inv_key[0]} contributions.')

                    inv_c_value = inv_c_value + inv_c_temp
                    inventory_flux = inv_c_value/s_data[species]["units_scaling"]["intem"]
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

def plot_country_flux(ds_all,species,plot_regions,model_labels,
                      model_colors,
                      plot_inventory=True,inventory_years=None,
                      data_dir=None,fix_y_axes=False,
                      add_prior_unc=False, set_global_leg=False,
                      country_codes_as_titles=None,plot_separate=True,
                      plot_combined=False,plot_separate_by_year=False,
                      period_override=None):
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
        model_labels (dict of str):
            Models and corresponding strings used to describe the model in the 
            plot legend.
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
        plot_separate (bool):
            If True, plots model results as separate lines.
        plot_combined (bool):
            If True, plots combined average results from all models.
        plot_separate_by_year (bool):
            If True, average model results by year (only meaningful for monthly inversions).
        period_override (list of str, optional):
            Inversion periods to include, to override the standards in species_info.json.
            Must be the same length as models, e.g. ['monthly',None,'yearly']
    Returns:
        fig (figure): 
            A plot per country/region.
    """
    
    # Create annual mean xarrays if needed
    if plot_separate_by_year == True:
        tmp = {m:ds_all[m].copy() for m in ds_all.keys()}
        for m in ds_all.keys():
            if 'elris' in m:
                del tmp[m]['covariance_country_flux_total_posterior']
                
        if period_override is not None: 
            ds_all_p = {m:tmp[m].groupby("time.year").mean().rename({'year':'time'}) if period_override[i] == 'monthly' else tmp[m] for i,m in enumerate(ds_all.keys())}
            for i,m in enumerate(ds_all.keys()):
                if period_override[i] == 'monthly':
                    ds_all_p[m]['time'] = (ds_all_p[m]['time']-1970).astype('datetime64[Y]')
                if 'elris' in m and period_override[i] == 'monthly':
                    ds_all_p[m]['country'] = ds_all_p[m]['country'].isel(time=0).drop('time')
                    ds_all_p[m]['country_fraction'] = ds_all_p[m]['country_fraction'].isel(time=0).drop('time')
                    ds_all_p[m] = ds_all_p[m].assign({'covariance_country_flux_total_posterior':
                                                      ds_all[m]['covariance_country_flux_total_posterior'].groupby("time.year").mean().rename({'year':'time'})})
                    
        elif s_data[species]["period"]=='monthly':
            ds_all_p = {m:tmp[m].groupby("time.year").mean().rename({'year':'time'}) for m in ds_all.keys()}
            for m in ds_all.keys():
                ds_all_p[m]['time'] = (ds_all_p[m]['time']-1970).astype('datetime64[Y]')
                if 'elris' in m:
                    ds_all_p[m]['country'] = ds_all_p[m]['country'].isel(time=0).drop('time')
                    ds_all_p[m]['country_fraction'] = ds_all_p[m]['country_fraction'].isel(time=0).drop('time')
                    ds_all_p[m] = ds_all_p[m].assign({'covariance_country_flux_total_posterior':
                                                      ds_all[m]['covariance_country_flux_total_posterior'].groupby("time.year").mean().rename({'year':'time'})})
        else:
            ds_all_p = ds_all
            
        del tmp
        
            
    else:
        ds_all_p = ds_all
    
    a,b = 0,0
    max_cf = []
    min_x = []
    max_x = []
    period_all = {}

    n_cols = math.ceil(len(plot_regions)/2)
    if n_cols <= 1:
        n_cols = 2
        
    fig,ax = plt.subplots(2,n_cols,figsize=(n_cols*6,8),constrained_layout=True)

    for i,country in enumerate(plot_regions):
        
        if plot_inventory == True:
            
            inv_colours = ['grey','black']
            
            if inventory_years == None:
                search_years = sorted(glob.glob(os.path.join(data_dir,'inventory',f'UNFCCC_inventory_{species}_*.nc')))
                inventory_years = [search_years[-1][-7:-3]]
            
            for y,i_year in enumerate(inventory_years):
            
                inventory_flux,inventory_time = extract_region_inventory_flux(country,data_dir,species,
                                                                              inventory_year=i_year)
                
                if inventory_flux is not None:
                    ax[a,b].bar(inventory_time,inventory_flux,
                                np.timedelta64(340, 'D'),color='white',edgecolor=inv_colours[y],align='edge',
                                label=f'Inventory {i_year}',zorder=0)
        
        post_pdfs = {}
        
        for j,m in enumerate(ds_all.keys()):
            
            m0 = m.split('_')[0]

            # Get inversion period
            if period_override is not None:
                if period_override[i] == 'monthly':
                    period_all[m] = 'monthly'
                elif period_override[i] == 'yearly':
                    period_all[m] = 'yearly'
                else:
                    period_all[m] = s_data[species]["period"]
            else:
                period_all[m] = s_data[species]["period"]
                
            region_time,region_flux_total_posterior,region_flux_total_prior,\
            region_flux_total_posterior_lower,region_flux_total_posterior_upper,\
            region_flux_total_prior_lower,region_flux_total_prior_upper = extract_region_flux(ds_all_p,m,m0,country)
            
            if region_time is not None:
        
                if plot_combined == True:
            
                    if j == 0:
                        all_region_flux_total_posterior = region_flux_total_posterior
                        all_region_flux_total_prior = region_flux_total_prior
                        all_region_flux_total_lower = region_flux_total_posterior_lower
                        all_region_flux_total_upper = region_flux_total_posterior_upper
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
                            
                if plot_separate == True:
                    ax[a,b].plot(region_time,
                                region_flux_total_posterior,
                                label=model_labels[m],color=model_colors[m][0])
                    
                    ax[a,b].plot(region_time,
                                region_flux_total_prior,
                                label=f'{model_labels[m]} prior',color=model_colors[m][0],linestyle='dashed')
                    
                    
                    ax[a,b].fill_between(region_time,
                                        region_flux_total_posterior_lower,
                                        region_flux_total_posterior_upper,
                                        alpha=0.3,color=model_colors[m][0])

                    if add_prior_unc:
                        ax[a,b].fill_between(region_time,
                                            region_flux_total_prior_lower,
                                            region_flux_total_prior_upper,
                                            alpha=0.1,color=model_colors[m][0])
                
                min_x.append(np.min(region_time).astype('datetime64[M]'))
                max_x.append(np.max(region_time).astype('datetime64[M]'))
                max_cf.append(ax[a,b].get_ylim()[1])
                
        if plot_combined == True:
            
            if i == 0:
                print('\nNOTE: This currently assumes that posterior PDFs are Gaussian. The average percentile is used '+
                    'to estimate an approximate standard deviation.\n')
            
            mean_country_flux_total_posterior = np.mean(all_region_flux_total_posterior,axis=0)
            mean_country_flux_total_prior = np.mean(all_region_flux_total_prior,axis=0)
            mean_country_flux_total_lower = np.mean(all_region_flux_total_lower,axis=0)
            mean_country_flux_total_upper = np.mean(all_region_flux_total_upper,axis=0)
            min_country_flux_total_lower = np.min(all_region_flux_total_lower,axis=0)
            max_country_flux_total_upper = np.max(all_region_flux_total_upper,axis=0)
            
            for j,m in enumerate(ds_all.keys()):
                if j == 0:
                    pdf_all = np.array([np.random.choice(post_pdfs[m][t,:],500) for t in range(post_pdfs[m].shape[0])])
                else:
                    pdf_all = np.hstack((pdf_all,
                                        np.array([np.random.choice(post_pdfs[m][t,:],500) for t in range(post_pdfs[m].shape[0])])))
            
            pdf_mean = np.mean(pdf_all,axis=1)
            pdf_std = np.std(pdf_all,axis=1)
                                        
            ax[a,b].plot(region_time.astype('datetime64[ns]'),
                            mean_country_flux_total_posterior,
                            label='Mean posterior',color='black')
            ax[a,b].plot(region_time.astype('datetime64[ns]'),
                                mean_country_flux_total_prior,
                                label='Mean prior',color='black',linestyle='dashed')
            
            ax[a,b].fill_between(region_time.astype('datetime64[ns]'),
                                            min_country_flux_total_lower,
                                            max_country_flux_total_upper,
                                            alpha=0.3,color='black',label='Min/max of post uncertainty')
            
            ax[a,b].plot(region_time.astype('datetime64[ns]'),
                                pdf_mean,
                                label='Mean of sampled post PDFs',color='dodgerblue')
            ax[a,b].fill_between(region_time.astype('datetime64[ns]'),
                                            pdf_mean-pdf_std,
                                            pdf_mean+pdf_std,
                                            alpha=0.3,color='dodgerblue',label='Std dev of sampled post PDFs')
            
            ax[a,b].fill_between(region_time.astype('datetime64[ns]'),
                                            mean_country_flux_total_lower,
                                            mean_country_flux_total_upper,
                                            alpha=0.3,color='yellow',label='Mean of post uncertainty')
                                        
        #format each subplot
        
        ax[a,b].set_ylabel(f'{s_data[species]["species_print"]} ({s_data[species]["units_print"]}g y$^{{-1}}$)')
        ax[a,b].set_xlim([np.min(min_x)-np.timedelta64(1,'M'),
                        np.max(max_x)+np.timedelta64(1,'M')])

        ncol = 2
        if set_global_leg == False:
            leg = ax[a,b].legend(ncol=ncol,borderpad=.4,columnspacing=1.0,fontsize=10)
            if plot_inventory == True:
                for l in leg.legendHandles[:-1]:
                    l.set_linewidth(3.0)
            else:
                for l in leg.legendHandles:
                    l.set_linewidth(3.0)
        
        if country_codes_as_titles == True:
            try:
                ax[a,b].set_title(f'{country}\n{regions_dict[country]}')
            except:
                ax[a,b].set_title(f'{country}')
        else:        
            ax[a,b].set_title(f'{country}')
        ax[a,b].grid(visible=True,which='major',alpha=0.4)
        ax[a,b].xaxis.set_minor_locator(MonthLocator())
        ax[a,b].xaxis.set_minor_formatter(NullFormatter())
        ax[a,b].xaxis.set_major_locator(YearLocator())
        
        #increase row and column counts
        if (b - (n_cols-1)) == 0:
            b = 0
            a += 1
        else:
            b += 1

    if set_global_leg:
        handles, labels = ax[0,0].get_legend_handles_labels()
        ncol=0   
        if (plot_separate or plot_separate_by_year):
            ncol=len(ds_all.keys())
        if plot_combined:
            ncol=ncol+3
        if plot_inventory == True:
            ncol=ncol+1
        leg = fig.legend(handles, labels, loc='upper center',ncol=ncol,borderpad=.4,columnspacing=1.0,fontsize=10,bbox_to_anchor=(0.5, 1.07))
        if plot_inventory == True:
            for l in leg.legendHandles:
                l.set_linewidth(3.0)
        else:
            for l in leg.legendHandles:
                l.set_linewidth(3.0)

    for a in range(2):
        for b in range(n_cols):
            if fix_y_axes == True:
                ax[a,b].set_ylim([0,(np.max(max_cf)+(0.1*np.max(max_cf)))])  
            elif type(fix_y_axes) == list:
                ax[a,b].set_ylim(fix_y_axes)
            
            elif fix_y_axes == False:
                ax[a,b].set_ylim(bottom=0)  

    print('NOTE: If all the data is not within axis limits, adjust the set_ylim parameter')
    
    return fig

#######################################################################################

def plot_country_flux_sectors(ds_all,species,sectors,plot_region,model_labels,
                      model_colors,
                      plot_inventory=True,inventory_years=None,
                      data_dir=None,fix_y_axes=False,
                      add_prior_unc=False, set_global_leg=False,
                      country_codes_as_titles=None,plot_separate=True,
                      plot_combined=False,plot_separate_by_year=False,
                      period_override=None):
    """
    Timeseries plot of prior and posterior region fluxes, from list of sectors.
    
    Args:
        ds_all (dictionary of datasets):
            xarray datasets of fluxes, scaled and sliced between 
            chosen dates.
        species (str): 
            Gas species, e.g. 'ch4'.
        plot_region (str):
            Country or region to plot, e.g. 'UK' or 'SWITZERLAND'
        sectors (list of str):
            List of sectors to plot, e.g. ['FF','nonFF','total']
        model_labels (dict of str):
            Models and corresponding strings used to describe the model in the 
            plot legend.
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
        plot_separate (bool):
            If True, plots model results as separate lines.
        plot_combined (bool):
            If True, plots combined average results from all models.
        plot_separate_by_year (bool):
            If True, average model results by year (only meaningful for monthly inversions).
        period_override (list of str, optional):
            Inversion periods to include, to override the standards in species_info.json.
            Must be the same length as models, e.g. ['monthly',None,'yearly']
    Returns:
        fig (figure): 
            A plot per country/region.
    """
        
    # Create annual mean xarrays if needed
    if plot_separate_by_year == True:
        tmp = {m:ds_all[m].copy() for m in ds_all.keys()}
        for m in ds_all.keys():
            if 'elris' in m:
                for sector in sectors:
                    del tmp[m][f'covariance_country_flux_{sector}_posterior']
                
        if period_override is not None: 
            ds_all_p = {m:tmp[m].groupby("time.year").mean().rename({'year':'time'}) if period_override[i] == 'monthly' else tmp[m] for i,m in enumerate(ds_all.keys())}
            for i,m in enumerate(ds_all.keys()):
                if period_override[i] == 'monthly':
                    ds_all_p[m]['time'] = (ds_all_p[m]['time']-1970).astype('datetime64[Y]')
                if 'elris' in m and period_override[i] == 'monthly':
                    ds_all_p[m]['country'] = ds_all_p[m]['country'].isel(time=0).drop('time')
                    ds_all_p[m]['country_fraction'] = ds_all_p[m]['country_fraction'].isel(time=0).drop('time')
                    for sector in sectors:
                        ds_all_p[m] = ds_all_p[m].assign({f'covariance_country_flux_{sector}_posterior':
                                                        ds_all[m][f'covariance_country_flux_{sector}_posterior'].groupby("time.year").mean().rename({'year':'time'})})
                        #TODO: continue from here:
        elif s_data[species]["period"]=='monthly':
            ds_all_p = {m:tmp[m].groupby("time.year").mean().rename({'year':'time'}) for m in ds_all.keys()}
            for m in ds_all.keys():
                ds_all_p[m]['time'] = (ds_all_p[m]['time']-1970).astype('datetime64[Y]')
                if 'elris' in m:
                    ds_all_p[m]['country'] = ds_all_p[m]['country'].isel(time=0).drop('time')
                    ds_all_p[m]['country_fraction'] = ds_all_p[m]['country_fraction'].isel(time=0).drop('time')
                    for sector in sectors:
                        ds_all_p[m] = ds_all_p[m].assign({f'covariance_country_flux_{sectpr}_posterior':
                                                        ds_all[m][f'covariance_country_flux_{sector}_posterior'].groupby("time.year").mean().rename({'year':'time'})})
        else:
            ds_all_p = ds_all
            
        del tmp
        
            
    else:
        ds_all_p = ds_all

    max_cf = []
    min_x = []
    max_x = []
    period_all = {}

    n_rows = len(sectors)
        
    fig,ax = plt.subplots(n_rows,1,figsize=(12,n_rows*3),constrained_layout=True)

    for i,sector in enumerate(sectors):
        '''
        #TODO: add split sector inventory data files and edit to plot this by sector
        if plot_inventory == True:
            
            inv_colours = ['grey','black']
            
            if inventory_years == None:
                search_years = sorted(glob.glob(os.path.join(data_dir,'inventory',f'UNFCCC_inventory_{species}_*.nc')))
                inventory_years = [search_years[-1][-7:-3]]
            
            for y,i_year in enumerate(inventory_years):
            
                inventory_flux,inventory_time = extract_region_inventory_flux(plot_region,data_dir,species,
                                                                              inventory_year=i_year)
                
                if inventory_flux is not None:
                    ax[a,b].bar(inventory_time,inventory_flux,
                                np.timedelta64(340, 'D'),color='white',edgecolor=inv_colours[y],align='edge',
                                label=f'Inventory {i_year}',zorder=0)
        '''
        post_pdfs = {}
        prior_plotted = False
        
        for j,m in enumerate(ds_all.keys()):
            
            m0 = m.split('_')[0]

            # Get inversion period
            if period_override is not None:
                if period_override[i] == 'monthly':
                    period_all[m] = 'monthly'
                elif period_override[i] == 'yearly':
                    period_all[m] = 'yearly'
                else:
                    period_all[m] = s_data[species]["period"]
            else:
                period_all[m] = s_data[species]["period"]
                
            region_time,region_flux_sector_posterior,region_flux_sector_prior,\
            region_flux_sector_posterior_lower,region_flux_sector_posterior_upper,\
            region_flux_sector_prior_lower,region_flux_sector_prior_upper = extract_region_flux(ds_all_p,m,m0,plot_region,sector)
            
            if region_time is not None:
        
                if plot_combined == True:
            
                    if j == 0:
                        all_region_flux_sector_posterior = region_flux_sector_posterior
                        all_region_flux_sector_prior = region_flux_sector_prior
                        all_region_flux_sector_lower = region_flux_sector_posterior_lower
                        all_region_flux_sector_upper = region_flux_sector_posterior_upper
                    else:
                        all_region_flux_sector_posterior = np.vstack((all_region_flux_sector_posterior,
                                                                    region_flux_sector_posterior))
                        all_region_flux_sector_prior = np.vstack((all_region_flux_sector_prior,
                                                                region_flux_sector_prior))
                        all_region_flux_sector_lower = np.vstack((all_region_flux_sector_lower,
                                                                region_flux_sector_posterior_lower))
                        all_region_flux_sector_upper = np.vstack((all_region_flux_sector_upper,
                                                                region_flux_sector_posterior_upper))
                        
                    post_pdfs[m] = np.array([np.random.default_rng().normal(loc=region_flux_sector_posterior[t],
                                                                            scale=np.mean(np.array([region_flux_sector_posterior[t]-region_flux_sector_posterior_lower[t],
                                                                                                    region_flux_sector_posterior_upper[t]-region_flux_sector_posterior[t]])),
                                                                            size=1000) for t in range(region_time.shape[0])])
                            
                if plot_separate == True:
                    
                    if prior_plotted == False:
                        ax[i].plot(region_time,
                                    region_flux_sector_prior,
                                    label=f'Prior mean',color='grey',linestyle='dashed')
                        prior_plotted = True
                    
                    ax[i].plot(region_time,
                                region_flux_sector_posterior,
                                label=model_labels[m],color=model_colors[m][0])

                    ax[i].fill_between(region_time,
                                        region_flux_sector_posterior_lower,
                                        region_flux_sector_posterior_upper,
                                        alpha=0.3,color=model_colors[m][0])

                    if add_prior_unc and region_flux_sector_prior_lower is not None:
                        ax[i].fill_between(region_time,
                                            region_flux_sector_prior_lower,
                                            region_flux_sector_prior_upper,
                                            alpha=0.1,color=model_colors[m][0])
                
                min_x.append(np.min(region_time).astype('datetime64[M]'))
                max_x.append(np.max(region_time).astype('datetime64[M]'))
                max_cf.append(ax[i].get_ylim()[1])
                
        if plot_combined == True:
            
            if i == 0:
                print('\nNOTE: This currently assumes that posterior PDFs are Gaussian. The average percentile is used '+
                    'to estimate an approximate standard deviation.\n')
            
            mean_country_flux_sector_posterior = np.mean(all_region_flux_sector_posterior,axis=0)
            mean_country_flux_sector_prior = np.mean(all_region_flux_sector_prior,axis=0)
            mean_country_flux_sector_lower = np.mean(all_region_flux_sector_lower,axis=0)
            mean_country_flux_sector_upper = np.mean(all_region_flux_sector_upper,axis=0)
            min_country_flux_sector_lower = np.min(all_region_flux_sector_lower,axis=0)
            max_country_flux_sector_upper = np.max(all_region_flux_sector_upper,axis=0)
            
            for j,m in enumerate(ds_all.keys()):
                if j == 0:
                    pdf_all = np.array([np.random.choice(post_pdfs[m][t,:],500) for t in range(post_pdfs[m].shape[0])])
                else:
                    pdf_all = np.hstack((pdf_all,
                                        np.array([np.random.choice(post_pdfs[m][t,:],500) for t in range(post_pdfs[m].shape[0])])))
            
            pdf_mean = np.mean(pdf_all,axis=1)
            pdf_std = np.std(pdf_all,axis=1)
                                        
            ax[i].plot(region_time.astype('datetime64[ns]'),
                            mean_country_flux_sector_posterior,
                            label='Mean posterior',color='black')
            ax[i].plot(region_time.astype('datetime64[ns]'),
                                mean_country_flux_sector_prior,
                                label='Mean prior',color='black',linestyle='dashed')
            
            ax[i].fill_between(region_time.astype('datetime64[ns]'),
                                            min_country_flux_sector_lower,
                                            max_country_flux_sector_upper,
                                            alpha=0.3,color='black',label='Min/max of post uncertainty')
            
            ax[i].plot(region_time.astype('datetime64[ns]'),
                                pdf_mean,
                                label='Mean of sampled post PDFs',color='dodgerblue')
            ax[i].fill_between(region_time.astype('datetime64[ns]'),
                                            pdf_mean-pdf_std,
                                            pdf_mean+pdf_std,
                                            alpha=0.3,color='dodgerblue',label='Std dev of sampled post PDFs')
            
            ax[i].fill_between(region_time.astype('datetime64[ns]'),
                                            mean_country_flux_sector_lower,
                                            mean_country_flux_sector_upper,
                                            alpha=0.3,color='yellow',label='Mean of post uncertainty')
        '''
        ncol = 2
        if set_global_leg == False:
            leg = ax[i].legend(ncol=ncol,borderpad=.4,columnspacing=1.0,fontsize=10)
            if plot_inventory == True:
                for l in leg.legendHandles[:-1]:
                    l.set_linewidth(3.0)
            else:
                for l in leg.legendHandles:
                    l.set_linewidth(3.0)
        else:        
            ax[i].set_title(f'{plot_region}'
        '''
        ax[i].grid(visible=True,which='major',alpha=0.4)
        #ax[i].xaxis.set_minor_locator(MonthLocator())
        #ax[i].xaxis.set_minor_formatter(NullFormatter())
        #ax[i].xaxis.set_major_locator(YearLocator())
        
        #locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        #formatter = mdates.ConciseDateFormatter(locator)
        #ax[i].xaxis.set_major_locator(locator)
        #ax[i].xaxis.set_major_formatter(formatter)
        #ax[i].xaxis.set_minor_locator(MonthLocator())
        #ax[i].xaxis.set_minor_formatter(NullFormatter())
    '''
    if set_global_leg:
        handles, labels = ax[0].get_legend_handles_labels()
        ncol=0   
        if (plot_separate or plot_separate_by_year):
            ncol=len(ds_all.keys())
        if plot_combined:
            ncol=ncol+3
        if plot_inventory == True:
            ncol=ncol+1
        leg = fig.legend(handles, labels, loc='upper center',ncol=ncol,borderpad=.4,columnspacing=1.0,fontsize=10,bbox_to_anchor=(0.5, 1.07))
        if plot_inventory == True:
            for l in leg.legendHandles:
                l.set_linewidth(3.0)
        else:
            for l in leg.legendHandles:
                l.set_linewidth(3.0)
    '''
    ncol = len(list(ds_all.keys()))+1
    handles, labels = ax[-1].get_legend_handles_labels()
    ax[0].legend(handles, labels, loc='upper right',ncol=ncol,borderpad=.4,columnspacing=1.0,fontsize=10,bbox_to_anchor=(1.0,1.3))
    
    for i,sector in enumerate(sectors):
        
        ax[i].set_ylabel(f'{plot_region} {sector}\n{s_data[species]["species_print"]} ({s_data[species]["units_print"]}g y$^{{-1}}$)')
        ax[i].set_xlim([np.min(min_x)-np.timedelta64(1,'M'),
                        np.max(max_x)+np.timedelta64(1,'M')])
        
        if fix_y_axes == True:
            ax[i].set_ylim([0,(np.max(max_cf)+(0.1*np.max(max_cf)))])  
        elif type(fix_y_axes) == list:
            ax[i].set_ylim(fix_y_axes)
        
        elif fix_y_axes == False:
            ax[i].set_ylim(bottom=0)  

    print('NOTE: If all the data is not within axis limits, adjust the set_ylim parameter')
    
    return fig

#####################################################################

def plot_spatial_flux(ds_all,species,plot_area,model_labels,cmap=None,
                      cmap_diff=None,c_border=None,period_override=None,
                      plot_site_locations=False,plot_point_markers=None,
                      season=None,sectors=None):
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
        model_labels (dict of str):
            Models and corresponding strings used to describe the model in the 
            plot legend.
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
        sectors (list of list of str):
            Sectors to sum for each model, e.g. [['FF','nonFF'],['FF','nonFF]]
            
    Returns:
        fig (figure): 
            A plot of spatial flux posterior and prior mean/mode and a plot 
            of the absolute difference between these, for each model.
    """
    
    if sectors == None:
        sectors = ['total' for i in range(len(models))]
    
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
        
        #try:
        
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
            
            for s,sector_name in enumerate(sectors[i]):
                if s == 0:
                    flux_total_prior = np.mean(ds_all[m][f'flux_{sector_name}_prior'].values,axis=0)
                    flux_total_posterior = np.mean(ds_all[m][f'flux_{sector_name}_posterior'].values,axis=0)
                    
                else:
                    flux_total_prior += np.mean(ds_all[m][f'flux_{sector_name}_prior'].values,axis=0)
                    flux_total_posterior += np.mean(ds_all[m][f'flux_{sector_name}_posterior'].values,axis=0)
                    
        else:
            
            for s,sector_name in enumerate(sectors[i]):
                if s == 0:
                    flux_total_prior = ds_all[m]['flux_total_prior'].groupby("time.season").mean().sel(season=season)
                    flux_total_posterior = ds_all[m]['flux_total_posterior'].groupby("time.season").mean().sel(season=season).values
                    
                else:
                    flux_total_prior += ds_all[m]['flux_total_prior'].groupby("time.season").mean().sel(season=season)
                    flux_total_posterior += ds_all[m]['flux_total_posterior'].groupby("time.season").mean().sel(season=season).values
                    
            time_out = f'{season} of {time_out}'
                    
        flux_diff = flux_total_posterior - flux_total_prior
        flux_diff[np.where(flux_diff) == np.nan] = 0.
        
        ax0.pcolormesh(lon,lat,flux_total_prior,
                        cmap=cmap,vmin=s_data[species]['fluxlim'][0],vmax=s_data[species]['fluxlim'][1],shading='nearest')

        ax1.pcolormesh(lon,lat,flux_total_posterior,
                        cmap=cmap,vmin=s_data[species]['fluxlim'][0],vmax=s_data[species]['fluxlim'][1],shading='nearest')
            
        ax0.set_title(f'{model_labels[m]}: prior')
        ax1.set_title(f'{model_labels[m]}: posterior')

        ax2.pcolormesh(lon,lat,flux_diff,
                        cmap=cmap_diff,vmin=s_data[species]['difflim'][0],vmax=s_data[species]['difflim'][1],shading='nearest')

        ax2.set_title(f'{model_labels[m]}: posterior - prior')

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
                        
                
        #except:
        #    print(f'ERROR: Either start and end dates are incorrect or there are missing data for model {m}.')
        #    print(f'Skipping plotting {m}.')
            
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

#####################################################################

def plot_spatial_flux_comparison(ds_all,species,plot_area,model_labels,
                                 cmap=None,cmap_diff=None,c_border=None,period_override=None,
                                 plot_site_locations=False,plot_point_markers=None,
                                 sectors=None):
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
        model_labels (dict of str):
            Models and corresponding strings used to describe the model in the 
            plot legend.
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
    Returns:
        fig (figure): 
            A plot of spatial flux posterior from two models a plot 
            of the absolute difference between these.
    """
    
    if sectors == None:
        sectors = ['total' for i in range(len(models))]
    
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
    flux_total_posterior = {}
    
    for i,m in enumerate(ds_all.keys()):
        
        lon = ds_all[m].longitude.values + (ds_all[m].longitude.values[1]-ds_all[m].longitude.values[1])/2
        lat = ds_all[m].latitude.values + (ds_all[m].latitude.values[1]-ds_all[m].latitude.values[1])/2
        
        all_keys.append(m)
        m0 = m.split('_')[0]
        
        for s,sector_name in enumerate(sectors[i]):
            if s == 0:
                flux_total_posterior[m] = np.mean(ds_all[m][f'flux_{sector_name}_posterior'].values,axis=0)
                
            else:
                flux_total_posterior[m] += np.mean(ds_all[m][f'flux_{sector_name}_posterior'].values,axis=0)
        
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
        
            ax[0].pcolormesh(lon,lat,flux_total_posterior[m],cmap=cmap,
                            vmin=s_data[species]['fluxlim'][0],vmax=s_data[species]['fluxlim'][1],shading='nearest',
                            )

            ax[0].set_title(f'{model_labels[m]}\nPosterior mean')
            
        elif i == 1:
            
            ax[1].pcolormesh(lon,lat,flux_total_posterior[m],
                            vmin=s_data[species]['fluxlim'][0],vmax=s_data[species]['fluxlim'][1],shading='nearest')

            ax[1].set_title(f'{model_labels[m]}\nPosterior mean')
            
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
    
    flux_diff = flux_total_posterior[all_keys[1]] - flux_total_posterior[all_keys[0]]
    flux_diff[np.where(flux_diff) == np.nan] = 0.
    
    ax[2].pcolormesh(lon,lat,flux_diff,
                    cmap=cmap_diff,vmin=s_data[species]['difflim'][0],vmax=s_data[species]['difflim'][1],shading='nearest')

    ax[2].set_title(f'{model_labels[all_keys[1]]} - {model_labels[all_keys[0]]}\nAbsolute difference')

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

    color_bar2 = fig.colorbar(cbar,orientation='horizontal',cmap=cmap,extend='max',ax=ax[0],shrink=0.9,pad=0.01)
    color_bar2.set_label(f'{s_data[species]["species_print"]}\n{time_out}\n(mol m$^{{-2}}$ s$^{{-1}}$)')

    color_bar2 = fig.colorbar(cbar,orientation='horizontal',cmap=cmap,extend='max',ax=ax[1],shrink=0.9,pad=0.01)
    color_bar2.set_label(f'{s_data[species]["species_print"]}\n{time_out}\n(mol m$^{{-2}}$ s$^{{-1}}$)')

    #difference colorbar
    levels_diff = np.linspace(s_data[species]['difflim'][0],s_data[species]['difflim'][1])
    cbar_diff = plt.cm.ScalarMappable(cmap=cmap_diff)
    cbar_diff.set_array(levels_diff)
    cbar_diff.set_clim(s_data[species]['difflim'])

    color_bar3 = fig.colorbar(cbar_diff,orientation='horizontal',extend='both',ax=ax[2],shrink=0.9,pad=0.01)
    color_bar3.set_label(f'{s_data[species]["species_print"]}\n{time_out}\n(mol m$^{{-2}}$ s$^{{-1}}$)')
    
    return fig

#####################################################################

def plot_spatial_flux_comparison_sectors(ds_all,species,plot_area,model_labels,
                                 cmap=None,cmap_diff=None,c_border=None,period_override=None,
                                 plot_site_locations=False,plot_point_markers=None,
                                 sectors=None):
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
        model_labels (dict of str):
            Models and corresponding strings used to describe the model in the 
            plot legend.
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
        sectors (list of str):
            List of sectors to separately compare fluxes for.
    Returns:
        fig (figure): 
            A plot of spatial flux posterior from two models a plot 
            of the absolute difference between these.
    """
    
    if sectors == None:
        sectors = ['total' for i in range(len(models))]
    
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

    fig,ax = plt.subplots(len(sectors[0]),3,constrained_layout=True,figsize=(n_cols*5,9),
                   subplot_kw={'projection':cartopy.crs.PlateCarree()})
    
    for i in range(len(sectors[0])):
        for j in range(3):
            if j == 2:
                border_color = 'dimgrey'
            else:
                border_color = c_border
            ax[i,j].add_feature(cartopy.feature.BORDERS,edgecolor=border_color,linewidth=1.)
            ax[i,j].coastlines(resolution='50m',color=border_color,linewidth=1.)
            ax[i,j].set_extent(region_limits[plot_area])

    all_keys = []
    flux_total_posterior = {}
    
    for s,sector in enumerate(sectors[0]):
    
        for i,m in enumerate(ds_all.keys()):
            
            lon = ds_all[m].longitude.values + (ds_all[m].longitude.values[1]-ds_all[m].longitude.values[1])/2
            lat = ds_all[m].latitude.values + (ds_all[m].latitude.values[1]-ds_all[m].latitude.values[1])/2
            
            if s == 0:
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
            
                ax[s,0].pcolormesh(lon,lat,np.mean(ds_all[m][f'flux_{sector}_posterior'].values,axis=0),cmap=cmap,
                                vmin=s_data[species]['fluxlim'][0],vmax=s_data[species]['fluxlim'][1],shading='nearest',
                                )

                ax[s,0].set_title(f'{model_labels[m]}\n{sector}')
                
            elif i == 1:
                
                ax[s,1].pcolormesh(lon,lat,np.mean(ds_all[m][f'flux_{sector}_posterior'].values,axis=0),
                                vmin=s_data[species]['fluxlim'][0],vmax=s_data[species]['fluxlim'][1],shading='nearest')

                ax[s,1].set_title(f'{model_labels[m]}\n{sector}')
                
            if plot_site_locations == True:
                if sites_info[m] is not None:
                    for s in sites_info[m]:
                        ax[s,0].scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='white',
                                    edgecolor='none',marker='o',s=30,zorder=2,alpha=0.5)
                        ax[s,1].scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='white',
                                    edgecolor='none',marker='o',s=30,zorder=2,alpha=0.5)
                        ax[s,2].scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='white',
                                    edgecolor='none',marker='o',s=30,zorder=2,alpha=0.5)
                        ax[s,0].scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='none',
                                    edgecolor='black',marker='o',s=30,zorder=2)
                        ax[s,1].scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='none',
                                    edgecolor='black',marker='o',s=30,zorder=2)
                        ax[s,2].scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='none',
                                    edgecolor='black',marker='o',s=30,zorder=2)
    
        flux_diff = np.mean(ds_all[all_keys[1]][f'flux_{sector}_posterior'].values,axis=0) - np.mean(ds_all[all_keys[0]][f'flux_{sector}_posterior'].values,axis=0)
        flux_diff[np.where(flux_diff) == np.nan] = 0.
    
        ax[s,2].pcolormesh(lon,lat,flux_diff,
                        cmap=cmap_diff,vmin=s_data[species]['difflim'][0],vmax=s_data[species]['difflim'][1],shading='nearest')

        ax[s,2].set_title(f'{model_labels[all_keys[1]]} - {model_labels[all_keys[0]]}\n{sector}')

        if plot_point_markers is not None:
            print(f'\nPlotting markers for: {plot_point_markers}')
            print(f'Edit lines below line {inspect.getframeinfo(inspect.currentframe()).lineno} to change marker colour')
            for p in plot_point_markers:
                if type(p) == list:
                    ax[s,0].scatter(p[0],p[1],color='black',marker='o',s=5,zorder=2)
                    ax[s,1].scatter(p[0],p[1],color='black',marker='o',s=5,zorder=2)
                    ax[s,2].scatter(p[0],p[1],color='black',marker='o',s=5,zorder=2)
                elif type(p) == str:
                    if p not in point_source_dict.keys():
                        print(f'{p} is not specified in point_source_dict, edit this to add a lat/lon location.')
                    else:
                        ax[s,0].scatter(point_source_dict[p][0],point_source_dict[p][1],color='black',marker='o',s=5,zorder=2)
                        ax[s,1].scatter(point_source_dict[p][0],point_source_dict[p][1],color='black',marker='o',s=5,zorder=2)
                        ax[s,2].scatter(point_source_dict[p][0],point_source_dict[p][1],color='black',marker='o',s=5,zorder=2)
                        

    #flux colorbar
    levels = np.linspace(s_data[species]['fluxlim'][0],s_data[species]['fluxlim'][1])
    cbar = plt.cm.ScalarMappable(cmap=cmap)
    cbar.set_array(levels)
    cbar.set_clim(s_data[species]['fluxlim'])

    color_bar2 = fig.colorbar(cbar,orientation='horizontal',cmap=cmap,extend='max',ax=ax[:,0],shrink=0.9,pad=0.01)
    color_bar2.set_label(f'{time_out}\nPosterior mean {s_data[species]["species_print"]} (mol m$^{{-2}}$ s$^{{-1}}$)')

    color_bar2 = fig.colorbar(cbar,orientation='horizontal',cmap=cmap,extend='max',ax=ax[:,1],shrink=0.9,pad=0.01)
    color_bar2.set_label(f'{time_out}\nPosterior mean {s_data[species]["species_print"]} (mol m$^{{-2}}$ s$^{{-1}}$)')

    #difference colorbar
    levels_diff = np.linspace(s_data[species]['difflim'][0],s_data[species]['difflim'][1])
    cbar_diff = plt.cm.ScalarMappable(cmap=cmap_diff)
    cbar_diff.set_array(levels_diff)
    cbar_diff.set_clim(s_data[species]['difflim'])

    color_bar3 = fig.colorbar(cbar_diff,orientation='horizontal',extend='both',ax=ax[:,2],shrink=0.9,pad=0.01)
    color_bar3.set_label(f'{time_out}\nAbsolute difference {s_data[species]["species_print"]} (mol m$^{{-2}}$ s$^{{-1}}$)')
    
    return fig

def plot_spatial_flux_per_timestamp(ds_all,species,plot_area,model_labels,end_date,
                                    cmap='viridis',c_border='floralwhite',
                                    var='flux_total_posterior',
                                    chop_by='year',dt=1,period_override=None,
                                    plot_site_locations=False,plot_point_markers=False):
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
        model_labels (dict of str):
            Models and corresponding strings used to describe the model in the
            plot legend.
        end_date (str):
            End date of sliced data, e.g. '2022-01-01' would include all
            data up to 2021-12-31.
        cmap (str):
            Colour map for flux plots.
        c_border (str):
            Colour for flux plot country borders.
        var (str):
            Variable to be plotted; options for 'flux_total_prior',
            'flux_total_posterior', 'posterior_prior_diff'
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
    Returns:
        fig (figure):
            A plot of spatial flux of the variable specified in var
            averaged over the number of time steps specified in dt.
            
    NOTE: HAS NOT BEEN EDITED TO WORK WITH MULTI-SECTOR MCMC
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

    var_labels = {'flux_total_prior':'Prior mean',
                  'flux_total_posterior':'Posterior mean',
                  'posterior_prior_diff':'Posterior-prior'}

    region_limits = {'UK':[-12,4,49,62],   #min_lon, max_lon, min_lat, max_lat
                    'FRANCE':[-6,9,42,52],
                    'GERMANY':[2,18,45,60],
                    'ITALY':[6,19,36,48],
                    'SWITZERLAND':[5.5,11,45,49],
                    'BENELUX':[1,9,48,55],
                    'NWEU':[-11,11,45,62],
                    'CWEU':[-12,27,37,66],
                    'EUROPE':[-98,40,10,80]}

    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    # Define variable specific settings
    if var == 'posterior_prior_diff':
        lim = s_data[species]['difflim']
        extend ='both'
    else:
        lim = s_data[species]['fluxlim']
        extend = 'max'

    # Figure size and averaging period
    n_lines = len(ds_all.keys())
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
            start_print[m] = to_datetime(t0_date[m]).strftime('%d/%m/%Y')
            end_print[m]   = (to_datetime(t1_date[m]) - np.timedelta64(1,'D')).strftime('%d/%m/%Y')

    else:
        # NOTE: It will only work properly if the data is complete between start_date and end_date
        nt = np.zeros(n_lines)
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

    # Plot fields
    for i in range(n_cols):
        for j,m in enumerate(ds_all.keys()):

            lon = ds_all[m].longitude.values
            lat = ds_all[m].latitude.values

            m0 = m.split('_')[0]

            # Compute averaged quantities
            if chop_by == 'season':
                if var == 'posterior_prior_diff':
                    var_plot = np.mean(ds_all[m]['flux_total_posterior'][indexes[m][i],:,:],axis=0) - np.mean(ds_all[m]['flux_total_prior'][indexes[m][i],:,:],axis=0)
                    var_plot[np.where(var_plot) == np.nan] = 0.
                else:
                    var_plot = np.mean(ds_all[m][var][indexes[m][i],:,:],axis=0)

                # Define string for caption
                if len(dt[i]) == 1:
                    time_out = (f'{start_print[m][i]}')
                else:
                    time_out = (f'{start_print[m][i]} - {end_print[m][i]}')

            else:
                if var == 'posterior_prior_diff':
                    slice_apost   = ds_all[m]['flux_total_posterior'].sel(time=slice(t0_date[m][i],t1_date[m][i]))
                    slice_apriori = ds_all[m]['flux_total_prior'].sel(time=slice(t0_date[m][i],t1_date[m][i]))
                    var_plot      = np.mean(slice_apost,axis=0) - np.mean(slice_apriori,axis=0)
                    var_plot[np.where(var_plot) == np.nan] = 0.
                else:
                    var_plot = np.mean(ds_all[m][var].sel(time=slice(t0_date[m][i],t1_date[m][i])),axis=0)

                # Define string for caption
                if dt == 1:
                    time_out = (f'{start_print[m][i]}')
                else:
                    time_out = (f'{start_print[m][i]} - {end_print[m][i]}')

            # Make plot
            if n_cols == 1 and n_lines == 1:
                ax.pcolormesh(lon,lat,var_plot,cmap=cmap,vmin=lim[0],vmax=lim[1],shading='nearest')
                ax.set_title(f'{model_labels[m]}\n{time_out}')
                ax_var = ax
            else:
                if n_lines == 1:
                    ax_var = ax[i] 
                elif n_cols == 1:
                    ax_var = ax[j]
                else:
                    ax_var = ax[j,i]

                ax_var.pcolormesh(lon,lat,var_plot,cmap=cmap,vmin=lim[0],vmax=lim[1],shading='nearest')
                ax_var.set_title(f'{time_out}')
                if i == 0:
                    if '\n' in model_labels[m]:
                        ax_var.text(-0.14, 0.25, f'{model_labels[m]}', size=14, transform=ax_var.transAxes, rotation=90)
                    else:
                        ax_var.text(-0.07, 0.25, f'{model_labels[m]}', size=14, transform=ax_var.transAxes, rotation=90)
                
            # Add site location
            if plot_site_locations == True:
                if sites_info[m] is not None:
                    for s in sites_info[m]:
                        ax_var.scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='white',
                                        edgecolor='none',marker='o',s=30,zorder=2,alpha=0.5)
                        ax_var.scatter(sites_info[m][s]['longitude'],sites_info[m][s]['latitude'],color='none',
                                    edgecolor='black',marker='o',s=30,zorder=2)
                
            # Add markers at specific locations
            if plot_point_markers is not None:
                if i == 0:
                    print(f'\nPlotting markers for: {plot_point_markers}')
                    print(f'Edit lines below line {inspect.getframeinfo(inspect.currentframe()).lineno} to change marker colour')
                for p in plot_point_markers:
                    if type(p) == list:
                        ax_var.scatter(p[0],p[1],color='black',marker='o',s=2,zorder=2)
                    elif type(p) == str:
                        if p not in point_source_dict.keys():
                            print(f'{p} is not specified in point_source_dict, edit this to add a lat/lon location.')
                        else:
                            ax_var.scatter(point_source_dict[p][0],point_source_dict[p][1],color='black',marker='o',s=2,zorder=2)

            #except:
            #    print(f'ERROR: Either start and end dates are incorrect or there is no model output from {m}.')
            #    print(f'Skipping plotting {m}.')

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

    color_bar.set_label(f'{var_labels[var]} {s_data[species]["species_print"]} (mol m$^{{-2}}$ s$^{{-1}}$)')
    fig.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05, wspace=0.04, hspace=0.12)

    return fig

#####################################################################

def plot_sites_timeseries(ds_all,var,start_date,end_date,model_labels,model_colors):
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
        model_labels (dict of str):
            Models and corresponding strings used to describe the model in the 
            plot legend.
        model_colors (dict of str):
            Models and corresponding colours used to plot the model.
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
                               data,c=model_colors[m][0],s=20,label=model_labels[m])
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