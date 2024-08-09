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

model_colors = {'intem':[['darkslateblue','dodgerblue'],
                         ['black','grey']],
                'rhime':[['darkgreen','green']],
                'elris':[['purple','mediumpurple']]}

model_q_indices = {'intem':[0,1],
                   'rhime':[0,1],
                   'elris':[0,1],
                   'mcmc':[0,1]}

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
            #try:
            site_index = np.where(ds_all[m][f'sitenames{s}'].astype(str) == site)[0][0]
            ds_all[m] = ds_all[m].sel(**{f'time{s}':slice(start_date,end_date),
                                        f'nsite{s}':site_index})
            #except:
            #    ds_all[m] = None
            #    print(f'No {m} obs found for {site} between {start_date} and {end_date}')
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
                                    ds_all[m]['qYapost{s}'].values[:,model_q_indices[m0][0]],
                                    ds_all[m]['qYapost{s}'].values[:,model_q_indices[m0][1]],
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

#####################################################################

def plot_spatial_flux(ds_all,species,plot_area,model_labels,sectors=None,cmap=None,
                      cmap_diff=None,c_border=None):
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
        plot_area (str):
            Lat/lon region to plot, options for 'UK', 'FRANCE', 'GERMANY',
            'NWEU','CWEU','EUROPE'.
        model_labels (dict of str):
            Models and corresponding strings used to describe the model in the 
            plot legend.
        cmap (str):
            Colour map for flux plots.
        cmap_diff (str):
            Colour map for difference plots.
        c_border (str):
            Colour for flux plot country borders.
    Returns:
        fig (figure): 
            A plot of spatial flux posterior and prior mean/mode and a plot 
            of the absolute difference between these, for each model.
    """
    
    if sectors == None:
        sectors = ['total' for i in range(len(models))]
    
    if cmap == None:
        cmap = 'viridis' #'Blues'
    if cmap_diff == None:
        cmap_diff = 'coolwarm'
    if c_border == None:
        c_border = 'floralwhite'
    
    n_cols = len(ds_all.keys())
    
    fluxlim = {'ch4':[0,1e-7],
               'c2h6':[0,1e-9],
        'hfc134a':[0,1e-11],
        'hfc143a':[0,5e-12],
        'hfc125':[0,1e-11],
        'hfc32':[0,1e-11],
        'hfc227ea':[0,1e-12],
        'pfc218':[0,5e-14],
        'sf6':[0,2e-13],
        'n2o':[0,1e-9]}

    difflim = {'ch4':[-1e-7,1e-7],
               'c2h6':[-1e-9,1e-9],
            'hfc134a':[-1e-11,1e-11],
            'hfc143a':[-5e-12,5e-12],
            'hfc125':[-1e-11,1e-11],
            'hfc32':[-1e-11,1e-11],
            'hfc227ea':[-1e-12,1e-12],
            'pfc218':[-5e-14,5e-14],
            'sf6':[-5e-13,5e-13],
            'n2o':[-1e-9,1e-9]}

    region_limits = {'UK':[-12,4,49,62],   #min_lon, max_lon, min_lat, max_lat
                    'FRANCE':[-6,9,42,52],
                    'GERMANY':[2,18,45,60],
                    'ITALY':[6,19,36,48],
                    'SWITZERLAND':[5.5,11,45,49],
                    'NWEU':[-11,11,45,62],
                    'CWEU':[-12,27,37,66],
                    'EUROPE':[-98,40,10,80]}

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
            ax_var.set_extent(region_limits[plot_area])

    for i,m in enumerate(ds_all.keys()):
        
        lon = ds_all[m].longitude.values + (ds_all[m].longitude.values[1]-ds_all[m].longitude.values[1])/2
        lat = ds_all[m].latitude.values + (ds_all[m].latitude.values[1]-ds_all[m].latitude.values[1])/2

        m0 = m.split('_')[0]
        
        try:
            
            for s,sector_name in enumerate(sectors[i]):
                if s == 0:
                    flux_total_prior = np.mean(ds_all[m][f'flux_{sector_name}_prior'][:,:-1,:-1],axis=0)
                    flux_total_posterior = np.mean(ds_all[m][f'flux_{sector_name}_posterior'][:,:-1,:-1],axis=0)
                    
                else:
                    flux_total_prior = np.mean(ds_all[m][f'flux_{sector_name}_prior'][:,:-1,:-1],axis=0)
                    flux_total_posterior = np.mean(ds_all[m][f'flux_{sector_name}_posterior'][:,:-1,:-1],axis=0)
                    
            flux_diff = flux_total_posterior - flux_total_prior

            if len(ds_all[m].time.values) == 1:
                time_out = to_datetime(ds_all[m].time.values[0].astype(s_data[species]["dt_units"][m0])).strftime('%d/%m/%Y')
            else:
                time_out = (f'{to_datetime(ds_all[m].time.values[0].astype(s_data[species]["dt_units"][m0])).strftime("%d/%m/%Y")} - '+
                            f'{to_datetime(ds_all[m].time.values[-1].astype(s_data[species]["dt_units"][m0])).strftime("%d/%m/%Y")}')

            if n_cols == 1:
                ax0 = ax[0]
                ax1 = ax[1]
                ax2 = ax[2]
            else:
                ax0 = ax[0,i]
                ax1 = ax[1,i]
                ax2 = ax[2,i]

            ax0.pcolormesh(lon,lat,flux_total_prior,cmap=cmap,
                            vmin=fluxlim[species][0],vmax=fluxlim[species][1],shading='flat')

            ax0.set_title(f'{model_labels[m]}:\nprior')
            
            ax1.pcolormesh(lon,lat,flux_total_posterior,cmap=cmap,
                            vmin=fluxlim[species][0],vmax=fluxlim[species][1],shading='flat')

            ax1.set_title(f'{model_labels[m]}:\nposterior')
            
            #flux_diff = np.mean(ds_all[m]['flux_total_posterior'][:,:-1,:-1],axis=0)-np.mean(ds_all[m]['flux_total_prior'][:,:-1,:-1],axis=0)
            flux_diff[np.where(flux_diff) == np.nan] = 0.
            
            ax2.pcolormesh(lon,lat,
                            flux_diff,
                            cmap=cmap_diff,vmin=difflim[species][0],vmax=difflim[species][1],shading='flat')

            ax2.set_title(f'{model_labels[m]}:\nposterior - prior')
                        
        except:
            print(f'ERROR: Either start and end dates are incorrect or there is no model output from {m}.')
            print(f'Skipping plotting {m}.')


    #flux colorbar
    levels = np.linspace(fluxlim[species][0],fluxlim[species][1])
    cbar = plt.cm.ScalarMappable(cmap=cmap)
    cbar.set_array(levels)
    cbar.set_clim(fluxlim[species])

    color_bar1 = fig.colorbar(cbar,orientation='vertical',cmap=cmap,extend='max',ax=ax[0,...],shrink=0.9,pad=0.005)
    color_bar1.set_label(f'Prior mean {s_data[species]["species_print"]}\n{time_out}\n(mol m$^{{-2}}$ s$^{{-1}}$)')

    color_bar2 = fig.colorbar(cbar,orientation='vertical',cmap=cmap,extend='max',ax=ax[1,...],shrink=0.9,pad=0.005)
    color_bar2.set_label(f'Posterior mean {s_data[species]["species_print"]}\n{time_out}\n(mol m$^{{-2}}$ s$^{{-1}}$)')

    #difference colorbar
    levels_diff = np.linspace(difflim[species][0],difflim[species][1])
    cbar_diff = plt.cm.ScalarMappable(cmap=cmap_diff)
    cbar_diff.set_array(levels_diff)
    cbar_diff.set_clim(difflim[species])

    color_bar3 = fig.colorbar(cbar_diff,orientation='vertical',extend='both',ax=ax[2,...],shrink=0.9,pad=0.005)
    color_bar3.set_label(f'Posterior - prior {s_data[species]["species_print"]}\n{time_out}\n(mol m$^{{-2}}$ s$^{{-1}}$)')
    
    return fig

#####################################################################

def plot_spatial_flux_comparison(ds_all,species,plot_area,model_labels,
                                 cmap=None,cmap_diff=None,c_border=None):
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
    Returns:
        fig (figure): 
            A plot of spatial flux posterior from two models a plot 
            of the absolute difference between these.
    """
    
    if cmap == None:
        cmap = 'viridis' #'Blues'
    if cmap_diff == None:
        cmap_diff = 'coolwarm'
    if c_border == None:
        c_border = 'floralwhite'
    
    n_cols = len(ds_all.keys())
    
    fluxlim = {'ch4':[0,1e-7],
        'hfc134a':[0,1e-11],
        'hfc143a':[0,5e-12],
        'hfc125':[0,1e-11],
        'hfc32':[0,1e-11],
        'hfc227ea':[0,1e-12],
        'pfc218':[0,5e-14],
        'sf6':[0,2e-13],
        'n2o':[0,1e-9]}

    difflim = {'ch4':[-1e-7,1e-7],
            'hfc134a':[-1e-11,1e-11],
            'hfc143a':[-5e-12,5e-12],
            'hfc125':[-1e-11,1e-11],
            'hfc32':[-1e-11,1e-11],
            'hfc227ea':[-1e-12,1e-12],
            'pfc218':[-5e-14,5e-14],
            'sf6':[-5e-13,5e-13],
            'n2o':[-1e-9,1e-9]}

    region_limits = {'UK':[-12,4,49,62],   #min_lon, max_lon, min_lat, max_lat
                    'FRANCE':[-6,9,42,52],
                    'GERMANY':[2,18,45,60],
                    'NWEU':[-11,11,45,62],
                    'CWEU':[-12,27,37,66]}

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
                time_out = to_datetime(ds_all[m].time.values[0].astype(s_data[species]["dt_units"][m0])).strftime('%d/%m/%Y')
            else:
                time_out = (f'{to_datetime(ds_all[m].time.values[0].astype(s_data[species]["dt_units"][m0])).strftime("%d/%m/%Y")} - '+
                            f'{to_datetime(ds_all[m].time.values[-1].astype(s_data[species]["dt_units"][m0])).strftime("%d/%m/%Y")}')
        
            ax[0].pcolormesh(lon,lat,
                            np.mean(ds_all[m]['flux_total_posterior'][:,:,:],axis=0),cmap=cmap,
                            vmin=fluxlim[species][0],vmax=fluxlim[species][1],shading='nearest',
                            )

            ax[0].set_title(f'{model_labels[m]}\nPosterior mean')
            
        elif i == 1:
            
            ax[1].pcolormesh(lon,lat,
                            np.mean(ds_all[m]['flux_total_posterior'][:,:-1,:-1],axis=0),cmap=cmap,
                            vmin=fluxlim[species][0],vmax=fluxlim[species][1],shading='flat')

            ax[1].set_title(f'{model_labels[m]}\nPosterior mean')
        
    flux_diff = (np.mean(ds_all[all_keys[1]]['flux_total_posterior'].values[:,:-1,:-1],axis=0)-
                 np.mean(ds_all[all_keys[0]]['flux_total_posterior'].values[:,:-1,:-1],axis=0))
    flux_diff[np.where(flux_diff) == np.nan] = 0.
    
    ax[2].pcolormesh(lon,lat,
                    flux_diff,
                    cmap=cmap_diff,vmin=difflim[species][0],vmax=difflim[species][1],shading='nearest')

    ax[2].set_title(f'{model_labels[all_keys[1]]} - {model_labels[all_keys[0]]}\nAbsolute difference')


    #flux colorbar
    levels = np.linspace(fluxlim[species][0],fluxlim[species][1])
    cbar = plt.cm.ScalarMappable(cmap=cmap)
    cbar.set_array(levels)
    cbar.set_clim(fluxlim[species])

    color_bar2 = fig.colorbar(cbar,orientation='horizontal',cmap=cmap,extend='max',ax=ax[0],shrink=0.9,pad=0.01)
    color_bar2.set_label(f'{s_data[species]["species_print"]}\n{time_out}\n(mol m$^{{-2}}$ s$^{{-1}}$)')

    color_bar2 = fig.colorbar(cbar,orientation='horizontal',cmap=cmap,extend='max',ax=ax[1],shrink=0.9,pad=0.01)
    color_bar2.set_label(f'{s_data[species]["species_print"]}\n{time_out}\n(mol m$^{{-2}}$ s$^{{-1}}$)')

    #difference colorbar
    levels_diff = np.linspace(difflim[species][0],difflim[species][1])
    cbar_diff = plt.cm.ScalarMappable(cmap=cmap_diff)
    cbar_diff.set_array(levels_diff)
    cbar_diff.set_clim(difflim[species])

    color_bar3 = fig.colorbar(cbar_diff,orientation='horizontal',extend='both',ax=ax[2],shrink=0.9,pad=0.01)
    color_bar3.set_label(f'{s_data[species]["species_print"]}\n{time_out}\n(mol m$^{{-2}}$ s$^{{-1}}$)')
    
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