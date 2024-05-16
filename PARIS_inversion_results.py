import xarray as xr
import numpy as np
from pandas import to_datetime
import matplotlib.pyplot as plt
import os
import glob
import math
from matplotlib.dates import YearLocator, MonthLocator
from matplotlib.ticker import NullFormatter
import pprint
import cartopy
from json import load

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
                     'LUXEMBOURG': 'LUX'}

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
                   2:[0.7,0.40]}

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

def read_flux(data_dir,species,models,model_filenames):
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
                                       
    Returns:
        ds_all (dictionary of datasets): 
            xarray dataset read directly from each model's flux netCDF.
    """
    
    ds_all = {}

    for m in models:
        print(f'\nAttempting to read data from {m}')
        
        m0 = m.split('_')[0]
        
        model_dir = model_filenames[m].split('_')[0]
        
        try:
            filepath = glob.glob(os.path.join(data_dir,model_dir,species,f'{model_filenames[m]}_{s_data[species]["model_species"][m0]}*_{s_data[species]["period"]}.nc'))
            print(f'Reading data from: {filepath[0]}')
            with xr.open_dataset(filepath[0]) as in_ds:
                ds_all[m] = in_ds
                print('Done!')
        except:
            try:
                if (model_filenames[m].split('_')[-1] == 'std*'):
                    alternative_filename = f'{model_filenames[m][0:-5]}_{m0}_obs_{m0}_baseline_optimized'
                    filepath = glob.glob(os.path.join(data_dir,model_dir,species,f'{alternative_filename}_{s_data[species]["model_species"][m0]}_{s_data[species]["period"]}.nc'))
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
    
    elris_scale = ['flux_total_prior','flux_total_posterior','percentile_flux_total_prior',
                'percentile_flux_total_posterior']
        
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

            # fix for flux scaling issue in ELRIS - to be removed once fixed in .nc files
            if 'elris_old' in m:
                for v in elris_scale:
                    ds_all[m][v].values = ds_all[m][v].values/1e12
                    print('Old ELRIS file! Applying additional scaling correction.')
        
    return ds_all

#####################################################################

def read_mf(data_dir,species,models,model_filenames):
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
                                       
    Returns:
        ds_all (dictionary of datasets): 
            xarray dataset read directly from each model's mole fraction netCDF.
    """

    ds_all = {}

    for m in models:
        
        m0 = m.split('_')[0]
        model_dir = model_filenames[m].split('_')[0]
        
        print(f'\nAttempting to read data from {m}')
        try:
            filepath = glob.glob(os.path.join(data_dir,model_dir,species,f'{model_filenames[m]}_{s_data[species]["model_species"][m0]}_{s_data[species]["period"]}_concentrations.nc'))
            print(f'Reading data from: {filepath[0]}')
            with xr.open_dataset(filepath[0]) as in_ds:
                ds_all[m] = in_ds
            print('Done!')
        except:
            try:
                if (model_filenames[m].split('_')[-1] == 'std*'):
                    alternative_filename = f'{model_filenames[m][0:-5]}_{m0}_obs_{m0}_baseline_optimized'
                    filepath = glob.glob(os.path.join(data_dir,model_dir,species,f'{alternative_filename}_{s_data[species]["model_species"][m0]}_{s_data[species]["period"]}_concentrations.nc'))
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
        print(f'\nMasking data from {m}')
        
        if 'Yav' in ds_all[m].keys():
            offset = int(np.mean(ds_all[m]['Yav'].values))
        else:
            offset = (ds_all[m].time.values[1].astype('datetime64[h]') - ds_all[m].time.values[0].astype('datetime64[h]')).astype(int)

        # fix to move elris timestamps back to the middle of av period - to be removed once fixed in .nc files
        if 'elris_old' in m:
            ds_all[m]['time'] = ds_all[m]['time'] - np.timedelta64(offset,'h')/2

        # round seconds to integer (correction for elris)
        if 'elris' in m:
            ds_all[m]['time'] = ds_all[m]['time'].dt.round('s')

        if site is not None:
            #try:
                if 'mcmc' in m:
                    site_index = np.where(ds_all[m][f'sitenames_{species}'].astype(str) == site)[0][0]
                    ds_all[m] = ds_all[m].sel({f'time_{species}':slice(start_date,end_date),
                                               f'nsite_{species}':site_index})
                else:
                    site_index = np.where(ds_all[m]['sitenames'].astype(str) == site)[0][0]
                    ds_all[m] = ds_all[m].sel(time=slice(start_date,end_date),
                                            nsite=site_index)
            #except:
            #    ds_all[m] = None
            #    print(f'No {m} obs found for {site} between {start_date} and {end_date}')
        else:
            try:
                ds_all[m] = ds_all[m].sel(time=slice(start_date,end_date))
            except:
                ds_all[m] = None
                print(f'No {m} obs found between {start_date} and {end_date}')
                
        if scale_units == True:
            print(f'Scaling {m} units by {s_data[species]["mf_units_scaling"]}')
            if ds_all[m] is not None:
                if 'mcmc' in m:
                    var_names = [k for k in ds_all[m].keys() if k not in [f'sitenames_{species}','Yav']]
                else:
                    var_names = [k for k in ds_all[m].keys() if k not in ['sitenames','Yav']]
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

def plot_obs_modelled_separate(ds_all,species,site,model_labels,
                               model_colors,
                             include=['Yobs','Yapriori','Yapost'],
                             diff_include=['Yapriori','Yapost'],
                             add_unc=True):
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
                  'Ybias':'posterior bias',
                  'YaprioriOUTER':'prior outer region mf',
                  'YapostOUTER':'posterior outer region mf',
                  'Yobs':'observed mf',
                  'uYobs':'observed mf uncertainty',
                  'uYmod':'model uncertainty'}
    var_colors = {'Yapriori':1,
                  'Yapost':1,
                  'YaprioriBC':1,
                  'YapostBC':1,
                  'Ybias':0,
                  'YaprioriOUTER':1,
                  'YapostOUTER':1,
                  'Yobs':0,
                  'uYobs':0,
                  'uYmod':1}
        
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
                               label=f'Obs ({model_labels[m]})',s=8,alpha=0.8,marker='s')
                    
                    if add_unc:
                        ax.errorbar(ds_all[m].time.values,
                                    ds_all[m]['Yobs'].values,
                                    ds_all[m]['uYobs'].values,
                                    color=model_colors[m][var_colors[var]],alpha=0.4,fmt='none')
                else:
                    
                    ax.scatter(ds_all[m].time.values,
                                ds_all[m]['Yobs'].values,
                                color='black',label=f'Obs ({model_labels[m]})',s=8,alpha=0.8,
                                marker='s')
                    
                    if add_unc:
                        ax.errorbar(ds_all[m].time.values,
                                    ds_all[m]['Yobs'].values,
                                    ds_all[m]['uYobs'].values,
                                        color='black',alpha=0.4,fmt='none')

            elif var == 'uYmod':
                uYmod = ds_all[m]['Yobs'].values - ds_all[m]['qYmod'].values[:,model_q_indices[m0][0]]
                ax.scatter(ds_all[m].time.values,
                           uYmod,
                           color=model_colors[m][var_colors[var]],label=f'{model_labels[m]} {var_labels[var]}',s=8,alpha=0.8)

            else:
                ax.plot(ds_all[m].time.values,
                        ds_all[m][var].values,
                        color=model_colors[m][var_colors[var]],alpha=0.8,
                        linewidth=2.,
                        label=f'{model_labels[m]} {var_labels[var]}')
                
                #ax.scatter(ds_all[m].time.values,
                #        ds_all[m][var].values,
                #        color=model_colors[m][var_colors[var]],alpha=0.5,
                #        s=8,
                #        label=f'{model_labels[m]} {var_labels[var]}')
                

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
                if var == 'uYmod':
                    var_plot = uYmod
                else:
                    var_plot = ds_all[m][var].values

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
        
        if int(ds_all[m].time.values[-1].astype('datetime64[M]')-ds_all[m].time.values[0].astype('datetime64[M]')) > 12:
            ax.xaxis.set_minor_locator(MonthLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.xaxis.set_major_locator(YearLocator())
        else:
            ax.xaxis.set_major_locator(MonthLocator())
                    
    for i in range(len(models)):
        ax_all[i].set_ylim([min(min_mf)-(0.02*min(min_mf)),
                            max(max_mf)+(0.05*max(max_mf))])
        
    print('NOTE: If all the data is not within axis limits, adjust the set_ylim')
    print('NOTE: If annotations in the histograms are not displaying correctly, adjust annotate_coords.')
    
    return fig

#####################################################################

def plot_obs_modelled_together(ds_all,species,site,model_labels,
                               model_colors,
                               include=['Yapost'],
                               diff_include=['Yapost'],
                               add_unc=True):
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
    Returns:
        fig (figure): 
            One timeseries and histogram plot containing data from all models.
    """

    var_labels = {'Yapriori':'prior mf',
                  'Yapost':'posterior mean mf',
                  'YaprioriBC':'prior baseline',
                  'YapostBC':'posterior mean baseline',
                  'Ybias':'posterior bias',
                  'YaprioriOUTER':'prior outer region mf',
                  'YapostOUTER':'posterior outer region mf',
                  'Yobs':'observed mf',
                  'uYobs':'uncertainty observed mf',
                  'uYmod':'model uncertainty'}
    var_colors = {'Yapriori':1,
                  'Yapost':0,
                  'YaprioriBC':1,
                  'YapostBC':0,
                  'Ybias':0,
                  'YaprioriOUTER':1,
                  'YapostOUTER':0,
                  'Yobs':0,
                  'uYobs':0,
                  'uYmod':0}
        
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
                                color=model_colors[m][var_colors[var]],label=f'Obs ({model_labels[m]})',s=5,alpha=0.5)

                    if add_unc:
                        ax.fill_between(ds_all[m].time.values,
                                        ds_all[m]['Yobs'].values - ds_all[m]['uYobs'].values,
                                        ds_all[m]['Yobs'].values + ds_all[m]['uYobs'].values,
                                        color=model_colors[m][var_colors[var]],alpha=0.2)
                else:
                    #ax.plot(ds_all[m].time.values,
                    #            ds_all[m]['Yobs'].values,
                    #            color='dimgrey',label=f'Obs ({model_labels[m]})')
                    ax.scatter(ds_all[m].time.values,
                                ds_all[m]['Yobs'].values,
                                color='dimgrey',label=f'Obs ({model_labels[m]})',s=5,alpha=0.5)

            elif var == 'uYmod':
                uYmod = ds_all[m]['Yobs'].values - ds_all[m]['qYmod'].values[:,model_q_indices[m0][0]]
                ax.scatter(ds_all[m].time.values,
                           uYmod,
                           color=model_colors[m][var_colors[var]],label=f'{model_labels[m]} {var_labels[var]}',s=8,alpha=0.8)

            else:
                ax.scatter(ds_all[m].time.values,
                        ds_all[m][var].values,
                        color=model_colors[m][var_colors[var]],alpha=0.5,
                        label=f'{model_labels[m]} {var_labels[var]}',
                        linewidth=2,s=8)

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
                if var == 'uYmod':
                    var_plot = uYmod
                else:
                    var_plot = ds_all[m][var].values

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
    
    if int(ds_all[m].time.values[-1].astype('datetime64[M]')-ds_all[m].time.values[0].astype('datetime64[M]')) > 12:
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_locator(YearLocator())
    else:
        ax.xaxis.set_major_locator(MonthLocator())
        
    ax.set_ylim([min(min_mf)-(0.02*min(min_mf)),
                            max(max_mf)+(0.05*max(max_mf))])
        
    print('NOTE: If all the data is not within axis limits, adjust the set_ylim')
    print('NOTE: If annotations in the histograms are not displaying correctly, adjust annotate_coords.')
    
    return fig

#####################################################################

def plot_obs_diff(ds_all,species,site,model_labels,
                               model_colors,
                               include=['Yapost'],
                               diff_include=['Yapost']):
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
    Returns:
        fig (figure): 
            A timeseries and histogram plot for each model included.
    """

    var_labels = {'Yapriori':'prior mf',
                  'Yapost':'posterior mean mf',
                  'YaprioriBC':'prior baseline',
                  'YapostBC':'posterior mean baseline',
                  'Ybias':'posterior bias',
                  'YaprioriOUTER':'prior outer region mf',
                  'YapostOUTER':'posterior outer region mf',
                  'Yobs':'observed mf',
                  'uYobs':'uncertainty observed mf',
                  'uYmod':'model uncertainty'}
    var_colors = {'Yapriori':1,
                  'Yapost':0,
                  'YaprioriBC':1,
                  'YapostBC':0,
                  'Ybias':0,
                  'YaprioriOUTER':1,
                  'YapostOUTER':0,
                  'Yobs':0,
                  'uYobs':0,
                  'uYmod':0}
        
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
        if var == 'uYmod':
            m00 = models[0].split('_')[0]
            m01 = models[1].split('_')[0]

            uYmod0 = ds_all[models[0]]['Yobs'].values - ds_all[models[0]]['qYmod'].values[:,model_q_indices[m00][0]]
            uYmod1 = ds_all[models[1]]['Yobs'].values - ds_all[models[1]]['qYmod'].values[:,model_q_indices[m01][0]]

            ax.scatter(ds_all[models[0]].time.values,
                       uYmod0 - uYmod1,
                       color=model_colors[models[0]][var_colors[var]],alpha=0.5,
                       label=f'{model_labels[models[0]]} - {model_labels[models[1]]}\n{var_labels[var]}',
                       linewidth=2,s=8)

        else:
            ax.scatter(ds_all[models[0]].time.values,
                       ds_all[models[0]][var].values - ds_all[models[1]][var].values,
                       color=model_colors[models[0]][var_colors[var]],alpha=0.5,
                       label=f'{model_labels[models[0]]} - {model_labels[models[1]]}\n{var_labels[var]}',
                       linewidth=2,s=8)

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
                if var == 'uYmod':
                    var_plot = ds_all[m]['Yobs'].values - ds_all[m]['qYmod'].values[:,model_q_indices[m0][0]]
                else:
                    var_plot = ds_all[m][var].values
            
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
    
    if int(ds_all[m].time.values[-1].astype('datetime64[M]')-ds_all[m].time.values[0].astype('datetime64[M]')) > 12:
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_locator(YearLocator())
    else:
        ax.xaxis.set_major_locator(MonthLocator())
        
    ax.set_ylim([min(min_mf)-(0.02*min(min_mf)),
                            max(max_mf)+(0.05*max(max_mf))])
        
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
                      plot_inventory=True,data_dir=None,fix_y_axes=False,
                      add_prior_unc=False, set_global_leg=False):
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
        data_dir (str): 
            Path to top data directory, used to read inventory data files.
        fix_y_axes (bool):
            If True, uses a consistent y axis for all plots.
        add_prior_unc (bool):
            If True, plots prior uncertainty as shaded area.
        set_global_leg (bool):
            If True, plots one single legend instead of one legend per subplot.
    Returns:
        fig (figure): 
            A plot per country/region.
    """
    
    if plot_inventory == True:
        with xr.open_dataset(os.path.join(data_dir,'inventory',f'UNFCCC_inventory_{s_data[species]["model_species"]["intem"]}.nc')) as f:
            inv_ds = f

    a,b = 0,0
    max_cf = []
    min_x = []
    max_x = []

    n_cols = math.ceil(len(plot_regions)/2)
    if n_cols <= 1:
        n_cols = 2
        
    fig,ax = plt.subplots(2,n_cols,figsize=(n_cols*6,8),constrained_layout=True)

    for i,country in enumerate(plot_regions):
        
        if plot_inventory == True:
            try:
                inv_c_index = np.where(inv_ds['country'].values == country)[0][0]
                ax[a,b].bar(inv_ds.time.values,inv_ds['inventory'].values[:,inv_c_index]/s_data[species]["units_scaling"]["intem"],
                            np.timedelta64(340, 'D'),color='white',edgecolor='black',align='edge',
                            label='Inventory 2023',zorder=0)
            except:
                try:
                    region_search = regions_dict[country]
                    country_list = region_search.split('-')

                    inv_c_index = [0]*len(country_list)
                    inv_c_value = np.zeros(len(inv_ds.time.values))

                    try:
                        for i,var in enumerate(country_list):
                            inv_key = [k for k, code in countrycodes_dict.items() if code == var]
                            inv_c_index[i] = np.where(inv_ds['country'].values == inv_key[0])[0][0]
                            inv_c_value = inv_c_value + inv_ds['inventory'].values[:,inv_c_index[i]]

                        ax[a,b].bar(inv_ds.time.values,inv_c_value/s_data[species]["units_scaling"]["intem"],
                                    np.timedelta64(340, 'D'),color='white',edgecolor='black',align='edge',
                                    label='Inventory 2023',zorder=0)

                        print(f'No inventory data available for {country}. Considering sum of individual countries: {region_search}')

                    except:
                        print(f'No inventory data available for {inv_key[0]}. Inventory data will not be plotted for {country}.')

                except:
                    print(f'No inventory data available for {country}')
        
        for m in ds_all.keys():
            
            m0 = m.split('_')[0]
            
            if m0 == 'intem':
                c_key = 'countrynumber'
            elif m0 == 'rhime':
                c_key = 'country'
            elif m0 == 'elris':
                c_key = 'country'
            elif m0 == 'mcmc':
                c_key = 'countrynumber'
            
            try:
                
                # fix for error in CW_EU definition in counrtycodes_dict and older InTEM netCDF files
                try:
                    country_search = countrycodes_dict[country]
                    country_index = np.where(ds_all[m][c_key].values.astype(str) == country_search)[0][0]
                
                except:
                    country_search = regions_dict_old[country]
                    country_index = np.where(ds_all[m][c_key].values.astype(str) == country_search)[0][0]
                    
                ax[a,b].plot(ds_all[m].time.values.astype('datetime64[ns]'),
                            ds_all[m]['country_flux_total_posterior'].values[:,country_index],
                            label=model_labels[m],color=model_colors[m][0])
                
                ax[a,b].plot(ds_all[m].time.values.astype('datetime64[ns]'),
                            ds_all[m]['country_flux_total_prior'].values[:,country_index],
                            label=f'{model_labels[m]} prior',color=model_colors[m][0],linestyle='dashed')
                
                max_cf.append(ax[a,b].get_ylim()[1])
                
                ax[a,b].fill_between(ds_all[m].time.values.astype('datetime64[ns]'),
                                    ds_all[m]['percentile_country_flux_total_posterior'].values[:,model_q_indices[m0][0],country_index],
                                    ds_all[m]['percentile_country_flux_total_posterior'].values[:,model_q_indices[m0][1],country_index],
                                    alpha=0.3,color=model_colors[m][0])
                
                if add_prior_unc:
                    ax[a,b].fill_between(ds_all[m].time.values.astype('datetime64[ns]'),
                                        ds_all[m]['percentile_country_flux_total_prior'].values[:,model_q_indices[m0][0],country_index],
                                        ds_all[m]['percentile_country_flux_total_prior'].values[:,model_q_indices[m0][1],country_index],
                                        alpha=0.1,color=model_colors[m][0])
                
                min_x.append(np.min(ds_all[m].time.values).astype('datetime64[M]'))
                max_x.append(np.max(ds_all[m].time.values).astype('datetime64[M]'))
        
            except:
                try:
                    region_search = countrycodes_dict[country]
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

                    ax[a,b].plot(ds_all[m].time.values.astype('datetime64[ns]'),
                                    region_flux_total_posterior,
                                    label=model_labels[m],color=model_colors[m][0])

                    ax[a,b].plot(ds_all[m].time.values.astype('datetime64[ns]'),
                                    region_flux_total_prior,
                                    label=f'{model_labels[m]} prior',color=model_colors[m][0],linestyle='dashed')

                    max_cf.append(ax[a,b].get_ylim()[1])

                    sigma_region_flux_total_prior = np.sqrt(sigma2_region_flux_total_prior)

                    if add_prior_unc:
                        ax[a,b].fill_between(ds_all[m].time.values.astype('datetime64[ns]'),
                                            region_flux_total_prior - sigma_region_flux_total_prior,
                                            region_flux_total_prior + sigma_region_flux_total_prior,
                                            alpha=0.1,color=model_colors[m][0])

                    # Compute posterior uncertainty from covariance matrix
                    try:
                        sigma2 = np.zeros(np.shape(ds_all[m]['covariance_country_flux_total_posterior'])[0])

                        for i in range(len(sigma2)):
                            sigma2[i] = country_index_vec.dot(ds_all[m]['covariance_country_flux_total_posterior'].values[i,:,:].dot(country_index_vec))

                        sigma_region_flux_total_posterior = np.sqrt(sigma2)

                        ax[a,b].fill_between(ds_all[m].time.values.astype('datetime64[ns]'),
                                    region_flux_total_posterior - sigma_region_flux_total_posterior,
                                    region_flux_total_posterior + sigma_region_flux_total_posterior,
                                    alpha=0.3,color=model_colors[m][0])

                    except:
                        print(f'WARNING: Covariance matrix is not available for {m}. A posteriori uncertainty of {country} emissions will not be plotted.')

                    min_x.append(np.min(ds_all[m].time.values).astype('datetime64[M]'))
                    max_x.append(np.max(ds_all[m].time.values).astype('datetime64[M]'))

                except:
                    print(f'ERROR: Either start and end dates are incorrect or there is no {country} emissions in {m}.')
                    print(f'Skipping plotting {m}.')
                                                    
        #format each subplot
        
        ax[a,b].set_ylabel(f'{s_data[species]["species_print"]} ({s_data[species]["units_print"]}g y$^{{-1}}$)')
        ax[a,b].set_xlim([np.min(min_x)-np.timedelta64(1,'M'),
                        np.max(max_x)+np.timedelta64(1,'M')])

        ncol = 2
        '''
        if len(ds_all.keys()) == 3:
            if plot_inventory == True:
                ncol = 2
            else:
                ncol = 3
        else:
            if plot_inventory == True:
                ncol = 3
            else:
                ncol = 2
        '''
        if set_global_leg == False:
            leg = ax[a,b].legend(ncol=ncol,borderpad=.4,columnspacing=1.0,fontsize=10)
            if plot_inventory == True:
                for l in leg.legendHandles[:-1]:
                    l.set_linewidth(3.0)
            else:
                for l in leg.legendHandles:
                    l.set_linewidth(3.0)
                
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
        ncol=len(ds_all.keys())
        if plot_inventory == True:
            ncol=ncol+1
        leg = fig.legend(handles, labels, loc='upper center',ncol=ncol,borderpad=.4,columnspacing=1.0,fontsize=10,bbox_to_anchor=(0.5, 1.07))
        if plot_inventory == True:
            for l in leg.legendHandles[:-1]:
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

def plot_spatial_flux(ds_all,species,plot_area,model_labels,cmap='viridis',
                                 cmap_diff='coolwarm',border_color_in='floralwhite'):
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
    Returns:
        fig (figure): 
            A plot of spatial flux posterior and prior mean/mode and a plot 
            of the absolute difference between these, for each model.
    """
    
    #cmap = 'viridis' #'Blues'
    #cmap_diff = 'coolwarm'
    
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
                border_color = border_color_in #'floralwhite'

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

            ax0.pcolormesh(lon,lat,
                            np.mean(ds_all[m]['flux_total_prior'][:,:-1,:-1],axis=0),cmap=cmap,
                            vmin=fluxlim[species][0],vmax=fluxlim[species][1],shading='flat')

            ax0.set_title(f'{model_labels[m]}: prior')
            
            ax1.pcolormesh(lon,lat,
                            np.mean(ds_all[m]['flux_total_posterior'][:,:-1,:-1],axis=0),cmap=cmap,
                            vmin=fluxlim[species][0],vmax=fluxlim[species][1],shading='flat')

            ax1.set_title(f'{model_labels[m]}: posterior')
            
            flux_diff = np.mean(ds_all[m]['flux_total_posterior'][:,:-1,:-1],axis=0)-np.mean(ds_all[m]['flux_total_prior'][:,:-1,:-1],axis=0)
            flux_diff[np.where(flux_diff) == np.nan] = 0.
            
            ax2.pcolormesh(lon,lat,
                            flux_diff,
                            cmap=cmap_diff,vmin=difflim[species][0],vmax=difflim[species][1],shading='flat')

            ax2.set_title(f'{model_labels[m]}: posterior - prior')
                
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

def plot_spatial_flux_comparison(ds_all,species,plot_area,model_labels,cmap='viridis',
                                 cmap_diff='coolwarm',border_color_in='floralwhite'):
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
    Returns:
        fig (figure): 
            A plot of spatial flux posterior from two models a plot 
            of the absolute difference between these.
    """
    
    #cmap = 'viridis' #'Blues'
    #cmap_diff = 'coolwarm'
    
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
            border_color = border_color_in #'floralwhite'
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
