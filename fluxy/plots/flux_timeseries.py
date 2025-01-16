from fluxy import config
import os
import glob
import math
import numpy as np
import xarray as xr
from matplotlib.ticker import NullFormatter
from matplotlib.dates import YearLocator, MonthLocator
import matplotlib.pyplot as plt

from fluxy.operators.regions import extract_region_flux, extract_region_inventory_flux
from fluxy.operators.rolling_mean import calc_rolling_mean
from fluxy.operators.resample_flux import resample_flux
from fluxy.operators.align_dataset import align_dataset
from fluxy.plots.utils import update_list_params

def determine_subplots_arrangement(subplot_number: int
                                   )->list[int]:
    """
    Determine number of columns and rows for the figure given the number of subplots to make.
    Args: 
        subplot_number: number of subplots to make.
    Returns: 
        n_cols,n_rows: number of columns and rows for the figure.
    """
    if subplot_number < 4:
        n_cols = subplot_number
        n_rows = 1
    elif subplot_number == 4:
        n_cols = 2
        n_rows = 2
    elif subplot_number > 4:
        n_cols = 4
        n_rows = math.ceil(subplot_number)/4
    elif subplot_number == 6:
        n_cols = 3
        n_rows = 2
    return n_cols,n_rows


def plot_country_flux(ds_all: dict[str,xr.Dataset],
                      species: str, 
                      plot_regions: list[str],
                      s_data: dict[str,dict],
                      m_data: dict[str,dict],
                      model_colors: dict[str,str],
                      start_date: str,
                      end_date: str,
                      ppt_mode: bool = False,
                      annex_mode: bool = False,
                      scale_co2eq: bool = False,
                      plot_inventory: bool = True,
                      inventory_years: list[str] | None = None,
                      data_dir: str | None = None,
                      fix_y_axes: bool = False,
                      add_prior: bool = True,
                      add_prior_unc: bool = False, 
                      set_global_leg: bool = False,
                      country_codes_as_titles: bool = False,
                      plot_separate: bool | list[bool] = True,
                      plot_combined: bool | list[bool] = False,
                      resample: str | list[str] | None = None,
                      resample_uncert_correlation: bool = False,
                      plot_resample_and_original: bool = False,
                      period_override: list[str] | None = None,
                      return_res: bool = False,
                      rolling_mean: bool | list[bool] = False,
                      apply_pop_scale: bool = True
                     ):#-> plt.figure | list : # DOn't know how to handle this 
    """
    Timeseries plot of prior and posterior country fluxes, from list of 
    areas in plot_regions.
    
    Args:
        ds_all: xarray datasets of fluxes, scaled and sliced between 
            chosen dates.
        species: Gas species, e.g. 'ch4'.
        plot_regions: Country or regions to plot, e.g. ['UNITED KINGDOM','SWITZERLAND']
        s_data: Dictionary of species with information for plotting (read from json file).
        m_data: Dictionary of inversion runs with filename and plot label (read from json file).
        model_colors: Models and corresponding colours used to plot the model.
        start_date: Start dates of the data to plot (used to slice inventory data).
        end_date: Start dates of the data to plot (used to slice inventory data).
        ppt_mode: If True, adjust global legend position to accomodate bigger fonts.
        annex_mode: If True, replace the labels with more concise versions for National Inventory Report Annexes.
        scale_co2eq: If True, adapt y-axis label to CO2-eq.
        plot_inventory: If True, plots inventory flux estimates as bars in each plot.
        inventory_years: List of inventory data from different years to include. If None, only plots the most recent inventory data.
        data_dir: Path to top data directory, used to read inventory data files.
        fix_y_axes: If True, uses a consistent y axis for all plots.
        add_prior: If True, plots prior as dashed lines.
        add_prior_unc: If True, plots prior uncertainty as shaded area.
        set_global_leg: If True, plots one single legend instead of one legend per subplot.
        country_codes_as_titles: If True, uses list of country codes as titles, instead of the region names.
        plot_separate: If True, plots model result as separate line. List must be of same size as models, e.g. [True, False, False].
            If a single boolean is provided, the same flag is assumed for all models.
        plot_combined: If True, the model is included in combined average result to be plotted. List must be of same size as models, e.g. [False, True, True].
            If a single boolean is provided, the same flag is assumed for all models.
        resample: Option to be passed to resample built-in function of xarray Dataset. For yearly average, 'YS' option should be used; 'QS-DEC' for seasonaly average.
            See http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
        resample_uncert_correlation: If True, calculates the resampled uncertainty as the mean from all averaged periods. 
            If False, recalculates uncertainty assuming no correlation between all averaged periods, by taking the square root of the summed variances, divided by the number of averaging periods.
        plot_resample_and_original: If True, plots both the resampled data and the data as its original frequency. If False, only plots the resampled data.
        period_override: Inversion periods to include, to override the standards in species_info.json. Must be the same length as models, e.g. ['monthly',None,'yearly']
        return_res: Wheter or not including a dictionnary with the results as output
        rolling_mean : If True, calculates a rolling mean of the data. List must be of same size as models, e.g. [False, True, True].
            If a single boolean is provided, the same flag is assumed for all models.
        apply_pop_scale : 
    Returns:
        fig: A plot per country/region.
        res_dict : If return_res, return also a dictionnary containaing the plotted results

    """
    if return_res:
        res_dict = {country:dict() for country in plot_regions}
        print('WARNING : Only return the annual combined results for now, so work only if plot_combined=True')
    
    # Convert some inputs to list and check there size
    plot_separate, plot_combined, rolling_mean, resample \
        = update_list_params([plot_separate, plot_combined, rolling_mean, resample], 
                             expected_size = len(ds_all.keys()))

    # Resample xarrays and set list of dataet that will be plotted
    if resample is not None:
        ds_all_resampled = resample_flux(ds_all, resample, resample_uncert_correlation)
        if plot_resample_and_original :
            all_datasets = [ds_all_resampled,ds_all]
        else :
            all_datasets = [ds_all_resampled]
    else:
        all_datasets = [ds_all]

    # Initialize variables
    max_cf = np.zeros(len(plot_regions))
    start_year = [9999]*len(plot_regions)
    end_year = [0]*len(plot_regions)
    min_x = []
    max_x = []
    period_all = {}
    
    if annex_mode:
        lw = 1
        alpha = 0.7
    else:
        lw = 1.5
        alpha = 1

    if 'all' in species: apply_pop_scale = False

    # Create figure
    n_cols, n_rows = determine_subplots_arrangement(len(plot_regions))
        
    fig,axL = plt.subplots(n_rows,n_cols,
                           sharex=True,
                           constrained_layout=True,
                           figsize=(n_cols*6,n_rows*4))
    
    # used to iterate through subplots
    count = 0

    for i,country in enumerate(plot_regions):

        ax = axL.flatten()[i]

        if plot_inventory :
            
            inv_colours = ['grey','black']
            
            if inventory_years is None:
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
        
        for ds in all_datasets:
        
            post_pdfs = {}
            k = 0

            ds_to_combined = []
            
            for j,m in enumerate(ds.keys()):
                    
                ds_region = extract_region_flux(ds[m],m,country,apply_pop_scale)
                
                if ds_region is not None and plot_separate[j] == True:            
                          
                    if ds_count == 0:
                        if annex_mode:
                            include_label = m_data[m]["label"].split()[0]
                            include_label_prior = f'{include_label} prior'
                        else:
                            include_label = m_data[m]["label"]
                            include_label_prior = f'{m_data[m]["label"]} prior'
                    else:
                        include_label = None
                        include_label_prior = None

                if rolling_mean[j]:
                    ds_region = calc_rolling_mean(ds_region)

                ax.plot(ds_region.time,
                        ds_region.region_flux_total_posterior,
                        label=include_label,
                        color=model_colors[m][0])
                
                if plot_combined[j]:
                    ds_to_combined.append(ds_region)
                else:
                    ax.fill_between(ds_region.time,
                                    ds_region.region_flux_total_posterior_lower,
                                    ds_region.region_flux_total_posterior_upper,
                                    alpha=0.3,
                                    color=model_colors[m][0])

                    if add_prior:
                        ax.plot(ds_region.time,
                                ds_region.region_flux_total_prior,
                                label=include_label_prior,
                                color=model_colors[m][0],
                                linestyle='dashed',
                                linewidth=lw,
                                alpha=alpha)
                        max_cf[i] = np.max((max_cf[i],np.nanmax(ds_region.region_flux_total_prior)))

                        if add_prior_unc:
                            ax.fill_between(ds_region.time,
                                            ds_region.region_flux_total_prior_lower,
                                            ds_region.region_flux_total_prior_upper,
                                            alpha=0.1,
                                            color=model_colors[m][0])
                            max_cf[i] = np.max((max_cf[i],np.nanmax(ds_region.region_flux_total_prior_upper)))

                    
                    min_x.append(ds_region.time.min())
                    max_x.append(ds_region.time.max())
                    max_cf[i] = np.max((max_cf[i],np.nanmax(ds_region.region_flux_total_prior_upper)))

                    if plot_inventory == True:
                        if inventory_flux is not None:
                            max_cf[i] = np.nanmax((max_cf[i],np.nanmax(inventory_flux[np.logical_and(inventory_time >= ds_region.time.min(),
                                                                                        inventory_time <= ds_region.time.max())])))
            
            ds_count += 1
        
            if plot_combined is not None and sum(plot_combined)!=0:

                ds_combined = align_dataset(ds_to_combined)
                
                ds_combined = xr.concat(ds_to_combined,'model')
                ds_combined['mean_country_flux_total_posterior'] = ds_combined['region_flux_total_posterior'].mean(dim='model')
                ds_combined['mean_country_flux_total_prior'] = ds_combined['region_flux_total_prior'].mean(dim='model')
                
                ds_combined['min_country_flux_total_lower'] = ds_combined['region_flux_total_posterior_lower'].min(dim='model')
                ds_combined['max_country_flux_total_upper'] = ds_combined['region_flux_total_posterior_upper'].max(dim='model')

                """
                post_pdfs[m] = np.array([np.random.default_rng().normal(loc=region_flux_total_posterior[t],
                                                                        scale=np.mean(np.array([region_flux_total_posterior[t]-region_flux_total_posterior_lower[t],
                                                                                                region_flux_total_posterior_upper[t]-region_flux_total_posterior[t]])),
                                                                        size=1000) for t in range(region_time.shape[0])])
                if i == 0:
                    print('\nNOTE: This currently assumes that posterior PDFs are Gaussian. The average percentile is used '+
                       'to estimate an approximate standard deviation.\n')
                """
                # Define labels
                if annex_mode:
                    labels_combined = {'prior':'PARIS prior',
                                      'posterior':'PARIS mean',
                                      'unc':'_nolegend_'}
                else:
                    labels_combined = {'prior':'Mean prior',
                                      'posterior':'Mean posterior',
                                      'unc':'Min/max of post uncertainty'}
                
                if return_res :
                    res_dict[country]['combined']= {'time':ds_combined.time,
                                                    'mean':ds_combined.region_flux_total_posterior,
                                                    'min':ds_combined.min_country_flux_total_lower,
                                                    'max':ds_combined.max_country_flux_total_upper}
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
                ax.plot(ds_combined.time,
                        ds_combined.mean_country_flux_total_posterior,
                        label=labels_combined['posterior'],
                        color='black',
                        linewidth=3.5)
                
                if add_prior:
                    ax.plot(ds_combined.time,
                            ds_combined.mean_country_flux_total_prior,
                            label=labels_combined['prior'],
                            color='black',
                            linestyle='dashed',
                            linewidth=lw,
                            alpha=alpha)
                
                ax.fill_between(ds_combined.time,
                                ds_combined.min_country_flux_total_lower,
                                ds_combined.max_country_flux_total_upper,
                                alpha=0.3,
                                color='grey',
                                label=labels_combined['unc'])
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
        
        # if period_all[list(ds.keys())[0]] == 'monthly' and resample != 'year':
        #     ax.set_xlim([np.min(min_x)-np.timedelta64(1,'M'),
        #                     np.max(max_x)+np.timedelta64(1,'M')])
        # else: #period_all[list(ds.keys())[0]] == 'yearly':
        #     ax.set_xlim([np.min(min_x)-np.timedelta64(7,'M'),
        #                     np.max(max_x)+np.timedelta64(7,'M')])        
        
        ncol = 2
        if annex_mode: ncol = 3
        if not set_global_leg:
            leg = ax.legend(ncol=ncol,borderpad=.4,columnspacing=1.0)
            if plot_inventory:
                for l in leg.legendHandles[:-1]:
                    l.set_linewidth(3.0)
            else:
                for l in leg.legendHandles:
                    l.set_linewidth(3.0)
        
        country_equivalent = {'NW_EU2':'NW EUROPE',
                              'CW_EU':'CENTRAL W EUROPE',
                              'NW_EU_CONTINENT':'NW CONTINENTAL EUROPE'}
        print_country = country_equivalent[country] if country in country_equivalent.keys() else country
        
        if country_codes_as_titles:
            try:
                ax.set_title(f'{print_country}\n{config.regions_dict[country]}')
            except:
                ax.set_title(f'{print_country}')
        else:
            ax.set_title(f'{print_country}')
            
        ax.grid(visible=True,which='major',alpha=0.4)

        # if (end_year[i]-start_year[i]) > 8:
        #     years_list = list(range(start_year[i],end_year[i]+2))
        #     region_time = np.array([np.datetime64(str(year), 'Y') for year in years_list])
        #     ax.set_xticks(region_time[::2])
        #     ax.set_xticklabels(region_time[::2].astype('datetime64[Y]'))
        #     ax.xaxis.set_minor_formatter(NullFormatter())

        # else:
        #     ax.xaxis.set_minor_locator(MonthLocator())
        #     ax.xaxis.set_minor_formatter(NullFormatter())
        #     ax.xaxis.set_major_locator(YearLocator())
        
        count += 1
        
        handles, labels = ax.get_legend_handles_labels()
        if any('Inventory' in l for l in labels) == True:
            handles_all = handles.copy()
            labels_all = labels.copy()
        
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
        # leg = fig.legend(handles_all, labels_all, loc='upper center',ncol=ncol,borderpad=.4,columnspacing=1.0,bbox_to_anchor=legend_loc)
        #if plot_inventory == True:
        #    for l in leg.legendHandles:
        #        l.set_linewidth(3.0)
        #else:
        #    for l in leg.legendHandles:
        #        l.set_linewidth(3.0)

    # loop through plots again to fix min/max axis values
    
    # fac = 1.1
    # if not(set_global_leg): fac = 1.2
    # for i,country in enumerate(plot_regions):
    #     if fix_y_axes == True:
    #         fig.axes[i].set_ylim([0,np.nanmax(max_cf)*fac])  
    #     elif (type(fix_y_axes) == list) == True:
    #         fig.axes[i].set_ylim(fix_y_axes)
    #     elif fix_y_axes == False:
    #         fig.axes[i].set_ylim([0,max_cf[i]*fac])  
    
    print('NOTE: If all the data is not within axis limits, adjust the set_ylim parameter')
    
    if return_res:
        return fig,res_dict
    else:
        return fig

