from fluxy import config
import os
import glob
import math
import numpy as np
import xarray as xr
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib.ticker import NullFormatter
from matplotlib.dates import YearLocator, MonthLocator
import matplotlib.pyplot as plt

from fluxy.operators.regions import extract_region_flux
from fluxy.operators.rolling_mean import calc_rolling_mean
from fluxy.operators.flux_resample import resample_flux
from fluxy.operators.flux_combine import combine_dataset
from fluxy.operators.flux_prepare_inventory import derive_inventories
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

def prepare_data_to_plot(ds_region,plot_separate, plot_combined, resample, rolling_mean, plot_resample_and_original,resample_uncert_correlation):
            
    # Convert some inputs to list and check there size
    plot_separate, plot_combined, resample \
        = update_list_params([plot_separate, plot_combined, resample], 
                            expected_size = len(ds_region.keys()))
    ds_to_plot = dict()
    
    if not any(resample) or plot_resample_and_original:
        ds_original_flux = {m:v for (i,(m,v)) in enumerate(ds_region.items()) if plot_separate[i]}
        ds_to_plot.update(ds_original_flux)

    if any(resample) : 
        ds_resampled = resample_flux(ds_region, resample, resample_uncert_correlation)
        ds_to_plot.update({m:v for (i,(m,v)) in enumerate(ds_resampled.items()) 
                           if plot_separate[i]})        
    
    if any(plot_combined):
        if all([resamp for comb,resamp in zip(plot_combined,resample) if comb]):
            ds_combined = combine_dataset(ds_resampled, plot_combined)
        else :
            ds_combined = combine_dataset(ds_region, plot_combined)
        ds_to_plot.update(ds_combined)

    if rolling_mean:
        ds_to_plot = {m: calc_rolling_mean(ds) for m,ds in ds_to_plot.items()}
    
    return ds_to_plot


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
                      rolling_mean: bool = False,
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
        rolling_mean : If True, calculates a rolling mean (xx years) for each of the data to plot.
        apply_pop_scale : 
    Returns:
        fig: A plot per country/region.
        res_dict : If return_res, return also a dictionnary containaing the plotted results

    """
    if return_res:
        res_dict = {country:dict() for country in plot_regions}
        print('WARNING : Only return the annual combined results for now, so work only if plot_combined=True')
    
    max_cf = np.zeros(len(plot_regions))
    min_x = []
    max_x = []
    lw, alpha = (1.0, 0.7) if annex_mode else (1.5, 1.0)

    # Create figure
    n_cols, n_rows = determine_subplots_arrangement(len(plot_regions))
        
    fig,axL = plt.subplots(n_rows,n_cols,
                           sharex=True,
                           constrained_layout=True,
                           figsize=(n_cols*6,n_rows*4))
    for i,country in enumerate(plot_regions):

        ax = axL.flatten()[i]

        if plot_inventory :
            inventories_to_plot = derive_inventories(data_dir,country,species,start_date,end_date,s_data,scale_co2eq,inventory_years)
            for i_inv,inventory in enumerate(inventories_to_plot) :
                ax.bar(inventory.time,inventory,
                       np.timedelta64(340-i_inv*20, 'D'),
                       edgecolor=inventory.plot_color,
                       align='edge',fill=False,
                       label=f'Inventory {inventory.year}',
                       zorder=0)
        
        ds_all_region = extract_region_flux(ds_all,country)

        ds_to_plot = prepare_data_to_plot(ds_all_region, plot_separate, plot_combined, 
                                            resample, rolling_mean, 
                                            plot_resample_and_original,
                                            resample_uncert_correlation)

        for m, ds_region in ds_to_plot.items():

            if m in m_data:
                m_org, add_label = m, ''
            elif m == 'combined':
                m_org, add_label = m, ''
                m_data[m_org] = {"label" : 'PARIS mean'}
                model_colors[m_org] = ['black','gray']
            elif m.replace('_resample','') in m_data:
                m_org, add_label = m.replace('_resample',''), ' (resampled)'

            if annex_mode:
                include_label = m_data[m_org]["label"].split()[0] + add_label
            else:
                include_label = m_data[m_org]["label"] + add_label
            include_label_prior = f'{include_label} prior'
                
            ax.plot(ds_region.time,
                    ds_region.region_flux_total_posterior,
                    label=include_label,
                    color=model_colors[m_org][0]) 
            ax.fill_between(ds_region.time,
                            ds_region.region_flux_total_posterior_lower,
                            ds_region.region_flux_total_posterior_upper,
                            alpha=0.3,
                            color=model_colors[m_org][0])   
            
            if add_prior:
                ax.plot(ds_region.time,
                        ds_region.region_flux_total_prior,
                        label=include_label_prior,
                        color=model_colors[m_org][0],
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
        
        # set legend if needed
        if not set_global_leg:
            ncol = 3 if annex_mode else 2
            leg = ax.legend(ncol=ncol,borderpad=.4,columnspacing=1.0)
            if plot_inventory:
                for l in leg.legendHandles[:-1]:
                    l.set_linewidth(3.0)
            else:
                for l in leg.legendHandles:
                    l.set_linewidth(3.0)
        
        # set title
        country_equivalent = {'NW_EU2':'NW EUROPE',
                              'CW_EU':'CENTRAL W EUROPE',
                              'NW_EU_CONTINENT':'NW CONTINENTAL EUROPE'}
        print_country = country_equivalent[country] if country in country_equivalent.keys() else country

        if country_codes_as_titles and country in config.regions_dict.keys():
            ax.set_title(f'{print_country}\n{config.regions_dict[country]}')
        else:
            ax.set_title(f'{print_country}')
        
        # set grid
        ax.grid(visible=True,which='major',alpha=0.4)
            
    # set xticks
    xlim = pd.to_datetime(ax.get_xlim(), unit='D', origin=pd.Timestamp('1970-01-01'))
    ax.set_xlim(np.datetime64(str(xlim.year[0]), 'Y'),
                np.datetime64(str(xlim.year[1]), 'Y')+1)

    if (xlim.year[1] - xlim.year[0]) > 10:
        xticks = np.array([np.datetime64(str(year), 'Y') for year in range(xlim.year[0],xlim.year[1],2)])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks.astype('datetime64[Y]'))
        ax.xaxis.set_minor_formatter(NullFormatter())

    else:
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_locator(YearLocator())
        
    if set_global_leg:
        ncol=0   
        if (plot_separate or resample):
            ncol=len(ds_all.keys())
        if (plot_combined and plot_separate):
            ncol=math.floor(len(ds_all.keys())/2)+2
        elif plot_combined:
            ncol=3
        if plot_inventory:
            ncol=ncol+1
            
        if n_rows > 1:
            if (ppt_mode):
                legend_loc = (0.5, 1.1)
            else:
                legend_loc = (0.5, 1.07)
        else:
            legend_loc = (0.5, 1.15)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, 
                   loc='upper center',
                   ncol=ncol,
                   borderpad=.4,
                   columnspacing=1.0,
                   )

    # fac = 1.1 if set_global_leg else 1.2
    
    # loop through plots again to fix min/max y-axis values
    # for i,country in enumerate(plot_regions):
    #     if fix_y_axes == True:
    #         fig.axes[i].set_ylim([0,np.nanmax(max_cf)*fac])  
    #     elif type(fix_y_axes) == list:
    #         fig.axes[i].set_ylim(fix_y_axes)
    #     elif fix_y_axes == False:
    #         fig.axes[i].set_ylim([0,max_cf[i]*fac])  
    
    print('NOTE: If all the data is not within axis limits, adjust the set_ylim parameter')
    plt.show()
    if return_res:
        return fig,res_dict
    else:
        return fig