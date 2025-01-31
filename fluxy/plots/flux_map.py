import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd

from pandas import to_datetime

from fluxy.io import load_countries_shape
from fluxy import config
from fluxy.plots.utils import extract_site_info, set_flux_limits, print_cbar_label, get_marker_coordinates, get_sites_coordinates, get_region_coordinates, add_site_markers, add_custom_markers, add_colorbar
from fluxy.operators.flux import get_flux_mean

def plot_flux_map(
    ds_all: dict[xr.Dataset], 
    species: str, 
    region: str | list[float], 
    config_data: dict, 
    cmap: str = 'viridis',
    cmap_diff: str = 'coolwarm', 
    c_border: str = 'floralwhite', 
    period_override: list = None,
    add_sites: bool = False, 
    add_markers: list[str] | list[list[float]] = None,
    season: str = None, 
    set_fluxlim: str | tuple = 'default', 
    set_fluxlim_percentile: float = None,
    plot_inversion_grid_flux: bool = False, 
    zoom_degree: float = 1,
    ) -> plt.Figure:
    """
    Plot posterior and prior fluxes and the difference between them for all models, time averaged.
    
    Args:
        ds_all (dictionary of datasets):
            xarray datasets of fluxes, scaled and sliced between 
            chosen dates.
        species (str): 
            Gas species, e.g. 'ch4'.
        region (str or list):
            Lat/lon region to plot, options for 'UK', 'FRANCE', 'GERMANY',
            'NWEU','CWEU','EUROPE'.
            A list with [min_lon, max_lon, min_lat, max_lat] can also be provided.
        config_data (dict of dict):
            Dictionary of models and species with information for plotting (read from json file).
        cmap (str):
            Colour map for flux plots.
        cmap_diff (str):
            Colour map for difference plots.
        c_border (str):
            Colour for flux plot country borders.
        period_override (list of str, optional):
            Inversion periods to include, to override the standards in species_info.json.
            Must be the same length as models, e.g. ['monthly',None,'yearly']
        add_sites (bool):
            If True, adds triangles with site locations to spatial plot.
        add_markers (list of str or list of lat/lon):
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

    # Determine geographical boundaries
    map_bounds = (
        get_region_coordinates(region, zoom_degree=zoom_degree)
        if isinstance(region, str)
        else tuple(region)
        if isinstance(region, list) and all(isinstance(coord, (int, float)) for coord in region)
        else ValueError("Invalid input: 'region' must be a string or a list of numbers.")
    )

    # Get species and models info
    species_info = config_data['species_info'][species]
    models_info = config_data['models_info']

    # Load geographical data and sites information
    gdf = load_countries_shape(map_bounds)
    sites_info = get_sites_coordinates(ds_all, config_data) if add_sites else "" #TODO move in the for loop once the info comes from the concentration files

    # Set flux limits #TODO Based on 'flux_total_posterior', is this the right way to do?
    fluxlim = set_flux_limits(
        ds_all, "flux_total_posterior", map_bounds, species_info, option=set_fluxlim, custom_percentile=set_fluxlim_percentile
    )
    difflim = (-fluxlim[1], fluxlim[1])

    # Initialize figure
    n_cols = len(ds_all)
    fig, ax = plt.subplots(3, n_cols, constrained_layout=True, figsize=(n_cols * 5, 9))

    for i, (model, ds) in enumerate(ds_all.items()):
        # Extract longitude and latitude
        lon, lat = ds.longitude, ds.latitude

        # Define and compute variables
        var_prior = "flux_total_prior"
        var_posterior = "flux_total_posterior_inversion_grid" if plot_inversion_grid_flux else "flux_total_posterior"

        prior = get_flux_mean(ds[var_prior], season)
        posterior = get_flux_mean(ds[var_posterior], season)
        diff = np.nan_to_num(posterior - prior)

        # Determine colorbar label (once, on the last iteration) #TODO Here, based on the last iteration. Check if consistent for all models?
        cbar_label = (
            print_cbar_label(ds, species_info, var_posterior, season, period_override[i] if period_override else None)
            if i == len(ds_all) - 1
            else ""
        )

        # Plot each variable (prior, posterior, and diff)
        vars_to_plot = [prior, posterior, diff]
        var_names = ["Prior", "Posterior", "Posterior - Prior"]
        
        for j, var in enumerate(vars_to_plot):
            ax_ji = ax[j] if n_cols == 1 else ax[j, i]

            # Determine plot settings
            is_diff = (j == 2)
            cmap_j = cmap_diff if is_diff else cmap
            vlim_j = difflim if is_diff else fluxlim
            marker_color = "black" if is_diff else "red"
            border_color = "dimgrey" if is_diff else c_border
            extend_j = "both" if is_diff else "max"

            # Plot the data
            im = ax_ji.pcolormesh(lon, lat, var, cmap=cmap_j, vmin=vlim_j[0], vmax=vlim_j[1], shading="nearest")
            gdf.boundary.plot(ax=ax_ji, facecolor="none", edgecolor=border_color, linewidth=1)
            ax_ji.set_xlim(map_bounds[:2])  # Longitude limits
            ax_ji.set_ylim(map_bounds[2:])  # Latitude limits 

            # Add titles
            ax_ji.set_title(f"{models_info[model]['label']}", fontsize=12) if j==0 else "" # Column titles
            ax_ji.set_ylabel(f"{var_names[j]}", fontsize=12) if i==0 else "" # Row titles

            # Add colorbar (only for the last column)
            if i == len(ds_all) - 1:
                add_colorbar(fig, ax[j, ...], im, cmap_j, extend_j, cbar_label)

            # Add sites and markers if specified
            if add_sites and model in sites_info:
                add_site_markers(ax_ji, sites_info[model], marker_color)
            if add_markers:
                add_custom_markers(ax_ji, add_markers, marker_color)

    return fig

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
                   #subplot_kw={'projection':cartopy.crs.PlateCarree()}
                   )

    for i in range(3):
        if i == 2:
            border_color = 'dimgrey'
        else:
            border_color = c_border
        #ax[i].add_feature(cartopy.feature.BORDERS,edgecolor=border_color,linewidth=1.)
        # ax[i].coastlines(resolution='50m',color=border_color,linewidth=1.)
        # ax[i].set_extent(region_limits[plot_area])

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
        for p in plot_point_markers:
            if type(p) == list:
                ax[0].scatter(p[0],p[1],facecolor='none',edgecolor='red',marker='^',s=30,zorder=2)
                ax[1].scatter(p[0],p[1],facecolor='none',edgecolor='red',marker='^',s=30,zorder=2)
                ax[2].scatter(p[0],p[1],facecolor='none',edgecolor='black',marker='^',s=30,zorder=2)
            elif type(p) == str:
                if p not in config.point_source_dict.keys():
                    print(f'{p} is not specified in config.point_source_dict, edit this to add a lat/lon location.')
                else:
                    ax[0].scatter(config.point_source_dict[p][0],config.point_source_dict[p][1],
                                  facecolor='none',edgecolor='red',marker='^',s=30,zorder=2)
                    ax[1].scatter(config.point_source_dict[p][0],config.point_source_dict[p][1],
                                  facecolor='none',edgecolor='red',marker='^',s=30,zorder=2)
                    ax[2].scatter(config.point_source_dict[p][0],config.point_source_dict[p][1],
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

def plot_spatial_flux_per_timestamp(ds_all, species, plot_area, end_date, s_data, m_data,
                                    cmap='viridis',c_border='floralwhite',
                                    var='flux_total_posterior', plot_combined=False, annex_mode=False,
                                    chop_by='year', dt=1,period_override=None,
                                    plot_site_locations=False, plot_point_markers=False, set_fluxlim='default',
                                    set_fluxlim_percentile=None, plot_inversion_grid_flux=False):
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
                    'BELGIUM':[2,7,49,52.5],
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
                                  #subplot_kw={'projection':cartopy.crs.PlateCarree()}
                                  )
        ax = ax_tmp.flatten()
    else:
        fig,ax = plt.subplots(n_lines,n_cols,figsize=(n_cols*4,n_lines*3), #3.25
                       #subplot_kw={'projection':cartopy.crs.PlateCarree()}
                       )

    # Add map
    for i in range(n_lines):
        for j in range(n_cols):

            if n_cols == 1 and n_lines == 1:
                #ax.add_feature(cartopy.feature.BORDERS,edgecolor=c_border,linewidth=1.)
                # ax.coastlines(resolution='50m',color=c_border,linewidth=1.)
                # ax.set_extent(region_limits[plot_area])
                pass
            else:
                if n_cols == 1:
                    ax_var = ax[i]
                elif n_lines == 1:
                    ax_var = ax[j]
                else:
                    ax_var = ax[i,j]

                #ax_var.add_feature(cartopy.feature.BORDERS,edgecolor=c_border,linewidth=1.)
                # ax_var.coastlines(resolution='50m',color=c_border,linewidth=1.)
                # ax_var.set_extent(region_limits[plot_area])

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
                        slice_apriori = ds_all[m]['flux_total_prior'].sel(time=slice(t0_date[m][i],t1_date[m][i]))
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
                    for p in plot_point_markers:
                        if type(p) == list:
                            ax_var.scatter(p[0],p[1],facecolor='none',edgecolor='red',marker='^',s=30,zorder=2)
                        elif type(p) == str:
                            if p not in config.point_source_dict.keys():
                                print(f'{p} is not specified in config.point_source_dict, edit this to add a lat/lon location.')
                            else:
                                ax_var.scatter(config.point_source_dict[p][0],config.point_source_dict[p][1],facecolor='none',edgecolor='red',marker='^',s=30,zorder=2)

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
                for p in plot_point_markers:
                    if type(p) == list:
                        ax_var.scatter(p[0],p[1],facecolor='none',edgecolor='red',marker='^',s=30,zorder=2)
                    elif type(p) == str:
                        if p not in config.point_source_dict.keys():
                            print(f'{p} is not specified in config.point_source_dict, edit this to add a lat/lon location.')
                        else:
                            ax_var.scatter(config.point_source_dict[p][0],config.point_source_dict[p][1],facecolor='none',edgecolor='red',marker='^',s=30,zorder=2)

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
