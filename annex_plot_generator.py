import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PARIS_inversion_results as func

###########################################
### GENERAL SETTINGS
###########################################
# Species to plot
monthly_species = ['ch4','n2o']

annual_species = ['hfc23','hfc32','hfc125','hfc134a','hfc143a','hfc152a',
                  'hfc227ea','hfc245fa','hfc365mfc','hfc4310mee',
                  'cf4','pfc116','pfc218','pfc318','sf6']

combined_species = ['all_hfc','all_pfc']

# Cities to plot
point_markers = {'UK': ['london','edinburgh','cardiff','belfast'],
                'SWITZERLAND': ['bern','zurich','geneva','basel','lausanne'],
                'GERMANY': ['berlin','hamburg','munich','koeln','frankfurt','essen'],
                'ITALY': ['rome','milan','naples','turin','palermo'],
                'NETHERLANDS': ['amsterdam','rotterdam','hague','utrecht','eindhoven'],
                'IRELAND': ['dublin','cork','limerick','galway','waterford'],
                'HUNGARY': ['budapest','debrecen','miskolc','szeged','pecs'],
                'NORWAY': ['oslo','bergen','sandnes','stavanger','drammen'],
                'BELGIUM': ['brussels','antwerp','ghent','charleroi','liege']}

point_markers['BENELUX'] = point_markers['NETHERLANDS'] + point_markers['BELGIUM'] + ['luxembourg']

# Path to results directory 
data_dir = '/project/paris/inverse_modelling/'

# Set ppt_mode to True for bigger fonts
ppt_mode = False

# Set annex_mode to True for shorter labels
annex_mode = True

# Start date of F-gases country fluxes
start_date_fgases = {
    'UK': '2008-01-01',
    'SWITZERLAND': '2008-01-01',
    'GERMANY': '2013-01-01',
    'ITALY': '2008-01-01',
    'NETHERLANDS': '2013-01-01',
    'BELGIUM': '2013-01-01',
    'BENELUX': '2013-01-01',
    'IRELAND': '2008-01-01',
    'HUNGARY': '2018-01-01',
    'NORWAY': '2018-01-01'
}

# Specify the percentile to use for the color scales in the flux spatial map
fluxlim_percentiles = {
    'UK': {
        'ch4': 0.95, 'n2o': 0.95, 'hfc32': 0.99, 'hfc125': 0.99, 'hfc134a': 0.99, 'hfc143a': 0.99,
        'cf4': 0.99, 'pfc116': 0.95, 'pfc218': 0.99, 'pfc318': 0.95, 'sf6': 0.99
    },
    'SWITZERLAND': {
        'ch4': 0.96, 'n2o': 0.96, 'hfc32': 0.98, 'hfc125': 0.98, 'hfc134a': 0.97, 'hfc143a': 0.97,
        'cf4': 0.98, 'pfc116': 0.98, 'pfc218': 0.975, 'pfc318': 0.96, 'sf6': 0.99
    },
    'GERMANY': {
        'ch4': 0.97, 'n2o': 0.97, 'hfc32': 0.99, 'hfc125': 0.99, 'hfc134a': 0.99, 'hfc143a': 0.99,
        'cf4': 0.995, 'pfc116': 0.995, 'pfc218': 0.98, 'pfc318': 0.99, 'sf6': 0.99
    },
    'ITALY': {
        'ch4': 0.95, 'n2o': 0.95, 'hfc32': 0.99, 'hfc125': 0.99, 'hfc134a': 0.99, 'hfc143a': 0.99,
        'cf4': 0.99, 'pfc116': 0.99, 'pfc218': 0.95, 'pfc318': 0.99, 'sf6': 0.95
    },
    'NETHERLANDS': {
        'ch4': 0.96, 'n2o': 0.97, 'hfc32': 0.97, 'hfc125': 0.97, 'hfc134a': 0.97, 'hfc143a': 0.96,
        'cf4': 0.99, 'pfc116': 0.99, 'pfc218': 0.97, 'pfc318': 0.99, 'sf6': 0.99
    },
    'BELGIUM': {
        'ch4': 0.95, 'n2o': 0.97, 'hfc32': 0.95, 'hfc125': 0.95, 'hfc134a': 0.94, 'hfc143a': 0.94,
        'cf4': 0.99, 'pfc116': 0.99, 'pfc218': 0.94, 'pfc318': 0.99, 'sf6': 0.98
    },
    'BENELUX': {
        'ch4': 0.96, 'n2o': 0.97, 'hfc32': 0.98, 'hfc125': 0.97, 'hfc134a': 0.97, 'hfc143a': 0.97,
        'cf4': 0.9925, 'pfc116': 0.99, 'pfc218': 0.97, 'pfc318': 0.99, 'sf6': 0.99
    },
    'IRELAND': {
        'ch4': 0.95, 'n2o': 0.95, 'hfc32': 0.95, 'hfc125': 0.95, 'hfc134a': 0.95, 'hfc143a': 0.95,
        'cf4': 0.99, 'pfc116': 0.99, 'pfc218': 0.95, 'pfc318': 0.95, 'sf6': 0.95
    },
    'HUNGARY': {
        'ch4': 0.99, 'n2o': 0.99, 'hfc32': 0.97, 'hfc125': 0.95, 'hfc134a': 0.95, 'hfc143a': 0.95,
        'cf4': 0.99, 'pfc116': 0.995, 'pfc218': 0.96, 'pfc318': 0.965, 'sf6': 0.98
    },
    'NORWAY': {
        'ch4': 0.95, 'n2o': 0.95, 'hfc32': 0.95, 'hfc125': 0.95, 'hfc134a': 0.95, 'hfc143a': 0.95,
        'cf4': 0.95, 'pfc116': 0.95, 'pfc218': 0.95, 'pfc318': 0.95, 'sf6': 0.95
    }
}

###########################################

def produce_plots(regions, output_path, inventory_years):

    ### Settings for country fluxes
    models_country_fluxes = ['intem_longrun', 'intem', 'elris', 'rhime'] # NOTE: only options are basic model names w/ and w/o longrun
    scale_co2eq = True
    plot_inventory = True
    fix_y_axes = False
    add_prior = True
    add_prior_unc = False
    set_global_leg = False
    country_codes_as_titles = None
    #plot_separate = [True,False,False,False] # NOTE: easy fix while there are no Rhime results for N2O
    #plot_combined = [False,True,True,True]
    plot_resample_and_original = False
    period_override = None

    ### Settings for spatial maps
    models_spatial_maps = ['intem','elris','rhime']
    plot_area = regions[0]
    plot_site_locations = True
    plot_point_markers = point_markers[regions[0]]
    convert_flux_units = True
    set_fluxlim = 'auto'
    plot_inversion_grid_flux = True

    ### Initialization
    s_data,m_data,m_colors,annotate_coords = func.initialize_settings(ppt_mode)
    annual_res_list = list()

    ### Models for country fluxes
    models = models_country_fluxes

    ### CH4 and N2O
    print('\n--- PLOTTING COUNTRY FLUXES FOR CH4/N2O ---')
    for species in monthly_species:

        # NOTE: easy fix while there are no Rhime results for N2O
        if species == 'n2o':
            models = ['intem_longrun', 'intem', 'elris']
            plot_separate = [True,False,False]
            plot_combined = [False,True,True]
        else:
            models = models_country_fluxes
            plot_separate = [True,False,False,False]
            plot_combined = [False,True,True,True]

        # Long time window
        start_date = '2008-01-01'
        end_date   = '2024-01-01'

        ds_all_flux = {}
        ds_all_flux_scaled = {}
        models_std = []

        ### Read and scale fluxes
        for m,model in enumerate(models):

            m0 = model.split('_')[0]

            model_read = f'{m0}_{s_data[species]["std_run"][m0]}'
            if 'longrun' in model: model_read = f'{model_read}_longrun'
            models_std.append(model_read)

            # use model_read instead of model
            ds_all_flux[model_read] = func.read_flux(data_dir,species,[model_read],s_data,m_data,period_override=period_override)[model_read]
            ds_all_flux_scaled[model_read] = func.slice_flux({model_read:ds_all_flux[model_read]},start_date,end_date,s_data,scale_units=True,
                                                             scale_co2eq=scale_co2eq,species=species)[model_read]

        ### Define plotting colors
        model_colors = func.set_model_colors(models_std,m_colors)

        # Annual averages
        resample = 'year'
        resample_uncert_correlation = False #recalculates uncertainties assuming no correlation

        # 1.1) Plot annual country fluxes from 2008 to 2023 from intem_longrun and combined from 3 std_run
        fig = func.plot_country_flux(ds_all_flux_scaled,species,regions,
                                     s_data,m_data,model_colors,start_date,end_date,ppt_mode,annex_mode,scale_co2eq,
                                     plot_inventory,inventory_years,data_dir,fix_y_axes,add_prior,
                                     add_prior_unc,set_global_leg,country_codes_as_titles=country_codes_as_titles,
                                     plot_separate=plot_separate,plot_combined=plot_combined,
                                     resample=resample,resample_uncert_correlation=resample_uncert_correlation,
                                     plot_resample_and_original=plot_resample_and_original,
                                     period_override=period_override)

        plot_name = f'{species}_country_flux_annual_longrun_{regions[0]}.png'
        full_path = os.path.join(output_path, plot_name)
        fig.savefig(full_path,bbox_inches='tight',pad_inches=0.2,dpi=300)
        plt.close()

        # PARIS time window
        start_date = '2018-01-01'
        end_date   = '2024-01-01'

        ### Re-slice the data
        for model_read in models_std:
            ds_all_flux_scaled[model_read] = func.slice_flux({model_read:ds_all_flux[model_read]},start_date,end_date,s_data,scale_units=True,
                                                             scale_co2eq=scale_co2eq,species=species)[model_read]

        # 1.2) Plot annual country fluxes from 2018 to 2023 from intem_longrun and combined from 3 std_run
        fig,res_dict = func.plot_country_flux(ds_all_flux_scaled,species,regions,
                                              s_data,m_data,model_colors,start_date,end_date,ppt_mode,annex_mode,scale_co2eq,
                                              plot_inventory,inventory_years,data_dir,fix_y_axes,add_prior,
                                              add_prior_unc,set_global_leg,country_codes_as_titles=country_codes_as_titles,
                                              plot_separate=plot_separate,plot_combined=plot_combined,
                                              resample=resample,resample_uncert_correlation=resample_uncert_correlation,
                                              plot_resample_and_original=plot_resample_and_original,
                                              period_override=period_override,
                                              return_res=True)

        plot_name = f'{species}_country_flux_annual_parisonly_{regions[0]}.png'
        full_path = os.path.join(output_path, plot_name)
        fig.savefig(full_path,bbox_inches='tight',pad_inches=0.2,dpi=300)
        plt.close()
        
        # Store results for .csv and table
        comb = res_dict[regions[0]]['combined']
        try:
            inv = res_dict[regions[0]]['inventory']
        except:
            print('NO INVENTORY FOUND')
            inv = {'time':comb['time'],
                   'value':np.array([np.NaN,]*len(comb['time']))}
            
        tmp = {'species':[species,]*2,'source':['NIR '+inventory_years[0],'PARIS mean']}
        for it,time in enumerate(comb['time'].astype('datetime64[Y]')):
            paris_val = f"{comb['mean'][it]:.0f} \\pm {(comb['max'][it]-comb['min'][it])/2:.0f}"
            inv_val = inv['value'][inv['time'].astype('datetime64[Y]')==time]
            if len(inv_val)==1:
                tmp[str(time)] = [f'{inv_val[0]:.0f}',paris_val]                            
            else:
                 tmp[str(time)] = [None,paris_val]      
        annual_res_list.append(pd.DataFrame(tmp))

        # Monthly country fluxes
        resample = None
        resample_uncert_correlation = False

        # 2) Plot monthly country fluxes from 2018 to 2023 from intem_longrun and combined from 3 std_run
        fig = func.plot_country_flux(ds_all_flux_scaled,species,regions,
                                     s_data,m_data,model_colors,start_date,end_date,ppt_mode,annex_mode,scale_co2eq,
                                     plot_inventory,inventory_years,data_dir,fix_y_axes,add_prior,
                                     add_prior_unc,set_global_leg,country_codes_as_titles=country_codes_as_titles,
                                     plot_separate=plot_separate,plot_combined=plot_combined,
                                     resample=resample,resample_uncert_correlation=resample_uncert_correlation,
                                     plot_resample_and_original=plot_resample_and_original,
                                     period_override=period_override)

        plot_name = f'{species}_country_flux_monthly_parisonly_{regions[0]}.png'
        full_path = os.path.join(output_path, plot_name)
        fig.savefig(full_path,bbox_inches='tight',pad_inches=0.2,dpi=300)
        plt.close()
        
    ### Total HFCs/PFCs (w/o HFC-4310mee)
    resample = None
    resample_uncert_correlation = False   
    rolling_mean = True      
    start_date = [start_date_fgases[plot_area],'2018-01-01','2018-01-01','2018-01-01']
    end_date   = ['2024-01-01','2024-01-01','2024-01-01','2024-01-01']

    # NOTE: easy fix while there are no Rhime results for N2O
    models = models_country_fluxes
    plot_separate = [True,False,False,False]
    plot_combined = [False,True,True,True]

    print('\n--- PLOTTING COUNTRY FLUXES FOR TOTAL HFC/PFC ---')
    for species in combined_species:

        ds_all_flux_scaled = {}

        ### Read and scale fluxes
        ds_all_flux_scaled = func.read_flux_total_fgases(data_dir,species,models,s_data,m_data,
                                                        regions,start_date,end_date,
                                                        period_override=period_override)

        ### Define plotting colors
        model_colors = func.set_model_colors(models,m_colors)

        # 3) Plot annual country fluxes from 2008 to 2023 from intem_longrun and combined from 3 std_run
        fig,res_dict = func.plot_country_flux(ds_all_flux_scaled,species,regions,
                                              s_data,m_data,model_colors,start_date,end_date,ppt_mode,annex_mode,scale_co2eq,
                                              plot_inventory,inventory_years,data_dir,fix_y_axes,add_prior,
                                              add_prior_unc,set_global_leg,country_codes_as_titles=country_codes_as_titles,
                                              plot_separate=plot_separate,plot_combined=plot_combined,
                                              resample=resample,resample_uncert_correlation=resample_uncert_correlation,
                                              plot_resample_and_original=plot_resample_and_original,
                                              period_override=period_override,
                                              return_res=True,
                                              rolling_mean=rolling_mean)

        plot_name = f'{species}_country_flux_annual_longrun_{regions[0]}.png'
        full_path = os.path.join(output_path, plot_name)
        fig.savefig(full_path,bbox_inches='tight',pad_inches=0.2,dpi=300)
        plt.close()
                              
        # Store results for .csv and table
        comb = res_dict[regions[0]]['combined']
        try:
            inv = res_dict[regions[0]]['inventory']
        except:
            print('NO INVENTORY FOUND')
            inv = {'time':comb['time'],
                   'value':np.array([np.NaN,]*len(comb['time']))}
            
        if "pfc" in species:
            n_digits = 2
        else:
            n_digits = 1

        tmp = {'species':[species,]*2,'source':['NIR '+inventory_years[0],'PARIS mean']}
        for it,time in enumerate(comb['time'].astype('datetime64[Y]')):
            paris_val = f"{comb['mean'][it]:.{n_digits}f} \\pm {(comb['max'][it]-comb['min'][it])/2:.{n_digits}f}"
            inv_val = inv['value'][inv['time'].astype('datetime64[Y]')==time]
            if len(inv_val)==1:
                tmp[str(time)] = [f'{inv_val[0]:.{n_digits}f}',paris_val]
            else:
                 tmp[str(time)] = [None,paris_val]      
        annual_res_list.append(pd.DataFrame(tmp))

    ### F-gases
    end_date   = '2024-01-01'
    resample   = None
    resample_uncert_correlation = False
    rolling_mean = True

    print('\n--- PLOTTING COUNTRY FLUXES FOR ALL F-GASES ---')
    for species in annual_species:

        ds_all_flux = {}
        ds_all_flux_scaled = {}
        models_std = []

        start_date = start_date_fgases[plot_area]
        start_year = start_date.split('-')[0]
        if species == 'hfc4310mee' and int(start_year) < 2011:
            start_date = '2011-01-01' # Fix for InTEM longrun which is zero in 2010

        ### Read and scale fluxes
        for m,model in enumerate(models):

            m0 = model.split('_')[0]

            model_read = f'{m0}_{s_data[species]["std_run"][m0]}'
            if 'longrun' in model: model_read = f'{model_read}_longrun'
            models_std.append(model_read)

            # use model_read instead of model
            ds_all_flux[model_read] = func.read_flux(data_dir,species,[model_read],s_data,m_data,period_override=period_override)[model_read]
            ds_all_flux_scaled[model_read] = func.slice_flux({model_read:ds_all_flux[model_read]},start_date,end_date,s_data,scale_units=True,
                                                             scale_co2eq=scale_co2eq,species=species)[model_read]

        ### Define plotting colors
        model_colors = func.set_model_colors(models_std,m_colors)

        # 4) Plot annual country fluxes from 2008 to 2023 from intem_longrun and combined from 3 std_run
        fig,res_dict = func.plot_country_flux(ds_all_flux_scaled,species,regions,
                                              s_data,m_data,model_colors,start_date,end_date,ppt_mode,annex_mode,scale_co2eq,
                                              plot_inventory,inventory_years,data_dir,fix_y_axes,add_prior,
                                              add_prior_unc,set_global_leg,country_codes_as_titles=country_codes_as_titles,
                                              plot_separate=plot_separate,plot_combined=plot_combined,
                                              resample=resample,resample_uncert_correlation=resample_uncert_correlation,
                                              plot_resample_and_original=plot_resample_and_original,
                                              period_override=period_override,
                                              return_res=True,
                                              rolling_mean=rolling_mean)

        plot_name = f'{species}_country_flux_annual_longrun_{regions[0]}.png'
        full_path = os.path.join(output_path, plot_name)
        fig.savefig(full_path,bbox_inches='tight',pad_inches=0.2,dpi=300)
        plt.close()
        
        # Store results for .csv and table
        comb = res_dict[regions[0]]['combined']
        try:
            inv = res_dict[regions[0]]['inventory']
        except:
            print('NO INVENTORY FOUND')
            inv = {'time':comb['time'],
                   'value':np.array([np.NaN,]*len(comb['time']))}
            
        tmp = {'species':[species,]*2,'source':['NIR '+inventory_years[0],'PARIS mean']}

        if species in ['hfc23','hfc32','hfc125','hfc134a','hfc143a','sf6']:
            n_digits = 2
        else:
            n_digits = 3

        for it,time in enumerate(comb['time'].astype('datetime64[Y]')):
            paris_val = f"{comb['mean'][it]:.{n_digits}f} \\pm {(comb['max'][it]-comb['min'][it])/2:.{n_digits}f}"
            inv_val = inv['value'][inv['time'].astype('datetime64[Y]')==time]
            if len(inv_val)==1:
                tmp[str(time)] = [f'{inv_val[0]:.{n_digits}f}',paris_val]                            
            else:
                 tmp[str(time)] = [None,paris_val]      
        annual_res_list.append(pd.DataFrame(tmp))

    ### Models for spatial maps
    models = models_spatial_maps

    # Settings for average posterior
    cmap = 'viridis'
    c_border = 'floralwhite'
    var = 'flux_total_posterior'
    plot_combined = True
    chop_by = 'year'

    start_date  = '2018-01-01'
    all_species = monthly_species + annual_species

    # All species
    print('\n--- PLOTTING MEAN POSTERIOR MAP FOR ALL SPECIES ---')
    for species in all_species:

        ds_all_flux = {}
        ds_all_flux_scaled = {}
        models_std = []
        
        if species in fluxlim_percentiles[plot_area].keys():
            set_fluxlim_percentile = fluxlim_percentiles[plot_area][species]
        else:
            set_fluxlim_percentile = None

        if species == "hfc4310mee":
            end_date = '2023-01-01' # NOTE: no 2023 results for HFC-4310mee
            dt = 5
        else:
            end_date = '2024-01-01'
            dt = 6

        # NOTE: easy fix while there are no Rhime results for N2O
        if species == 'n2o':
            models = ['intem', 'elris']
        else:
            models = models_spatial_maps

        ### Read and scale fluxes
        for m,model in enumerate(models):

            m0 = model.split('_')[0]

            model_read = f'{m0}_{s_data[species]["std_run"][m0]}'
            models_std.append(model_read)

            # use model_read instead of model
            ds_all_flux[model_read] = func.read_flux(data_dir,species,[model_read],s_data,m_data,period_override=period_override)[model_read]
            ds_all_flux_scaled[model_read] = func.slice_flux({model_read:ds_all_flux[model_read]},start_date,end_date,s_data,scale_units=True,
                                                             convert_flux_units=convert_flux_units,species=species)[model_read]

        # 5) Plot spatial map of the posterior fluxes averaged between 2018 and 2023 (combined from 3 std_run)
        fig = func.plot_spatial_flux_per_timestamp(ds_all_flux_scaled,species,plot_area,end_date,s_data,m_data,
                                                    cmap=cmap,c_border=c_border,var=var,
                                                    plot_combined=plot_combined, annex_mode=annex_mode,
                                                    chop_by=chop_by,dt=dt,period_override=period_override,
                                                    plot_site_locations=plot_site_locations,
                                                    plot_point_markers=plot_point_markers,
                                                    set_fluxlim=set_fluxlim, set_fluxlim_percentile=set_fluxlim_percentile,
                                                    plot_inversion_grid_flux=plot_inversion_grid_flux)

        plot_name = f'{species}_posterior_map_{regions[0]}.png'
        full_path = os.path.join(output_path, plot_name)
        fig.savefig(full_path,bbox_inches='tight',pad_inches=0.2,dpi=300)
        plt.close()

    # Settings for seasonal difference to the mean
    cmap = 'coolwarm'
    c_border = 'dimgrey'
    chop_by = 'season'
    dt = [[12,1,2],[3,4,5],[6,7,8],[9,10,11]]
    var = 'posterior_mean_diff'
    plot_combined = True

    # CH4 and N2O
    start_date = '2018-01-01'
    end_date   = '2024-01-01'

    print('\n--- PLOTTING SEASONAL POSTERIOR MAP FOR CH4/N2O ---')
    for species in monthly_species:

        ds_all_flux = {}
        ds_all_flux_scaled = {}
        models_std = []

        if species in fluxlim_percentiles[plot_area].keys():
            set_fluxlim_percentile = fluxlim_percentiles[plot_area][species]
        else:
            set_fluxlim_percentile = None
        
        # NOTE: easy fix while there are no Rhime results for N2O
        if species == 'n2o':
            models = ['intem', 'elris']
        else:
            models = models_spatial_maps

        ### Read and scale fluxes
        for m,model in enumerate(models):

            m0 = model.split('_')[0]

            model_read = f'{m0}_{s_data[species]["std_run"][m0]}'
            models_std.append(model_read)

            # use model_read instead of model
            ds_all_flux[model_read] = func.read_flux(data_dir,species,[model_read],s_data,m_data,period_override=period_override)[model_read]
            ds_all_flux_scaled[model_read] = func.slice_flux({model_read:ds_all_flux[model_read]},start_date,end_date,s_data,scale_units=True,
                                                             convert_flux_units=convert_flux_units,species=species)[model_read]

        # 6) Plot spatial maps of the seasonal posterior fluxes (averaged between 2018 and 2023) subtracted by the mean (combined from 3 std_run)
        fig = func.plot_spatial_flux_per_timestamp(ds_all_flux_scaled,species,plot_area,end_date,s_data,m_data,
                                                    cmap=cmap,c_border=c_border,var=var,
                                                    plot_combined=plot_combined,annex_mode=annex_mode,
                                                    chop_by=chop_by,dt=dt,period_override=period_override,
                                                    plot_site_locations=plot_site_locations,
                                                    plot_point_markers=plot_point_markers,
                                                    set_fluxlim = set_fluxlim, set_fluxlim_percentile=set_fluxlim_percentile,
                                                    plot_inversion_grid_flux=plot_inversion_grid_flux)

        plot_name = f'{species}_seasonal_map_{regions[0]}.png'
        full_path = os.path.join(output_path, plot_name)
        fig.savefig(full_path,bbox_inches='tight',pad_inches=0.2,dpi=300)
        plt.close()

    print('\n--- ALL PLOTS GENERATED SUCCESSFULLY! ---')


    print('\n\n\n--- GENERATING TABLES ---')
    annual_res = pd.concat(annual_res_list).reset_index(drop=True).fillna(value=' ')
    
    print('\n\nTABLE HFC\n\n')
    hfc_res = annual_res[annual_res.species.apply(lambda x : x[:3].lower()=='hfc')].copy()
    hfc_res['species'] = hfc_res.species.apply(lambda x : x.replace('hfc','HFC-'))
    make_table(hfc_res,f'{output_path}/hfc_res_{regions[0]}.tex')
    hfc_res.to_csv(f'{output_path}/hfc_res_{regions[0]}.csv',index=False)
    
    print('\n\nTABLE PFC\n\n')
    pfc_res = annual_res[annual_res.species.apply(lambda x : x[:3].lower()=='pfc' or x.lower()=='cf4')].copy()
    pfc_res['species'] = pfc_res.species.apply(lambda x : x.replace('pfc','PFC-'))
    pfc_res['species'] = pfc_res.species.apply(lambda x : x.replace('cf4','PFC-14'))
    make_table(pfc_res,f'{output_path}/pfc_res_{regions[0]}.tex')
    pfc_res.to_csv(f'{output_path}/pfc_res_{regions[0]}.csv',index=False)
    
    print('\n\nTABLE main gases\n\n')
    
    main_gases_res = annual_res[annual_res.species.isin(['ch4','n2o','sf6','all_pfc','all_hfc'])].copy()
    main_gases_res['species'] = main_gases_res.species.apply(lambda x : x.upper().replace('ALL_','Total '))
    make_table(main_gases_res,f'{output_path}/main_gases_res_{regions[0]}.tex')
    main_gases_res.to_csv(f'{output_path}/main_gases_res_{regions[0]}.csv',index=False)
    
                              
    print('\n--- TABLES GENERATED SUCCESSFULLY! ---')
    return annual_res


def make_table(df,output_path,
               descriptive_cols=['species','source'],
               hline_place={'source':'PARIS mean'}
              ):
    if 'hfc' in output_path:
        species = 'HFCs'
    elif 'pfc' in output_path:
        species = 'PFCs'
    if 'main_gases' in output_path:
        species = 'the main greenhouse gases of focus'
    # Set latex Table env and number of cols
    tmp = output_path.split('/')[-1].split('.')[0]
    label = '\n \\label{'+tmp+'}'
    tmp = "Emissions estimation for "+species+" in $\\rm{TgCO}_{2}\\rm{-eq} \\cdot \\rm{yr}^{-1}$ according to the National Inventory Report (NIR) 2024 and the inversions done in the PARIS project. For the PARIS estimation, the mean of the 3 inversion models is displayed, along with a range of uncertainty estimated via the half distance between the maximum and minimum uncertainties of the different models."  
    caption = '\n \\caption{'+tmp+'}'
    begin = '\\begin{table}[H]\n \\small'+label+caption+'\n \\begin{center}\n  \\begin{tabular}{ '+len(descriptive_cols)*'l '+(len(df.columns)-len(descriptive_cols))*'l '+'}'
    
    # Set first line with columns title
    header = '     '+len(descriptive_cols)*' & '
    for y in df.columns[len(descriptive_cols):]:
        header += y
        if y!=df.columns[-1]: 
            header+=' & '
    
    table = begin + '\n' + header + ' \\\\ \hline' + '\n'
    
    # Iterate over lines of dataframe
    prev_species = ''
    for idRow,row in df.iterrows():
        # Indentation
        l = '    '
        
        # Test if value for first column needed
        if row[descriptive_cols[0]]==prev_species:
            l += ' & '
        else:
            l += row[descriptive_cols[0]]+' & '
        prev_species = row[descriptive_cols[0]]
        
        # Add values for other descriptive columns
        for col in descriptive_cols[1:]:
            l += row[col]+' & '
        
        # Add yearly values
        for y in df.columns[len(descriptive_cols):]:
            l += '$ '+row[y]+' $'
            if y!=df.columns[-1]: 
                l+=' & '
        
        # End line
        l += ' \\\\ '
        
        # Add hline if needed
        for key in hline_place.keys():
            if row[key]==hline_place[key]:
                l += ' \hline '
        
        # Add line to table
        table+= l+'\n '
    
    # Close latex env
    end = str('  \\end{tabular}\n \\end{center}\n\\end{table}')
    
    table += end
    
    with open(output_path, "w") as text_file:
        text_file.write(table)
