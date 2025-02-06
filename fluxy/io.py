import os
import glob
import xarray as xr
import numpy as np
import json 
import geopandas as gpd
import logging

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from pathlib import Path
from typing import Literal

from fluxy import config
from fluxy.operators.regions import extract_region_flux
from fluxy.operators.select import slice_flux

logger = logging.getLogger(__name__)

def read_json(filepath: os.PathLike) -> dict[str, dict]:
    """
    Reads json file.

    Args:
        filepath (str or Path):
            Path to json file including filename.
    Returns:
        json_data (dictionary of dictionaries):
            Dictionary with data read from filepath.
    """

    filepath = Path(filepath)

    if not filepath.is_file():
        raise FileNotFoundError(f'Cannot find {filepath}.')

    with open(filepath, "r") as f:
        json_data = json.load(f)

    return json_data

def read_config_files() -> dict[str, dict]:
    """
    Reads all configuration json files.

    Returns:
        data_dict (dictionary of dictionaries):
            Dictionary with keys equal to json basename (without extension).
            Each key points to a dictionary with the data from each json file.
    """

    # Get location of json files
    parent_dir = Path(__file__).parent.parent
    configs_dir = parent_dir / 'configs'

    # List of json files to be read
    json_files = configs_dir.glob('*.json')

    # Read json files
    data_dict = {}
    for file in json_files:
        data =  read_json(file)
        filename = file.stem
        data_dict[filename] = data

    return data_dict

def get_filename(model, specie, period, file_pattern, config_data, data_dir):

    # Get file name tags
    name_tags = model.split('_')
    model_name = name_tags[0]
    param_tags = name_tags[3:]

    # Replace parameter tags by dict values in config
    filename_tags = config_data["models_info"].get("filename_tags", None)
    if filename_tags is not None:
        for i,param in enumerate(param_tags):
            string_in_file = filename_tags.get(param, None)

            if string_in_file is not None:
                string_in_file = string_in_file.replace("<model>", model_name.lower())
                name_tags[i+3] = string_in_file
                
    # Add domain to filename tags
    name_tags.insert(2, config_data["models_info"]["domain"])

    # Build filename
    model_filename = "_".join(name_tags)

    # Get species name
    specie_print = specie
    if (species_names := config_data["models_info"].get("species_name")) and \
       (model_species := species_names.get(model_name)) and \
       (species_tag   := model_species.get(specie)):
        specie_print = species_tag
            
    # Define filepath
    data_dir = Path(data_dir)
    filepath = data_dir / model_name / specie / f'{model_filename}_{specie_print}_{period}{file_pattern}'
    
    return filepath

def read_model_output(
    data_dir: os.PathLike,
    file_type: Literal["concentration","flux"],
    specie: str,
    models: list[str],
    config_data: dict[str, dict],
    period: str | list[str] = 'yearly',
) -> dict[str, xr.Dataset]:
    """
    Extracts mole fraction or flux timeseries data from each model.

    Args:
        data_dir (str): 
            Path to top data directory.
        specie (str): 
            Gas species, e.g. 'ch4'.
        models (list of str): 
            Keys specifying model names, e.g. ['intem','elris']
        config_data (dict of dict):
            Dictionary with settings read from json file.
            Use json filenames as keys.
        period (str or list of str):
            Inversion period as specified in the model filename.
            If it is a string, the same period is considered for all models.
            If it is a list, one value per model must be specified, e.g. ['monthly','yearly']
    Returns:
        ds_all (dictionary of datasets): 
            xarray dataset read directly from each model's mole fraction netCDF.
    """
    
    if isinstance(period, str):
        period = [period]*len(models)
    
    if len(period) != len(models):
        raise ValueError(f'period must be a string or a list of the same length as models.')

    # Define file pattern
    if file_type == 'flux':
        file_pattern = '.nc'
    elif file_type == 'concentration':
        file_pattern = '_concentrations.nc'
    else:
        raise ValueError(f'file_pattern must be equal to "concentration" or "flux".')

    ds_all = {}

    for i,m in enumerate(models):
        filepath = get_filename(m, specie, period[i], file_pattern, config_data, data_dir)

        # Check if files exists
        if not filepath.is_file():
            logger.warning(f'Cannot find {file_type} file: {filepath}.')
            continue
 
        # Read file
        logger.info(f'Reading {file_type} file: {filepath}')
        ds_all[m] = xr.open_dataset(filepath)

        # Easy fix for InTEM ("units" attribute is wrongly set to "unit")
        vars_to_check = ['country_flux_total_prior', 'country_flux_total_posterior',
                         'percentile_country_flux_total_prior','percentile_country_flux_total_posterior']
        
        if file_type == 'flux':
            for var in vars_to_check:
                if 'units' not in ds_all[m][var].attrs.keys() and 'unit' in ds_all[m][var].attrs.keys():
                    ds_all[m][var].attrs['units'] = ds_all[m][var].attrs['unit']

    return ds_all

def read_flux_total_fgases(data_dir,species,models,config_data,regions,
                           start_date,end_date,period='yearly',apply_pop_scale=True):
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
        config_data (dict of dict):
            Dictionary with settings read from json file.
            Use json filenames as keys.
        start_date (str):
            Date to slice data from, e.g. '2021-01-01'
        end_date (str):
            Date to slice data to, e.g. '2022-01-01' would include all
            data up to 2021-12-31.
        period (str or list of str):
            Inversion period as specified in the model filename.
            If it is a string, the same period is considered for all models.
            If it is a list, one value per model must be specified, e.g. ['monthly','yearly']
                 
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

    if isinstance(period, str):
        period = [period]*len(models)
    
    if len(period) != len(models):
        raise ValueError(f'period must be a string or a list of the same length as models.')
        
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

            species_info = config_data['species_info'][species]
            
            #tries to read from standard filename
            try:
                # Get standard run name tag
                std_run_all = config_data["models_info"]["standard_run"]
                std_run = std_run_all["default"][species]
                if (model_std_run := std_run_all.get(m0)) and (run_name := model_std_run.get(species)):
                    std_run = run_name
                
                model_read = f'{m0}_{std_run}'
                if 'longrun' in model: model_read = f'{model_read}_longrun'
                
                ds_in[model] = read_model_output(data_dir,"flux",species,[model_read],config_data,period[m])[model_read]
                ds_in[model] = slice_flux({model:ds_in[model]},config_data,start_date[m],end_date[m])[model]

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
                    region_flux_total_posterior_lower = (region_flux_total_posterior-region_flux_total_posterior_lower) * 1e3 * species_info['gwp'] * 1e-12
                    region_flux_total_posterior_upper = (region_flux_total_posterior_upper-region_flux_total_posterior) * 1e3 * species_info['gwp'] * 1e-12
                    region_flux_total_prior_lower = (region_flux_total_prior-region_flux_total_prior_lower) * 1e3 * species_info['gwp'] * 1e-12
                    region_flux_total_prior_upper = (region_flux_total_prior_upper-region_flux_total_prior) * 1e3 * species_info['gwp'] * 1e-12
                    
                    region_flux_total_posterior = region_flux_total_posterior * 1e3 * species_info['gwp'] * 1e-12
                    region_flux_total_prior = region_flux_total_prior * 1e3 * species_info['gwp'] * 1e-12

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
                
            if m0 == 'InTEM':
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

def load_countries_shape(
    region_bounds: tuple =None
    ) -> gpd:
    """
    Load Natural Earth vector map data and optionally filters for a specific region.

    Args:
        region_bounds (tuple, optional):
            A tuple of (min_lon, max_lon, min_lat, max_lat) to filter the map.
            Default is None, which loads the full world.

    Returns:
        gdf (GeoDataFrame): 
            A GeoDataFrame containing the country boundaries for the specified region.
    """

    # Scale of the map (1:50m)
    res = "50m"  # Can be 10m, 50m, 110m

    this_file = Path(__file__).parent.parent
    path_to_save = this_file / "data" / "ne_data"
    url = f"https://naturalearth.s3.amazonaws.com/{res}_cultural/ne_{res}_admin_0_countries.zip"
    path_to_save.mkdir(parents=True, exist_ok=True)

    shpfile = path_to_save / f"ne_{res}_admin_0_countries.shp"

    if not shpfile.is_file():
        resp = urlopen(url)
        zipfile = ZipFile(BytesIO(resp.read()))
        zipfile.extractall(path_to_save)

    gdf = gpd.read_file(shpfile)

    # Update the missing ISO_A3 values in the data
    name_to_iso_a3_mapping = {
        'Norway': 'NOR',
        'Kosovo': 'KOS',
        'France': 'FRA',
        'Indian Ocean Ter.': 'IOT',
    }

    for name, iso_a3 in name_to_iso_a3_mapping.items():
        gdf.loc[gdf['NAME'] == name, 'ISO_A3'] = iso_a3

    # If a region is specified, filter the GeoDataFrame
    if region_bounds:
        min_lon, max_lon, min_lat, max_lat = region_bounds
        gdf = gdf.cx[min_lon:max_lon, min_lat:max_lat]

    return gdf