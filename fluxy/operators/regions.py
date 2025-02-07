import glob
import os
import logging
import numpy as np
import pandas as pd
import xarray as xr

from fluxy import config
from fluxy.operators.select import get_units_conversion_factor

logger = logging.getLogger(__name__)


def extract_region_flux(
    ds_all: dict[str, xr.Dataset], country: str
) -> dict[str, xr.Dataset]:
    """
    Finds the index of a chosen region name and extracts the country flux
    variables for this region.
    Either extracts values directly from the dataset (if this region definition
    exists in the file) or calculates values by taking the sum of smaller regions
    (if this region definition does not exist in the file).

    Args:
        ds_all: xarray datasets of fluxes, scaled and sliced between 
            chosen dates.
        country: name of the country to extract.

    Returns:
        ds_output: dictionnary of datasets. The dataset variables are :
            - 'posterior',
            - 'prior',
            - 'posterior_lower',
            - 'posterior_upper',
            - 'prior_lower',
            - 'prior_upper'
    """
    ds_output = dict()

    for m, ds in ds_all.items():
        #########################################################################################
        # To be move to read_flux
        m0 = m.split("_")[0].lower()
        min_percentile_index = 0
        max_percentile_index = 1
        
        if m0 == 'elris':
            ds['country'] = ds['country'].astype('str')
            ds = ds.set_index(countrynumber='country').rename({'countrynumber':'country'})

        elif m0 == 'intem':            
            ds = ds.rename({'countrynumber':'country'})

            if 'BEL' not in ds.country and 'LUX'  not in ds.country:
                logger.warning(f"InTEM does not estimate separate BELGIUM emissions.\n A population ratio of {config.bel_pop_r} is being used to scale InTEM's total BELGIUM+LUXEMBOURG estimate.")

                r = config.bel_pop_r

                variables_with_country = [var for var in ds.data_vars if "country" in ds[var].dims]
                numerical_vars = [var for var in variables_with_country 
                                  if np.issubdtype(ds[var].dtype, np.number) and var != "country_fraction"]

                ds_bel = r * ds[numerical_vars].sel(country='BEL-LUX')
                ds_lux = (1-r) * ds[numerical_vars].sel(country='BEL-LUX')

                del ds_bel['country']
                del ds_lux['country']

                ds_bel['countryname'] = xr.DataArray(data = ['BELGIUM',] * ds_bel.time.size,
                                                    dims = ['time',],
                                                    coords = {'time': ds_bel.time},
                                                    attrs = ds.countryname.attrs)
                
                ds_lux['countryname'] = xr.DataArray(data = ['LUXEMBOURG',] * ds_lux.time.size,
                                                    dims = ['time',],
                                                    coords = {'time': ds_lux.time},
                                                    attrs = ds.countryname.attrs)
                
                ds_bellux = xr.concat([ds_bel, ds_lux], pd.Index(['BEL','LUX'], name='country'))
                ds = xr.merge([ds, ds_bellux])

        elif m0 == 'rhime':
            ds['country'] = [config.countrycodes_dict.get(x, x) for x in ds['country'].values]

        elif m0 == 'flexinvert':
            ds['percentile_country_flux_total_posterior'] = xr.concat([ds['country_flux_total_posterior']
                                                                    - ds['country_flux_error_posterior'],
                                                                    ds['country_flux_total_posterior']
                                                                    + ds['country_flux_error_posterior']],
                                                                    pd.Index([0,1], name = 'percentile'))
            
            ds['percentile_country_flux_total_prior'] = xr.concat([ds['country_flux_total_prior']
                                                                - ds['country_flux_error_prior'],
                                                                ds['country_flux_total_prior']
                                                                + ds['country_flux_error_prior']],
                                                                pd.Index([0,1], name = 'percentile'))
        #########################################################################################

        country_search = config.countrycodes_dict[country]
        # search for existing region names

        available_countries = ds["country"].values
        
        if country_search not in available_countries and country in config.regions_dict.keys() :
            region_search = config.regions_dict[country]

            logger.info(f'{country} emissions are not present in {m}. Considering covariance matrix and sum of individual countries: {region_search}.')

            country_list = region_search.split('-')
            ds_region = ds.sel({'country':country_list})

            for v in ["posterior", "prior"]:
                ds_region[v] = ds_region[f"country_flux_total_{v}"].sum(dim="country")

            ds_region['sigma_prior'] = np.sqrt(((ds.country_flux_total_prior
                                                 - ds.percentile_country_flux_total_prior.isel(percentile=min_percentile_index)
                                                 )** 2
                                                ).sum(dim="country"))
            
            if 'covariance_country_flux_total_posterior' in ds.variables:
                ds_region['sigma_posterior'] = np.sqrt(ds_region['covariance_country_flux_total_posterior'].sum(dim='country').sum(dim='country'))
                
            else:
                logger.warning(f'Covariance matrix is not available for {m}. A posteriori uncertainty of {country} emissions will not be plotted.')
                ds_region['sigma_posterior'] = np.nan * ds_region['posterior']
                
            for v in ["posterior", "prior"]:
                ds_region[f"{v}_lower"] = ds_region[v] - ds_region[f"sigma_{v}"]
                ds_region[f"{v}_upper"] = ds_region[v] + ds_region[f"sigma_{v}"]

        elif country_search in available_countries:
            ds_region = ds.sel({"country": country_search})

            for v in ["posterior", "prior"]:
                ds_region[v] = ds_region[f"country_flux_total_{v}"]

                ds_region[f"{v}_lower"] = ds_region[f"percentile_country_flux_total_{v}"
                                                    ].isel(percentile=min_percentile_index)
                ds_region[f"{v}_upper"] = ds_region[f"percentile_country_flux_total_{v}"
                                                    ].isel(percentile=max_percentile_index)

        else:
            raise ValueError(f'{country_search} ({country}) is not available for {m}')

        for v in ["posterior", "prior"]:
            ds_region[f"{v}_lower"] = ds_region[f"{v}_lower"].clip(min=0)

        ds_output[m] = ds_region[
            ["posterior","posterior_lower","posterior_upper",
             "prior","prior_lower","prior_upper",
            ]
        ]
    return ds_output


def extract_region_inventory_flux(
    data_dir: str,
    country: str,
    specie: str,
    unit: str,
    s_data: dict[str, dict],
    inventory_year: int | str | None = None,
) -> xr.Dataset:
    """
    Extracts inventory flux values for regions that exists,
    or calculates total inventory flux values for aggregated regions.

    Args:
        data_dir: directory which contains the data (should have inside a directory named 'inventory').
        specie: Gas species, e.g. 'ch4'.
        unit: unit in which the inventory should be converted.
        s_data: Dictionary of species with information for plotting (read from json file).
        inventory_year: year of inventory to get.

    Returns:
        dataset with country selected

    """

    # Find filename
    if inventory_year is not None:
        filepath = os.path.join(
            data_dir,
            "inventory",
            f"UNFCCC_inventory_{specie}_{inventory_year}.nc",
        )
    else:
        filelist = sorted(
            glob.glob(os.path.join(data_dir, "inventory", f"UNFCCC_inventory_{specie}_*.nc"))
        )
        if filelist:
            filepath = filelist[-1]
            inventory_year = int(filepath.split("_")[-1].split(".")[0])
        else:
            filepath = os.path.join(
                data_dir,
                "inventory",
                f'UNFCCC_inventory_{s_data[specie]["model_species"]["intem"]}.nc',
            )
            inventory_year = None
    inv_ds = xr.open_dataset(filepath)['inventory'] 

    gwp = 1
    target_unit = unit
    if "CO2-eq" in target_unit:
        gwp = s_data[specie]["gwp"]
        target_unit = unit.replace("CO2-eq","")
        logger.info(f'Converting to mass of CO2-eq using GWP = {gwp}.')
    scaling_factor = get_units_conversion_factor(inv_ds.units.replace('/yr',' yr-1'), target_unit, s_data[specie]["molar_mass"])

    inv_ds = inv_ds * scaling_factor * gwp
    inv_ds.attrs['units'] = unit
    inv_ds.attrs['year'] = inventory_year

    if country in inv_ds['country']:
        return inv_ds.sel(country=country)

    region_search = config.regions_dict[country]
    country_list = region_search.split('-')
    logger.info(f'No inventory data available for {country}. Considering sum of individual countries: {region_search}')

    country_list_update = [country if country in inv_ds['country'] 
                            else dict(map(reversed, config.countrycodes_dict.items()))[country] # type: ignore
                            for country in country_list]
    return inv_ds.sel(country=country_list_update).sum(dim='country', keep_attrs=True)