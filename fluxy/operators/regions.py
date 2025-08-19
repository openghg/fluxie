import glob
import logging
import numpy as np
import xarray as xr

from pathlib import Path

from fluxy.operators.convert import get_units_conversion_factor

logger = logging.getLogger(__name__)


def extract_region_flux(
    ds_all: dict[str, xr.Dataset],
    country: str,
    regions_info: dict[str, str],
    keep_country_dim: bool = False,
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
        regions_info: dictionary with country and region names (read from json file).
        keep_country_dim: if True, re-put country dimension on the output datasets.

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
    country_codes = regions_info.get("country_codes", {})
    country_search = country_codes.get(country, country)
    min_percentile_index = 0
    max_percentile_index = 1

    if all([("all" in ds.attrs.get("species", "")) for ds in ds_all.values()]):
        ds_output = {m: ds.sel({"country": country_search}) for m, ds in ds_all.items()}
        return ds_output

    target_vars = [
        "posterior",
        "posterior_lower",
        "posterior_upper",
        "prior",
        "prior_lower",
        "prior_upper",
    ]

    dict_regions: dict[str, str] = regions_info.get("regions", {})

    flag_percentile = False
    flag_stdev = False

    for m, ds in ds_all.items():
        # search for existing region names
        available_countries = ds["country"].values.astype(str)

        if country_search not in available_countries and country in dict_regions.keys():
            region_search = dict_regions[country]

            logger.info(
                f"{country} emissions are not present in {m}. Considering covariance matrix and sum of individual countries: {region_search}."
            )

            country_list = region_search.split("-")
            ds_region = ds.sel({"country": country_list})
            if "country_2" in ds_region.dims:
                ds_region = ds_region.sel({"country_2": country_list})

            for v in ["posterior", "prior"]:
                ds_region[v] = ds_region[f"flux_total_{v}_country"].sum(
                    dim="country", keep_attrs=True
                )

            if "percentile_flux_total_prior_country" in ds_region.variables:
                ds_region["sigma_prior"] = np.sqrt(
                    (
                        (
                            ds_region["flux_total_prior_country"]
                            - ds_region["percentile_flux_total_prior_country"].isel(
                                percentile=min_percentile_index
                            )
                        )
                        ** 2
                    ).sum(dim="country")
                )
            elif "stdev_flux_total_prior_country" in ds_region.variables:
                ds_region["sigma_prior"] = np.sqrt(
                    ((ds_region["stdev_flux_total_prior_country"]) ** 2).sum(dim="country")
                )
            else:
                ds_region["sigma_prior"] = xr.zeros_like(
                    ds_region["flux_total_prior_country"]
                ).sum(dim="country")

            if "covariance_flux_total_posterior_country" in ds_region.variables:
                ds_region["sigma_posterior"] = np.sqrt(
                    ds_region["covariance_flux_total_posterior_country"]
                    .sum(dim="country")
                    .sum(dim="country_2")
                )

            else:
                logger.warning(
                    f"Covariance matrix is not available for {m}. A posteriori uncertainty of {country} emissions will not be plotted."
                )
                ds_region["sigma_posterior"] = np.nan * ds_region["posterior"]

            for v in ["posterior", "prior"]:
                ds_region[f"{v}_lower"] = ds_region[v] - ds_region[f"sigma_{v}"]
                ds_region[f"{v}_upper"] = ds_region[v] + ds_region[f"sigma_{v}"]

        elif country_search in available_countries:
            ds_region = ds.sel({"country": country_search})

            for v in ["posterior", "prior"]:
                ds_region[v] = ds_region[f"flux_total_{v}_country"]
                var_percentile = f"percentile_flux_total_{v}_country"
                var_stdev = f"stdev_flux_total_{v}_country"

                if var_percentile in ds_region.variables:
                    da = ds_region[var_percentile]
                    ds_region[f"{v}_lower"] = da.isel(percentile=min_percentile_index)
                    ds_region[f"{v}_upper"] = da.isel(percentile=max_percentile_index)

                    # Print info
                    flag_percentile = True
                    confidence_interval = (
                        ds_region["percentile"][max_percentile_index].values
                        - ds_region["percentile"][min_percentile_index].values
                    ) * 100
                    logger.info(
                        f"Using {var_percentile} to plot {m} {v} country flux {confidence_interval:.1f}% confidence interval."
                    )
                elif var_stdev in ds_region.variables:
                    ds_region[f"{v}_lower"] = ds_region[v] - ds_region[var_stdev]
                    ds_region[f"{v}_upper"] = ds_region[v] + ds_region[var_stdev]

                    # Print info
                    flag_stdev = True
                    logger.info(
                        f"Using {var_stdev} to plot {m} {v} country flux 68.2% confidence interval."
                    )
                else:
                    da = ds_region[f"flux_total_{v}_country"]
                    ds_region[f"{v}_lower"] = da
                    ds_region[f"{v}_upper"] = da

        else:
            raise ValueError(f"{country_search} ({country}) is not available for {m}")

        for v in ["posterior", "prior"]:
            ds_region[f"{v}_lower"] = ds_region[f"{v}_lower"].clip(min=0)

        ds_region = ds_region[target_vars]

        if keep_country_dim and "country" not in ds_region.dims:
            ds_region = ds_region.expand_dims(
                dim={
                    "country": [
                        country_search,
                    ]
                }
            )

        ds_output[m] = ds_region

    if flag_percentile and flag_stdev:
        logger.warning(
            f"Confidence intervals in {country} are being computed from percentile or stdev depending on the dataset. Set logging level to INFO to check for consistency."
        )

    return ds_output


def extract_region_inventory_flux(
    data_dir: str,
    country: str,
    species: str,
    unit: str,
    s_data: dict[str, dict],
    r_data: dict[str, str],
    inventory_year: int | str | None,
    inventory_filename: str,
) -> xr.Dataset:
    """
    Extracts inventory flux values for regions that exists,
    or calculates total inventory flux values for aggregated regions.

    Args:
        data_dir: directory which contains the data (should have inside a directory named 'inventory').
        specie: Gas species, e.g. 'ch4'.
        unit: unit in which the inventory should be converted.
        s_data: Dictionary of species with information for plotting (read from json file).
        r_data: Dictionary with country and region names (read from json file).
        inventory_year: year of inventory to get.
        inventory_filename: Name of inventory file: {inventory_filename}_{species}_{inventory_year}
    Returns:
        dataset with country selected

    """
        
    # Find filename
    if inventory_year is not None:
        filepath = (
            Path(data_dir) / "inventory" /
            f"{inventory_filename}_{species}_{inventory_year}.nc"
        )
    else:
        filelist = sorted(
            (Path(data_dir) / "inventory").glob(f"{inventory_filename}_{species}_*.nc")
        )
        if filelist:
            filepath = filelist[-1]
            inventory_year = int(str(filepath).split("_")[-1].split(".")[0])
        else:
            filepath = (
                Path(data_dir)
                / "inventory"
                / f'{inventory_filename}_{s_data[species]["model_species"]["intem"]}.nc'
            )
            inventory_year = None

    inv_ds_all = xr.open_dataset(filepath)
        
    if 'missing_data' in inv_ds_all.attrs:
        if inv_ds_all.attrs['missing_data'] != '[]':
            logger.warning(f"Inventory is missing data: {inv_ds_all.attrs['missing_data']}")
    else:
        logger.warning(f'No missing_data variable available in inventory files, assuming all data present.')
    
    #first option left for compatability with older inventory netcdfs, can be removed later
    inv_ds = inv_ds_all['inventory'] if 'inventory' in inv_ds_all.keys() else inv_ds_all["flux_total_inventory_country"]

    gwp = 1
    target_unit = unit
    origin_unit = inv_ds.units.replace("/yr", " yr-1").replace("/y", " yr-1")
    molar_mass = s_data[species]["molar_mass"] if "all" not in species else None
    if "CO2-eq" in target_unit:
        if "CO2eq" not in origin_unit.replace("-", ""):
            gwp = s_data[species]["gwp"]
        else:
            origin_unit = origin_unit.replace("CO2-eq", "").replace("CO2eq", "")
        target_unit = unit.replace("CO2-eq", "")
        logger.info(f"Converting to mass of CO2-eq using GWP = {gwp}.")
        
    scaling_factor = get_units_conversion_factor(origin_unit, target_unit, molar_mass)

    inv_ds = inv_ds * scaling_factor * gwp
    inv_ds.attrs["units"] = unit
    inv_ds.attrs["year"] = inventory_year

    # Get country_codes only if regions_info exists    
    country_codes = r_data.get("country_codes", {})
    # Look for the code if country_codes is defined, otherwise assume the code was given as input
    country_search = country_codes.get(country, country)
    
    if country_search in inv_ds["country"]:
        return inv_ds.sel(country=country_search) # new format
    elif country in inv_ds["country"]:
        return inv_ds.sel(country=country) # old format (would only work if the user specifies the country name)

    # if grouped countries:
    available_countries = inv_ds["country"].values.astype(str)
    dict_regions: dict[str, str] = r_data.get("regions", {})
    
    if (country_search not in available_countries and country in dict_regions.keys()):
        region_search = dict_regions[country]
        country_list = region_search.split("-")
        inv_ds = inv_ds.sel({"country": country_list})
        
        logger.info(
        f"No inventory data available for {country}. Considering sum of individual countries: {region_search}"
        )
    elif country_search in available_countries:
        inv_ds = inv_ds.sel({"country": country_search})

    return inv_ds.sum(dim="country", keep_attrs=True)

