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
    country_search = regions_info["country_codes"][country]
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

    for m, ds in ds_all.items():
        # search for existing region names
        available_countries = ds["country"].values.astype(str)

        if (
            country_search not in available_countries
            and country in regions_info["regions"].keys()
        ):
            region_search = regions_info["regions"][country]

            logger.info(
                f"{country} emissions are not present in {m}. Considering covariance matrix and sum of individual countries: {region_search}."
            )

            country_list = region_search.split("-")
            ds_region = ds.sel({"country": country_list})
            if "country_2" in ds_region.dims:
                ds_region = ds_region.sel({"country_2": country_list})

            for v in ["posterior", "prior"]:
                ds_region[v] = ds_region[f"country_flux_total_{v}"].sum(dim="country")

            ds_region["sigma_prior"] = np.sqrt(
                (
                    (
                        ds.country_flux_total_prior
                        - ds.percentile_country_flux_total_prior.isel(
                            percentile=min_percentile_index
                        )
                    )
                    ** 2
                ).sum(dim="country")
            )

            if "covariance_country_flux_total_posterior" in ds.variables:
                ds_region["sigma_posterior"] = np.sqrt(
                    ds_region["covariance_country_flux_total_posterior"]
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
                ds_region[v] = ds_region[f"country_flux_total_{v}"]

                ds_region[f"{v}_lower"] = ds_region[
                    f"percentile_country_flux_total_{v}"
                ].isel(percentile=min_percentile_index)
                ds_region[f"{v}_upper"] = ds_region[
                    f"percentile_country_flux_total_{v}"
                ].isel(percentile=max_percentile_index)

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

    return ds_output


def extract_region_inventory_flux(
    data_dir: str,
    country: str,
    species: str,
    unit: str,
    s_data: dict[str, dict],
    r_data: dict[str, str],
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
        r_data: Dictionary with country and region names (read from json file).
        inventory_year: year of inventory to get.

    Returns:
        dataset with country selected

    """

    # Find filename
    if inventory_year is not None:
        filepath = (
            Path(data_dir) / "inventory"
            f"UNFCCC_inventory_{species}_{inventory_year}.nc"
        )
    else:
        filelist = sorted(
            (Path(data_dir) / "inventory").glob(f"UNFCCC_inventory_{species}_*.nc")
        )
        if filelist:
            filepath = filelist[-1]
            inventory_year = int(str(filepath).split("_")[-1].split(".")[0])
        else:
            filepath = (
                Path(data_dir)
                / "inventory"
                / f'UNFCCC_inventory_{s_data[species]["model_species"]["intem"]}.nc'
            )
            inventory_year = None
    inv_ds = xr.open_dataset(filepath)["inventory"]

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

    if country in inv_ds["country"]:
        return inv_ds.sel(country=country)

    region_search = r_data["regions"][country]
    country_list = region_search.split("-")
    logger.info(
        f"No inventory data available for {country}. Considering sum of individual countries: {region_search}"
    )

    country_list_update = [
        (
            country
            if country in inv_ds["country"]
            else dict(map(reversed, r_data["country_codes"].items()))[country]
        )  # type: ignore
        for country in country_list
    ]
    return inv_ds.sel(country=country_list_update).sum(dim="country", keep_attrs=True)
