import logging
import os
import numpy as np
import xarray as xr
from pathlib import Path
from fluxie.operators.convert import get_units_conversion_factor,convert_units_co2eq
from fluxie.operators.flux_align_dataset import align_lat_lon

logger = logging.getLogger(__name__)


def open_and_align_sector_dataset(
    sector_prop_path: str, ds_ref: xr.Dataset
) -> xr.Dataset:
    """
    Open dataset containing prior sector fluxes and align it temporaly and spatially on reference dataset.

    Args:
        sector_prop_path: path to dataset containing prior sector fluxes
        ds_ref: dataset used as reference to align along

    Returns:
        ds_sectors: dataset with prior sector fluxes align on reference dataset
    """

    with xr.open_dataset(sector_prop_path) as f:
        # timely align sector dataset on main dataset

        ds_sectors = f.sel(time=ds_ref["time"].values, method="ffill")
        ds_sectors["time"] = ds_ref["time"]

        # spatially align sector dataset on main dataset
        if "lat" in ds_sectors.coords:
            ds_sectors = ds_sectors.rename({"lat": "latitude", "lon": "longitude"})
            _, ds_sectors = align_lat_lon([ds_ref, ds_sectors], "latitude")
            _, ds_sectors = align_lat_lon([ds_ref, ds_sectors], "longitude")

    return ds_sectors


def create_cell_area(
    ds_ref: xr.Dataset, cell_area_test_file: bool, sector_file: str
) -> xr.DataArray:
    """
    Open dataset containing prior sector fluxes and align it temporaly and spatially on reference dataset.

    Args:
        ds_ref: dataset for which we want to calculate the area of the grid cells
        cell_area_test_file: Only used in tests. If True, extracts cell_area from a smaller test file
            with restricted lat/lons.
        sector_file: path to sector dataset, used to define domain if domain not in ds_ref attributes

    Returns:
        cell_area: dataarray containing cell size
    """
    parent_dir = Path(__file__).parent.parent.parent
    if cell_area_test_file:
        configs_dir = parent_dir / "data" / "tests" / "configs"
    else:
        configs_dir = parent_dir / "configs"
    if "domain" in ds_ref.attrs:
        domain = ds_ref.attrs["domain"]
    else:
        domain = sector_file.split("_")[0]
        logger.warning(
            f"No domain info in dataset attributes, so reading domain from sector_filename: {domain}"
        )
    with xr.open_dataset(os.path.join(configs_dir, f"{domain}_cell_area.nc")) as f:
        cell_area = f.cell_area

    _, cell_area = align_lat_lon([ds_ref, cell_area], "latitude")
    _, cell_area = align_lat_lon([ds_ref, cell_area], "longitude")

    return cell_area


def scale_by_sector_proportions(
    data_dir: str,
    ds_all: dict[xr.Dataset],
    species: str,
    country_flux_units_print: str,
    regions=None,
    config_data=dict[str, dict],
    sector_file: str = "EUROPE_EDGAR",
    create_sectors: list[bool] | bool = True,
    create_region_sector_totals: list[bool] | bool = True,
    sectors: list[str] | None = None,
    cell_area_test_file: bool = False,
) -> xr.Dataset:
    """
    Produces sector level fluxes by scaling prior and posterior fluxes by the
    proportional contribution of each sector to the total flux in each grid cell
    (at transport model resolution). Fluxes are then summed over region/country areas
    to produce region/country sector totals.

    Args:
        data_dir (str):
            Path to top data directory.
        ds (xarray dataset):
            xarray dataset with model data.
        species (str):
            Gas species, e.g. 'ch4'.
        sectors (list of str):
            Emissions sectors to include, options for 'agriculture', 'waste',
            'energy' and 'industry'.
        country_flux_units_print (str):
            Units for country flux, e.g. 'Gg yr-1'
        config_data (dict):
            config_data (dict of dict):
            Dictionary with settings read from json file.
            Use json filenames as keys.
        sector_file (str):
            Start of sector file name, e.g. 'EUROPE_EDGAR'
        create_sectors (bool):
            If True, calculates sector spatial fluxes. Can be set as a 
        create_region_sector_totals (bool):
            If True, sums spatial fluxes over country_fraction masks
            to create country/region totals.
        cell_area_test_file (bool):
            Only used in tests. If True, extracts cell_area from a smaller test file
            with restricted lat/lons.
    Returns:
        ds (xarray dataset):
            Input ds, with added flux_sector_prior and flux_sector_posterior variables.
            If regions is not None, also contains region/country sector total variables.
    """

    logger.warning(
        f"Scaling fluxes to create sector flux totals, this can be slow (>1 minute to run) if using lots of models and regions."
    )
    
    if type(create_sectors) == bool:
        create_sectors = [create_sectors] * len(list(ds_all.keys()))
        logger.warning('Creating sector-level emissions for all models.')
    if type(create_region_sector_totals) == bool:
        create_region_sector_totals = [create_region_sector_totals] * len(list(ds_all.keys()))
        logger.warning('Creating sector-level region emissions for all models.')
        

    r_data = config_data.get("regions_info", {})["country_codes"]

    # list of both grouped country code (e.g. BEL-LUX-NLD) and separated country codes
    # e.g BEL, LUX, NLD to create sector fluxes for, as different models use different methods
    # for summing emissions over grouped regions
    if regions:
        region_codes = []
        for r in regions:
            if r in r_data.keys():
                region_codes.append(r_data[r])
                if "-" in r_data[r]:
                    region_codes += r_data[r].split("-")

    sector_prop_path = sector_prop_path = os.path.join(
        data_dir, "PRIOR",species, f"PRIOR_{sector_file}.nc"
    )

    logger.info(f"Using {sector_prop_path} to scale total fluxes into sector fluxes.")

    ds_all_out = {}
    scaling_factor_all = {}

    for m, (model, ds) in enumerate(ds_all.items()):
        
        if create_sectors[m]:

            ds_sectors = open_and_align_sector_dataset(sector_prop_path, ds)
            if "flux_total_posterior" in list(ds_sectors.keys()) or "flux_energy_posterior" in list(ds_sectors.keys()):
                var_search = "posterior"
            elif "flux_total_prior" in list(ds_sectors.keys()) or "flux_energy_prior" in list(ds_sectors.keys()): 
                var_search = "prior"
                                
            if f"flux_total_{var_search}" not in list(ds_sectors.keys()):
                flux_vars = [v for v in list(ds_sectors.keys()) if v.startswith('flux') and v.endswith(var_search)]
                logger.warning(f"No '_total_' variable available in sector scaling dataset, so creating this by summing sectors: {flux_vars}")
                
                for v,var in enumerate(flux_vars):
                    if v == 0:
                        ds_sectors[f"flux_total_{var_search}"] = ds_sectors[var]
                    else:
                        ds_sectors[f"flux_total_{var_search}"] += ds_sectors[var]                
            
            ###TEMPORARY FIX FOR RHIME UN-ROUNDED LATITUDES AND LONGTIUDES
            for c in ['latitude','longitude']:
                ds = ds.assign_coords({c:np.round(ds[c],3)})
                ds_sectors = ds_sectors.assign_coords({c:np.round(ds[c],3)})
            ###
                
            if m == 0 and sectors == None:
                sectors = [v.split("_")[-1] for v in ds_sectors if "total" not in v]
                logger.warning(
                    "No sectors specified, so reading sector list from sector_flux file."
                    + f" Used sectors: {sectors}"
                )

            # Convert prior and posterior flux for each sector
            for s in sectors:
                print(f'Working on {s}...')
                scaling_factor_all[s] = ds_sectors[f"flux_{s}_{var_search}"] / ds_sectors[f"flux_total_{var_search}"]
                scaling_factor_all[s] = scaling_factor_all[s].where(
                    ds_sectors[f"flux_total_{var_search}"] != 0, 0
                )

                for suff in ["posterior", "prior"]:
                    ds[f"flux_{s}_{suff}"] = (
                        ds[f"flux_total_{suff}"] * scaling_factor_all[s]
                    )
                    ds[f"flux_{s}_{suff}"].attrs = {
                        "units": ds[f"flux_total_{suff}"].attrs["units"],
                        "_FillValue": np.nan,
                        "long_name": f"{suff} {s} {species} flux, created by scaling total flux by sector proportions",
                    }

            # Convert prior and posterior country flux for each sector if needed
            if create_region_sector_totals[m]:
                # derive factor for unit conversion
                units_factor = convert_units_co2eq(
                    ds[f"flux_{s}_prior"].attrs["units"].replace(" m-2 ", " "),
                    country_flux_units_print,
                    config_data["species_info"][species],
                )

                # check for country_fraction
                if "country_fraction" not in ds:
                    raise ValueError(
                        f"country_fraction variable not present in dataset for {model} "
                        "Cannot add sector country totals."
                    )

                # check for cell_area and create if needed
                if "cell_area" not in ds:
                    ds["cell_area"] = create_cell_area(ds, cell_area_test_file, sector_file)
                logger.warning(
                    "cell_area should be in meter square and flux in something per m-2. No check is made on this for the moment."
                )

                # calculate sector country flux (prior/posterior) from sector flux (prior/posterior), cell_area, country_fraction and unit_factor
                for i, s in enumerate(sectors):
                    if regions:
                        ds_flux_country_list = list()

                        #for country in np.unique(region_codes):
                        for country in regions:    
                            if country not in ds.country:
                                continue
                            tmp_list = []
                            for suff in ["posterior", "prior"]:
                                tmp_list.append(
                                    (
                                        ds[f"flux_{s}_{suff}"]
                                        * ds["country_fraction"].sel(country=country)
                                        * ds["cell_area"]
                                    ).sum(dim=["latitude", "longitude"])
                                    * units_factor
                                )
                                tmp_list[-1].name = f"flux_{s}_{suff}_country"
                                #print(tmp_list)
                            ds_flux_country_list.append(
                                xr.merge(tmp_list, compat="no_conflicts")
                            )

                        if len(regions) > 1:
                            ds_flux_country = xr.concat(ds_flux_country_list, dim="country")
                        else:
                            ds_flux_country = ds_flux_country_list[0]

                        ds = xr.merge(
                            [ds, ds_flux_country], compat="no_conflicts", join="outer"
                        )

                    else:
                        if i == 0 and m == 0:
                            logger.warning(
                                "Calculating sector fluxes for all regions. This can be very slow."
                            )
                        for suff in ["posterior", "prior"]:
                            ds[f"flux_{s}_{suff}_country"] = (
                                ds[f"flux_{s}_{suff}"]
                                * ds["country_fraction"]
                                * ds["cell_area"]
                            ).sum(dim=["latitude", "longitude"]) * units_factor

                    # add atributes to created variables
                    for suff in ["posterior", "prior"]:
                        ds[f"flux_{s}_{suff}_country"].attrs = {
                            "units": country_flux_units_print,
                            "_FillValue": np.nan,
                            "long_name": f"country {s} {species} {suff} flux, created by scaling total flux by sector proportions",
                        }

        ds_all_out[model] = ds

    return ds_all_out
