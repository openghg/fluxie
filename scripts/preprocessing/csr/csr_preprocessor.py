"""
Script to convert the CSR prior and posterior flux results into fluxy format. 
It can be run by the csr_preprocess.ipynb notebook.
"""

import xarray as xr
import os
import re
import calendar
import numpy as np
import pandas as pd
import logging
from pathlib import Path

rename_candidates = {
    "longitude": ["lon"],
    "latitude": ["lat"],
    "time": ["mtime"],
    "country": ["regname"],
    "area": ["cell_area"],
}

combine_candidates = {
    "flux_combined": ["flux_total_prior", "flux_total_posterior"],
    "flux": ["flux_total_prior_country", "flux_total_posterior_country"],
    "flux_unc": [
        "stdev_flux_total_prior_country",
        "stdev_flux_total_posterior_country",
    ],
}

unit_conversions = {"PgC/yr": 1, "TgC/yr": 1 / 1000, "Tmol/yr": 12.01 / 1000}

flux_unit = "mol m-2 s-1"

country_code_EUROPE30f = [
    "none",
    "AUT",
    "BEL",
    "BGR",
    "CHE",
    "CYP",
    "CZE",
    "DEU",
    "DNK",
    "EST",
    "GRC",
    "ESP",
    "FIN",
    "FRA",
    "HRV",
    "HUN",
    "IRL",
    "ITA",
    "LTU",
    "LUX",
    "LVA",
    "MLT",
    "NLD",
    "POL",
    "PRT",
    "ROU",
    "SWE",
    "SVN",
    "SVK",
    "GBR",
    "LAND",
    "OCEAN",
]

country_code_EUROCOM21f = [
    "DEU",
    "FIN",
    "FRA",
    "ITA",
    "NOR",
    "POL",
    "SWE",
    "TUR",
    "WEE",
    "CEE",
    "NOE",
    "SOE",
    "SEE",
    "EAE",
    "BNL",
    "UKI",
    "IBE",
    "E28",
    "E27",
    "E15",
    "EUR",
    "LAND",
    "OCEAN",
]

country_code_FLUXYf = [
    "CHE",
    "SWE",
    "ESP",
    "SVK",
    "SVN",
    "PRT",
    "POL",
    "NOR",
    "LUX",
    "ITA",
    "IRL",
    "HUN",
    "DEU",
    "CZE",
    "HRV",
    "BEL",
    "AUT",
    "OCEAN",
]

country_code_FLUXYALLf = [
    "CHE",
    "SWE",
    "ESP",
    "SVK",
    "SVN",
    "ROU",
    "PRT",
    "POL",
    "NOR",
    "MLT",
    "LUX",
    "LTU",
    "LVA",
    "ITA",
    "IRL",
    "HUN",
    "GRC",
    "DEU",
    "EST",
    "CZE",
    "CYP",
    "HRV",
    "BGR",
    "BEL",
    "AUT",
    "TUR",
    "WEE",
    "CEE",
    "NOE",
    "SOE",
    "SEE",
    "EAE",
    "BNL",
    "UKI",
    "IBE",
    "E28",
    "E27",
    "E15",
    "EUR",
    "LAND",
    "OCEAN",
]

header_names_rhs = [
    "time",
    "row",
    "col",
    "covar_pri",
    "covar_post",
    "corr_pri",
    "corr_post",
    "deriv_mu",
]


def preprocess(
    path_to_prior: str,
    path_to_posterior: str,
    path_to_prior_country: str,
    path_to_posterior_country: str,
    path_to_uncertainty_country: str,
    path_to_output: str,
    species: str,
):
    """
    Main function, which converts the CSR flux results into the fluxy format.

    Args:
        path_to_prior (str):
            Full path to gridded CSR prior emission file
        path_to_posterior (str):
            Full path to gridded CSR posterior emission file
        path_to_prior_country (str):
            Full path to country-aggregated CSR prior emission file
        path_to_posterior_country (str):
            Full path to country-aggregated CSR posterior emission file
        path_to_uncertainty_country (str):
            Full path to the CSR "right-hand-side" results containing the flux uncertainties
        path_to_output (str):
            Full path to the directory where the results in fluxy format are to be stored
        species (str):
            Species (e.g. "ch4", "co2")

    Note:
        So far EUROPE30f, EUROCOM21f, FLUXYf, FLUXYALLf country masks are implemented.
    """

    # --- combine gridded and country-aggregated prior and posterior fluxes ---

    ds_prior = xr.open_dataset(path_to_prior, decode_timedelta=True)
    ds_posterior = xr.open_dataset(path_to_posterior, decode_timedelta=True)
    ds_prior_country = xr.open_dataset(path_to_prior_country, decode_timedelta=True)
    ds_posterior_country = xr.open_dataset(
        path_to_posterior_country, decode_timedelta=True
    )

    ds_prior = _rename(ds_prior)
    ds_prior = _convert_time(ds_prior)
    ds_prior = _combine_fluxes(ds_prior, species)
    ds_posterior = _rename(ds_posterior)
    ds_posterior = _convert_time(ds_posterior)
    ds_posterior = _combine_fluxes(ds_posterior, species)
    ds_prior_country = _rename(ds_prior_country)
    ds_prior_country = _rename_country_id(ds_prior_country)
    ds_prior_country = _convert_time(ds_prior_country)
    ds_prior_country = _combine_fluxes_country(ds_prior_country, species)
    ds_posterior_country = _rename(ds_posterior_country)
    ds_posterior_country = _rename_country_id(ds_posterior_country)
    ds_posterior_country = _convert_time(ds_posterior_country)
    ds_posterior_country = _combine_fluxes_country(ds_posterior_country, species)

    ds = _combine_variable(
        ds_prior, ds_posterior, ds_prior_country, ds_posterior_country, species=species
    )

    for var in ds.data_vars:
        if "rt" in ds[var].dims:
            ds[var] = ds[var].isel(rt=0)

    if "rt" in ds.coords:
        ds = ds.drop_vars("rt")

    drop = [
        "flux_land",
        "flux_ocean",
        "flux_subt",
        "flux_excl",
        "lproc",
        "lrt",
        "lspec",
        "dt",  # remove to avoid encoding issues (anyway not needed)
    ]

    to_drop = [v for v in ds.variables if any(sub in v for sub in drop)]

    ds = ds.drop_vars(to_drop)

    prior = ds["flux_total_prior"].values
    posterior = ds["flux_total_posterior"].values

    # --- add uncertainties for country-aggregated fluxes from RHS-runs ---

    if isinstance(path_to_uncertainty_country, list):
        # check if RHS results are available for each flux time-step (e.g. each year or month)
        if len(ds["time"]) != len(path_to_uncertainty_country):
            print(
                "Number of RHS results and number of time steps do not match! Stop RHS processing."
            )
        else:
            data_prior = np.zeros((len(ds["country"]), len(ds["time"])))
            data_posterior = np.zeros((len(ds["country"]), len(ds["time"])))
            for i in range(0, len(path_to_uncertainty_country)):
                if os.path.exists(path_to_uncertainty_country[i]):
                    # read RHS results for each flux time-step
                    df_cross = pd.read_csv(
                        path_to_uncertainty_country[i],
                        comment="#",
                        sep=" ",
                        names=header_names_rhs,
                        skipinitialspace=True,
                    )
                    unc_prior_country_file = np.sqrt(
                        df_cross.loc[df_cross["row"] == df_cross["col"], "covar_pri"]
                    )
                    unc_posterior_country_file = np.sqrt(
                        df_cross.loc[df_cross["row"] == df_cross["col"], "covar_post"]
                    )
                    unc_prior_country_file = _tmolyr_to_kg_s_unc_rhs(
                        unc_prior_country_file, ds["time"][i], species
                    )
                    unc_posterior_country_file = _tmolyr_to_kg_s_unc_rhs(
                        unc_posterior_country_file, ds["time"][i], species
                    )
                else:
                    unc_prior_country_file = np.nan
                    unc_posterior_country_file = np.nan
                data_prior[:, i] = unc_prior_country_file
                data_posterior[:, i] = unc_posterior_country_file

            # write to xarray:
            unc_prior = xr.DataArray(
                data=data_prior,
                coords={"country": ds["country"].values, "time": ds["time"].values},
                dims=["country", "time"],
                name="stdev_flux_total_prior_country",
            )
            unc_prior.attrs["units"] = "kg s-1"

            unc_posterior = xr.DataArray(
                data=data_posterior,
                coords={"country": ds["country"].values, "time": ds["time"].values},
                dims=["country", "time"],
                name="stdev_flux_total_posterior_country",
            )
            unc_posterior.attrs["units"] = "kg s-1"

            ds["stdev_flux_total_prior_country"] = unc_prior
            ds["stdev_flux_total_posterior_country"] = unc_posterior

    else:
        print("Uncertainties for country-aggregated fluxes are not available.")

    print("_save_dataset")

    _save_dataset(ds, path_to_output)


def _rename(ds):
    """
    Renames variables of the CSR output files.

    Args:
        ds (xarray.Dataset):
            CSR output with original variable names
    Returns:
        ds (xarray.Dataset):
            CSR output with renamed variables
    """

    rename_dict = {}
    for target_name, possible_names in rename_candidates.items():
        for name in possible_names:
            if name in ds.dims or name in ds.coords or name in ds.data_vars:
                rename_dict[name] = target_name
                break

    return ds.rename(rename_dict) if rename_dict else ds


def _rename_country_id(ds_in):
    """
    Renames country codes of the country-aggregated CSR output files.

    Args:
        ds_in (xarray.Dataset):
            Country-aggregated CSR output with original contry codes
    Returns:
        ds (xarray.Dataset):
            Country-aggregated CSR output with renamed country codes
    """

    ds = ds_in.copy()
    if "EUROPE30f" in ds.attrs["filename"]:
        ds["country"] = (("reg"), country_code_EUROPE30f)
    if "EUROCOM21f" in ds.attrs["filename"]:
        ds["country"] = (("reg"), country_code_EUROCOM21f)
    if "FLUXYf" in ds.attrs["filename"]:
        ds["country"] = (("reg"), country_code_FLUXYf)
    if "FLUXYALLf" in ds.attrs["filename"]:
        ds["country"] = (("reg"), country_code_FLUXYALLf)

    return ds


def _combine_fluxes(ds_in, species: str):
    """
    Combines gridded land and ocean fluxes (if available) and converts units.

    Args:
        ds_in (xarray.Dataset):
            Gridded CSR output with land and ocean fluxes
        species (str):
            Species (e.g. "ch4", "co2")

    Returns:
        ds (xarray.Dataset):
            Gridded CSR output with combined land and ocean fluxes
    """

    ds = ds_in.copy()
    land = _check_for_units(f"{species}flux_land", ds_in)
    ocean = _check_for_units(f"{species}flux_ocean", ds_in)

    if land is None and ocean is None:
        print("DEBUG: Only combined flux available!")
        # Fallback to combined flux if neither land nor ocean is available
        combined = _check_for_units(f"{species}flux", ds_in)
        if combined is None:
            raise ValueError(
                f"No {species}flux_land, {species}flux_ocean, or {species}flux in dataset"
            )
        flux_combined = combined
        attrs = _copy_attrs(combined)
    elif land is None:
        flux_combined = ocean
        attrs = _copy_attrs(ocean)
    elif ocean is None:
        flux_combined = land
        attrs = _copy_attrs(land)
    else:
        print("DEBUG: Land & ocean flux separately available.")
        flux_combined = land + ocean
        attrs = _copy_attrs(land)
        attrs.update(_copy_attrs(ocean))

    flux_combined.attrs = attrs

    ds[f"{species}{list(combine_candidates.keys())[0]}"] = flux_combined

    return ds


def _combine_fluxes_country(ds_in, species: str):
    """
    Converts units of country-aggregated fluxes.

    Args:
        ds_in (xarray.Dataset):
            Country-aggregated CSR output
        species (str):
            Species (e.g. "ch4", "co2")

    Returns:
        ds (xarray.Dataset):
            Country-aggregated CSR output with updated units
    """

    ds = ds_in.copy()
    flux = _check_for_units_country(f"{species}flux", ds_in, species)

    ds[f"{species}{list(combine_candidates.keys())[1]}"] = flux

    return ds


def _copy_attrs(var, default=None):
    """
    Copies attributes of variables.

    Args:
        var (xarray.DataArray):
            Variables of a xarray.Dataset
    Returns:
        var (xarray.DataArray):
            Attributes of the variable
    """

    return var.attrs.copy() if hasattr(var, "attrs") else (default or {})


def _combine_variable(
    ds_prior, ds_post, ds_prior_country, ds_post_country, species: str
):
    """
    Merges prior, posterior, gridded and country-aggregated fluxes into one dataset.

    Args:
        ds_prior(xarray.Dataset):
            Dataset with gridded prior fluxes
        ds_posterior(xarray.Dataset):
            Dataset with gridded posterior fluxes
        ds_prior_country(xarray.Dataset):
            Dataset with country-aggregated prior fluxes
        ds_posterior_country(xarray.Dataset):
            Dataset with country-aggregated posterior fluxes
        species (str):
            Species (e.g. "ch4", "co2")
    Returns:
        ds (xarray.Dataset):
            Combined dataset
    """

    ds = ds_prior.copy()
    for v, new_vars in combine_candidates.items():
        varname = f"{species}{v}"
        if varname in ds and varname in ds_post:
            prior_data = _check_for_units(varname, ds)
            post_data = _check_for_units(varname, ds_post)

            ds[new_vars[0]] = prior_data
            ds[new_vars[1]] = post_data
            ds = ds.drop_vars([varname])

        if varname in ds_prior_country and varname in ds_post_country:
            prior_data_country = _check_for_units_country(
                varname, ds_prior_country, species
            )
            post_data_country = _check_for_units_country(
                varname, ds_post_country, species
            )

            # (reg,time) -> (country,time)
            country = ds_prior_country["country"]
            prior_data_country.coords["reg"] = country
            prior_data_country = prior_data_country.rename({"reg": "country"})
            post_data_country.coords["reg"] = country
            post_data_country = post_data_country.rename({"reg": "country"})

            ds[new_vars[0]] = prior_data_country
            ds[new_vars[1]] = post_data_country

        else:
            print(f"INFO: {varname} not found in dataset.")

    return ds


def _check_for_units(varname: str, ds):
    """
    Checks and converts gridded flux units into mol m-2 s-1.

    Args:
        varname (str):
            Variable name
        ds (xarray.Dataset):
            Dataset with gridded fluxes
    Returns:
        data (xarray.DataArray):
            Gridded fluxes in units mol m-2 s-1
    """

    if varname not in ds:
        logging.warning(f"Variable {varname} not found in dataset.")
        return None

    data = ds[varname]
    if data.size == 0 or np.all(np.isnan(data)):
        logging.warning(f"Variable {varname} is empty or all NaN.")
        return None

    # Copy attrs to preserve metadata
    data.attrs = ds[varname].attrs.copy()

    if "units" in data.attrs:
        unit = data.attrs["units"]
        if "dxyp" not in ds:
            raise KeyError(
                "Dataset is missing 'dxyp' (grid cell area), required for flux conversion."
            )

        area = ds["dxyp"]
        years = ds["time"].dt.year

        if unit in unit_conversions:
            data = data * unit_conversions[unit]
            data = _pgcyr_to_mol_m2_s(data, area, years)
            data.attrs["units"] = "mol m-2 s-1"
        else:
            logging.info(
                f"No conversion applied for variable {varname} with unit '{unit}'."
            )

    return data


def _check_for_units_country(varname: str, ds, species: str):
    """
    Checks and converts country-aggregated flux units into kg s-1.

    Args:
        varname (str):
            Variable name
        ds (xarray.Dataset):
            Dataset with country-aggregated fluxes
        species (str):
            Species (e.g. "ch4", "co2")
    Returns:
        data (xarray.DataArray):
            Country-aggregated fluxes in units kg s-1
    """

    if varname not in ds:
        logging.warning(f"Variable {varname} not found in dataset.")
        return None

    data = ds[varname]
    if data.size == 0 or np.all(np.isnan(data)):
        logging.warning(f"Variable {varname} is empty or all NaN.")
        return None

    # Copy attrs to preserve metadata
    data.attrs = ds[varname].attrs.copy()

    if "units" in data.attrs:
        unit = data.attrs["units"]

        years = ds["time"].dt.year

        if unit in unit_conversions:
            data = data * unit_conversions[unit]
            data = _pgcyr_to_kg_s_tracer(data, years, species)
            data.attrs["units"] = "kg s-1"
        else:
            logging.info(
                f"No conversion applied for variable {varname} with unit '{unit}'."
            )

    return data


def _pgcyr_to_mol_m2_s(value_pgcyr, area_m2, years):
    """
    Converts PgC yr-1 into mol m-2 s-1.

    Args:
        value_pgcyr (xarray.DataArray):
            Fluxes in PgC yr-1
        area_m2 (xarray.DataArray):
            Area in m2
        years (array):
            Years of the fluxes
    Returns:
        flux (xarray.DataArray):
            Fluxes in mol m-2 s-1
    """

    grams = value_pgcyr * 1e15
    mols = grams / 12.01

    days_in_year = np.array([366 if calendar.isleap(y) else 365 for y in years])
    seconds_per_year = days_in_year * 24 * 60 * 60

    # Make this an xarray DataArray to allow broadcasting
    seconds_per_year = xr.DataArray(
        seconds_per_year, dims=["time"], coords={"time": value_pgcyr["time"]}
    )

    flux = mols / seconds_per_year / area_m2

    return flux


def _pgcyr_to_kg_s_tracer(value_pgcyr, years, species: str):
    """
    Converts PgC yr-1 into kg s-1.

    Args:
        value_pgcyr (xarray.DataArray):
            Fluxes in PgC yr-1
        years (array):
            Years of the flux record
        species (str):
            Species (e.g. "ch4", "co2")
    Returns:
        flux (xarray.DataArray):
            Fluxes in kg s-1
    """

    kilograms = value_pgcyr * 1e12
    if species == "ch4":
        kilograms_tracer = kilograms / 12.01 * 16.04
    if species == "co2":
        kilograms_tracer = kilograms / 12.01 * 44.01

    days_in_year = np.array([366 if calendar.isleap(y) else 365 for y in years])
    seconds_per_year = days_in_year * 24 * 60 * 60

    # Make this an xarray DataArray to allow broadcasting
    seconds_per_year = xr.DataArray(
        seconds_per_year, dims=["time"], coords={"time": value_pgcyr["time"]}
    )

    flux = kilograms_tracer / seconds_per_year

    return flux


def _convert_time(ds):
    """
    Converts time in seconds since 2000-01-01 into datetime64.

    Args:
        ds (xarray.Dataset):
            Dataset with time coordinate in seconds since 2000-01-01
    Returns:
        ds (xarray.Dataset):
            Dataset with time coordinate in datatime64 format
    """

    if np.issubdtype(ds["time"].dtype, np.datetime64):
        # already datetime64 — no need to convert
        return ds

    # convert seconds since 2000-01-01 to datetime64
    base = np.datetime64("2000-01-01T00:00:00", "s")
    time_seconds = ds["time"].values.astype("timedelta64[s]")
    new_time = base + time_seconds

    # set the new time coordinate, forcing ns resolution to avoid warnings
    ds = ds.assign_coords(time=new_time.astype("datetime64[ns]"))

    return ds


def _tmolyr_to_kg_s_unc_rhs(unc_rhs, time_rhs, species: str):
    """
    Converts flux uncertainties in Tmol yr-1 into kg s-1.

    Args:
        unc_rhs (series):
            Flux uncertainties in Tmol yr-1 (from "right-hand-side" run)
        time_rhs (xarray.DataArray):
            Flux time step
        species (str):
            Species (e.g. "ch4", "co2")
    Returns:
        unc_rhs (series):
            Flux uncertainties in kg s-1
    """

    # seconds per year
    year = pd.to_datetime(time_rhs.values).year
    days_in_year = np.array([366 if calendar.isleap(year) else 365])
    seconds_per_year = days_in_year * 24 * 60 * 60
    # convert Tmol/yr -> kg/s
    if species == "ch4":
        unc_rhs = unc_rhs * 16.04 * 1e9 / seconds_per_year
    if species == "co2":
        unc_rhs = unc_rhs * 44.01 * 1e9 / seconds_per_year

    return unc_rhs


def _save_dataset(ds, path: str):
    """
    Saves the dataset.

    Args:
        ds (xarray.Dataset):
            Dataset
        path (str):
            Path where the dataset is to be stored
    """

    ds.encoding.clear()
    for v in ds.variables:
        ds[v].encoding.clear()
    try:
        if os.path.exists(path):
            os.remove(path)
        # make sure parent folders exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(path, engine="netcdf4")
        print(f"✅ File saved successfully: {path}")
    except Exception as e:
        print(f"❌ Error when trying to save file {path}: {e}")
