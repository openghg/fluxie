"""
Script to convert the concentration ts files from CSR format to netcdf fluxy format.
It can be run by the csr_preprocess.ipynb notebook.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import netCDF4 as nc
from netCDF4 import Dataset
import xarray as xr
import os
import glob
import warnings
import sys

warnings.filterwarnings("ignore")


def preprocess_conc(
    path_to_prior_conc: str,
    path_to_posterior_conc: str,
    path_to_farfield_conc: str,
    path_to_output_conc: str,
    start_year: int,
    end_year: int,
    species: str,
):
    """
    Main function, which converts the CSR concentration time series into the fluxy format.

    Args:
        path_to_prior_conc (str):
            Full path to CSR prior concentration file
        path_to_posterior_conc (str):
            Full path to CSR posterior concentration file
        path_to_farfield_conc (str):
            Full path to CSR farfield contribution file
        path_to_output_conc (str):
            Full path to the directory where the results in fluxy format are to be stored
        start_year (int):
            First year of time series
        end_year (int):
            Last year of time series
        species (str):
            Species (e.g. "ch4", "co2")
    """

    years = np.arange(start_year, end_year)
    time_vec_df = pd.DataFrame()
    time_vec_df = _create_ts_yi_ye_tk(start_year, end_year)

    files = [
        os.path.basename(f)
        for f in glob.glob(path_to_posterior_conc + "*." + species + ".ts")
    ]
    ids = [s[2:5] for s in files]
    print(files)
    da_all = pd.DataFrame(
        np.empty((0, 16)),
        columns=[
            "frac_time",
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "lat",
            "lon",
            "alt",
            "obs",
            "std",
            "mod",
            "identifier",
            "flag",
            "datetime",
        ],
    )
    da_allp = pd.DataFrame(
        np.empty((0, 16)),
        columns=[
            "frac_time",
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "lat",
            "lon",
            "alt",
            "obs",
            "std",
            "mod",
            "identifier",
            "flag",
            "datetime",
        ],
    )
    da_allfar = pd.DataFrame(
        np.empty((0, 16)),
        columns=[
            "frac_time",
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "lat",
            "lon",
            "alt",
            "obs",
            "std",
            "mod",
            "identifier",
            "flag",
            "datetime",
        ],
    )

    for ff in list(range(0, len(files))):
        print("file:", ff)
        df = pd.read_csv(
            path_to_posterior_conc + files[ff],
            comment="#",
            delim_whitespace=True,
            header=None,
            names=[
                "frac_time",
                "year",
                "month",
                "day",
                "hour",
                "minute",
                "second",
                "lat",
                "lon",
                "alt",
                "obs",
                "std",
                "mod",
            ],
        )
        # parse_dates={"date_time": ["year", "month", "day", "hour", "minute", "second"]}
        dfp = pd.read_csv(
            path_to_prior_conc + files[ff],
            comment="#",
            delim_whitespace=True,
            header=None,
            names=[
                "frac_time",
                "year",
                "month",
                "day",
                "hour",
                "minute",
                "second",
                "lat",
                "lon",
                "alt",
                "obs",
                "std",
                "mod",
            ],
        )
        dffar = pd.read_csv(
            path_to_farfield_conc + files[ff],
            comment="#",
            delim_whitespace=True,
            header=None,
            names=[
                "frac_time",
                "year",
                "month",
                "day",
                "hour",
                "minute",
                "second",
                "lat",
                "lon",
                "alt",
                "obs",
                "std",
                "mod",
            ],
        )

        m_f = np.array(df)
        m_fp = np.array(dfp)
        m_far = np.array(dffar)
        df = pd.DataFrame(df)
        dfp = pd.DataFrame(dfp)
        dffar = pd.DataFrame(dffar)

        # ---------This has an effect and far field conc has to cut for 2006-2023! (far field contrib. are maybe not only for 2006-2023)
        sel = dffar["year"] <= end_year  # use only data until 2023
        dffar = dffar[sel]
        sel = dffar["year"] >= start_year  # use only data from and after 2006
        dffar = dffar[sel]

        df["flag"] = 1
        dfp["flag"] = 1
        dffar["flag"] = 1

        df_datetime_obj = pd.to_datetime(
            df[["year", "month", "day", "hour", "minute", "second"]]
        )  # is now apply "to_datetime" and get a object from
        dfp_datetime_obj = pd.to_datetime(
            dfp[["year", "month", "day", "hour", "minute", "second"]]
        )
        dffar_datetime_obj = pd.to_datetime(
            dffar[["year", "month", "day", "hour", "minute", "second"]]
        )

        df["datetime"] = df_datetime_obj
        dfp["datetime"] = dfp_datetime_obj
        dffar["datetime"] = dffar_datetime_obj

        list_df = []
        list_dfp = []
        list_dffar = []
        for i in range(0, df_datetime_obj.size):
            list_df.append(df_datetime_obj[i].strftime("%Y-%m-%d %H:%M:%S"))
        for i in range(0, dfp_datetime_obj.size):
            list_dfp.append(dfp_datetime_obj[i].strftime("%Y-%m-%d %H:%M:%S"))
        for i in range(0, dffar_datetime_obj.size):
            list_dffar.append(dffar_datetime_obj[i].strftime("%Y-%m-%d %H:%M:%S"))

        # ---now add columns to dataframes
        df[
            "datetime_str"
        ] = list_df  # define a new column in df with name 'datetime_str' and fill this with list_df
        dfp["datetime_str"] = list_dfp
        dffar["datetime_str"] = list_dffar

        # ---now according merging problem get info about dtypes
        df["datetime_str"] = df["datetime_str"].astype("datetime64[ns]")
        dfp["datetime_str"] = dfp["datetime_str"].astype("datetime64[ns]")
        dffar["datetime_str"] = dffar["datetime_str"].astype("datetime64[ns]")

        # ----------------------find doublicates if needed-----------------------
        # before merging; here to check, if also on 31.12. are prior, post, and LBC are general available
        df1 = dffar
        df2 = df
        # find rows in d1 which have id, which are not available in d2
        # you could use isin, with negation operator, so that we filter out the rows in df1 that have ids that also exist in df2:
        # out = df1[~df1['id'].isin(df2['id'])]
        out = df1[~df1["datetime_str"].isin(df2["datetime_str"])]

        merged_df = pd.merge(df, time_vec_df, on="datetime_str", how="outer")
        merged_dfp = pd.merge(dfp, time_vec_df, on="datetime_str", how="outer")
        merged_dffar = pd.merge(dffar, time_vec_df, on="datetime_str", how="outer")
        m_df_df = np.array(merged_df)
        m_df_dfp = np.array(merged_dfp)
        m_df_dffar = np.array(merged_dffar)

        # if there are duplicated observations, remove them all
        duplicates_df = merged_df[merged_df["datetime_str"].duplicated(keep=False)][
            "datetime_str"
        ]
        merged_df = merged_df.loc[~merged_df["datetime_str"].isin(duplicates_df)]
        duplicates_dfp = merged_dfp[merged_dfp["datetime_str"].duplicated(keep=False)][
            "datetime_str"
        ]
        merged_dfp = merged_dfp.loc[~merged_dfp["datetime_str"].isin(duplicates_dfp)]
        duplicates_dffar = merged_dffar[
            merged_dffar["datetime_str"].duplicated(keep=False)
        ]["datetime_str"]
        merged_dffar = merged_dffar.loc[
            ~merged_dffar["datetime_str"].isin(duplicates_dffar)
        ]

        # Now use only timesteps for merged_dffar  which are also in the merged_df Dataframe
        # and delete in merged_dffar2 the not needed columns after merging
        # (in the far field are more times as in the prior,posterior fwd runs (reason missing footprints, and for 2024 data)
        merged_dffar2 = pd.merge(
            merged_dffar, df, on="datetime_str", how="left"
        )  # use only keys from rigth dataframe
        merged_dffar2 = merged_dffar2.drop(
            columns=[
                "frac_time_y",
                "year_y",
                "month_y",
                "day_y",
                "hour_y",
                "minute_y",
                "second_y",
            ],
            axis=1,
        )
        merged_dffar2 = merged_dffar2.drop(
            columns=[
                "lat_y",
                "lon_y",
                "alt_y",
                "obs_y",
                "std_y",
                "mod_y",
                "flag_y",
                "datetime_y",
            ],
            axis=1,
        )
        merged_dffar2.columns = [
            col[:-2] if col.endswith("_x") else col for col in merged_dffar2.columns
        ]

        merged_df["identifier"] = ff + 1
        merged_dfp["identifier"] = ff + 1
        merged_dffar2["identifier"] = ff + 1
        da_all = pd.concat(
            [da_all, merged_df], axis=0
        )  # this add the values of the current file to da_all (all combined)
        da_allp = pd.concat([da_allp, merged_dfp], axis=0)
        da_allfar = pd.concat([da_allfar, merged_dffar2], axis=0)

    print("_save_dataset")

    _save_dataset_conc(
        da_all, da_allp, da_allfar, species, path_to_output_conc, files, ids
    )


def _create_ts_yi_ye_tk(ystart: int, yend: int):
    """
    Creates time series of hours between start and end year.

    Args:
        ystart (int):
            First year of time series
        yend (int):
            Last year of time series
    Returns:
        df (DataFrame):
            Dataframe with one row per hour between start and end year
    """

    start_date = "1/1/" + str(ystart)
    end_date = "1/1/" + str(yend + 1)

    df = pd.DataFrame()
    idx = (pd.date_range(start=start_date, end=end_date, freq="1h"),)
    df = idx[0].to_frame(
        index=False, name="datetime_str"
    )  # idx is a typle, so get with [0] the index 0, which is the DatetimeIndex field

    return df


def _calc_time_delta(time):
    """
    Calculates time difference compared to 1970-01-01 in days.

    Args:
        time (Series):
            Time vector
    Returns:
        time_delta (Series):
            Time difference compared to 1970-01-01 in days
    """

    time_delta = (time - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "D")

    return time_delta


def _save_dataset_conc(
    da_all, da_allp, da_allfar, species: str, path_to_output_conc: str, files, ids
):
    """
    Writes the data into a netcdf file and saves it.

    Args:
        da_all (DataFrame):
            Dataframe with prior concentrations
        da_allp (DataFrame):
            Dataframe with posterior concentrations
        da_allfar (DataFrame):
            Dataframe with farfield contributions
        species (str):
            Species (e.g. "ch4", "co2")
        path_to_output_conc:
            Full path to the directory where the results in fluxy format are to be stored
        files (list):
            List with file names of the concentration time series
        ids (list):
            List with station codes
    """

    # ----------------------create nc files and define required dims---------------------------------
    # define dimensions
    ncfile = Dataset(path_to_output_conc, mode="w", format="NETCDF4")
    indexdim = ncfile.createDimension("index", len(da_all["datetime_str"]))
    nbndsdim = ncfile.createDimension("nbnds", 2)
    percentiledim = ncfile.createDimension("percentile", 2)
    platformdim = ncfile.createDimension("platform", len(files))
    # add variables
    times = ncfile.createVariable("time", np.float64, ("index"))
    times.units = "days since 1970-01-01"
    times.long_name = "time of mid of observation interval; UTC"
    times.standard_name = "time"
    times.axis = "T"
    times.calendar = "proleptic_gregorian"

    time_vec = pd.to_datetime(
        da_all["datetime_str"], format="%Y-%m-%d %H:%M:%S"
    )  # this is the full time series 2006-2023 multipl with stat
    delta_dd = _calc_time_delta(time_vec)  # delta_dd stands for delta_days
    delta_dd_df = delta_dd.to_frame()
    times[:] = delta_dd_df["datetime_str"]

    platforms = ncfile.createVariable("platform", str, ("platform",))
    platforms.long_name = (
        "identifier of observing platform; e.g., 3 letter ID for surface in-situ sites"
    )
    for p in list(range(0, len(ids))):
        platforms[p] = ids[p]

    identifiers = ncfile.createVariable("number_of_identifier", "int16", ("index"))
    identifiers.long_name = "Index of identifier of observing platform"
    identifiers.units = "1"
    identifiers[:] = da_all["identifier"]

    obss = ncfile.createVariable("mf_observed", np.float32, ("index"))
    if species == "ch4":
        units = "ppb"
    if species == "co2":
        units = "ppm"
    obss.units = units
    obss.long_name = "observed_mole_fraction"
    obss[:] = da_all["obs"]

    posteriors = ncfile.createVariable("mf_posterior", np.float32, ("index"))
    posteriors.units = units
    posteriors.long_name = "aposteriori_simulated_mole_fraction"
    posteriors[:] = da_all["mod"]

    priors = ncfile.createVariable("mf_prior", np.float32, ("index"))
    priors.units = units
    priors.long_name = "apriori_simulated_mole_fraction"
    priors[:] = da_allp["mod"]

    farfield_prior = ncfile.createVariable("mf_bc_prior", np.float32, ("index"))
    farfield_prior.units = units
    farfield_prior.long_name = "farfield_contribution_simulated_mole_fraction"
    farfield_prior[:] = da_allfar["mod"]

    farfield_posterior = ncfile.createVariable("mf_bc_posterior", np.float32, ("index"))
    farfield_posterior.units = units
    farfield_posterior.long_name = "farfield_contribution_simulated_mole_fraction"
    farfield_posterior[:] = da_allfar["mod"]

    lats = ncfile.createVariable("latitude", np.float32, ("index"))
    lats.long_name = "latitude"
    lats.units = "degrees_north"
    lats[:] = da_all["lat"]

    lons = ncfile.createVariable("longitude", np.float32, ("index"))
    lons.long_name = "longitude"
    lons.units = "degrees_east"
    lons[:] = da_all["lon"]

    alts = ncfile.createVariable("altitude", np.float32, ("index"))
    alts.long_name = "height above ground level"
    alts.units = "m agl"
    alts[:] = da_all["alt"]

    flags = ncfile.createVariable(
        "assimilation_flag", "i4", ("index"), fill_value=-9999
    )
    flags.long_name = "indicating whether observation was used in inversion/assimilation. 0: not used; 1: used"
    flags.units = "1"
    flags[:] = da_all["flag"]

    # add global attributes
    ncfile.title = (
        "observed and simulated atmospheric " + str.upper(species) + " concentration"
    )
    ncfile.species = str.upper(species)
    ncfile.inversion = "CarboScope-Regional"
    ncfile.domain = "Europe"
    ncfile.institution = "MPI Biogeochemistry Jena"
    ncfile.close()
    print(path_to_output_conc + "HAS CREATED FOR: " + str(len(files)) + " STATIONS")
