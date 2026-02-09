"""
Script to convert the concentration ts files from CSR format to netcdf fluxy format.
It can be run by the csr_preprocess.ipynb notebook.
"""

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import os
import glob
import warnings

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

    time_vec_df = _create_ts_yi_ye_tk(start_year, end_year)

    files = [
        os.path.basename(f)
        for f in glob.glob(path_to_posterior_conc + "*." + species + ".ts")
    ]
    ids = [s[2:5] for s in files]
    print("Following files have been found:")
    print(files)

    cols = [
        "frac_time",
        "year",
        "month",
        "day", "hour",
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
        "datetime"
    ]

    da_all_posterior = pd.DataFrame(np.empty((0, len(cols))), columns=cols)
    da_all_prior = pd.DataFrame(np.empty((0, len(cols))), columns=cols)
    da_all_farfield = pd.DataFrame(np.empty((0, len(cols))), columns=cols)

    cols_csr = [
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
        "mod"
    ]

    for ff in list(range(0, len(files))):
        print("file:", ff)

        # Common csv params
        params = {
            'comment': '#',
            'delim_whitespace': True,
            'header': None,
            'names': cols_csr
        }

        # Read CSV using common params
        df_posterior_conc = pd.read_csv(path_to_posterior_conc + files[ff], **params)
        df_prior_conc = pd.read_csv(path_to_prior_conc + files[ff], **params)
        df_far_field = pd.read_csv(path_to_farfield_conc + files[ff], **params)

        # Filter far field contribution data to match the requested time range (start_year to end_year)
        sel = df_far_field["year"] <= end_year  # use only data until 2023
        df_far_field = df_far_field[sel]
        sel = df_far_field["year"] >= start_year  # use only data from and after 2006
        df_far_field = df_far_field[sel]

        # Process dataframes
        datetime_cols = ["year", "month", "day", "hour", "minute", "second"]
        df_posterior_conc = _process_dataframe(df_posterior_conc, time_vec_df, datetime_cols)
        df_prior_conc = _process_dataframe(df_prior_conc, time_vec_df, datetime_cols)
        df_far_field = _process_dataframe(df_far_field, time_vec_df, datetime_cols)

        # Merge datasets
        merged_df_posterior_conc = pd.merge(df_posterior_conc, time_vec_df, on="datetime_str", how="outer")
        merged_df_prior_conc = pd.merge(df_prior_conc, time_vec_df, on="datetime_str", how="outer")
        merged_df_far_field = pd.merge(df_far_field, time_vec_df, on="datetime_str", how="outer")

        # If there are duplicated observations, remove them all
        duplicates_df = merged_df_posterior_conc[merged_df_posterior_conc["datetime_str"].duplicated(keep=False)][
            "datetime_str"
        ]
        merged_df_posterior_conc = merged_df_posterior_conc.loc[~merged_df_posterior_conc["datetime_str"].isin(duplicates_df)]
        duplicates_dfp = merged_df_prior_conc[merged_df_prior_conc["datetime_str"].duplicated(keep=False)][
            "datetime_str"
        ]
        merged_df_prior_conc = merged_df_prior_conc.loc[~merged_df_prior_conc["datetime_str"].isin(duplicates_dfp)]
        duplicates_dffar = merged_df_far_field[
            merged_df_far_field["datetime_str"].duplicated(keep=False)
        ]["datetime_str"]
        merged_df_far_field = merged_df_far_field.loc[
            ~merged_df_far_field["datetime_str"].isin(duplicates_dffar)
        ]
        
        # Use only timesteps for merged_df_far_field that are also present
        # in the merged_df_posterior_conc DataFrame, and remove unnecessary
        # columns from merged_df_far_field_2 after merging.
        # (Far field data contains more timesteps than prior/posterior forward runs due to missing footprints and 2024 data.)
        merged_df_far_field_2 = pd.merge(
            merged_df_far_field, df_posterior_conc, on="datetime_str", how="left"
        )  # use only keys from right dataframe
        merged_df_far_field_2 = merged_df_far_field_2.drop(
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
        merged_df_far_field_2 = merged_df_far_field_2.drop(
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
        merged_df_far_field_2.columns = [
            col[:-2] if col.endswith("_x") else col for col in merged_df_far_field_2.columns
        ]

        merged_df_posterior_conc["identifier"] = ff + 1
        merged_df_prior_conc["identifier"] = ff + 1
        merged_df_far_field_2["identifier"] = ff + 1

        # Concatenate data from current file to accumulated results
        da_all_posterior = pd.concat(
            [da_all_posterior, merged_df_posterior_conc], axis=0
        )
        da_all_prior = pd.concat([da_all_prior, merged_df_prior_conc], axis=0)
        da_all_farfield = pd.concat([da_all_farfield, merged_df_far_field_2], axis=0)

    print("Saving the datasets")

    _save_dataset_conc(
        da_all_posterior, da_all_prior, da_all_farfield, species, path_to_output_conc, files, ids
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

    idx = (pd.date_range(start=start_date, end=end_date, freq="1h"),)
    df = idx[0].to_frame(
        index=False, name="datetime_str"
    )  # idx is a tuple, so get with [0] the index 0, which is the DatetimeIndex field

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


def _process_dataframe(df, time_vec_df, datetime_cols):
    """
    Processes dataframes and adds datetime identifier.

    Args:
        df (DataFrame):
            Dataframe with concentrations
        time_vec_df (DataFrame):
            Dataframe with datetime (from _create_ts_yi_ye_tk())
        datetime_cols (list):
            List with datetime column names
    Returns:
        df (Dataframe):
            Processed dataframe
    """
    
    # Convert to datetime
    dt_obj = pd.to_datetime(df[datetime_cols])
    # Add datetime and formatted string columns
    df = df.copy()
    df["flag"] = 1
    df["datetime"] = dt_obj
    df["datetime_str"] = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in dt_obj]
    df["datetime_str"] = df["datetime_str"].astype("datetime64[ns]")

    return df


def _save_dataset_conc(
    da_all_posterior, da_all_prior, da_all_farfield, species: str, path_to_output_conc: str, files, ids
):
    """
    Writes the data into a netcdf file and saves it.

    Args:
        da_all_posterior (DataFrame):
            Dataframe with posterior concentrations
        da_all_prior (DataFrame):
            Dataframe with prior concentrations
        da_all_farfield (DataFrame):
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
    ncfile.createDimension("index", len(da_all_posterior["datetime_str"]))
    ncfile.createDimension("nbnds", 2)
    ncfile.createDimension("percentile", 2)
    ncfile.createDimension("platform", len(files))
    # add variables
    times = ncfile.createVariable("time", np.float64, ("index"))
    times.units = "days since 1970-01-01"
    times.long_name = "time of mid of observation interval; UTC"
    times.standard_name = "time"
    times.axis = "T"
    times.calendar = "proleptic_gregorian"

    # Time vector for the full time series (years multiplied by number of stations)
    time_vec = pd.to_datetime(
        da_all_posterior["datetime_str"], format="%Y-%m-%d %H:%M:%S"
    )
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
    identifiers[:] = da_all_posterior["identifier"]

    obss = ncfile.createVariable("mf_observed", np.float32, ("index"))
    if species == "ch4":
        units = "ppb"
    elif species == "co2":
        units = "ppm"
    else:
        raise ValueError(f"Unsupported species: {species}. Only 'ch4' and 'co2' are supported.")
    obss.units = units
    obss.long_name = "observed_mole_fraction"
    obss[:] = da_all_posterior["obs"]

    posteriors = ncfile.createVariable("mf_posterior", np.float32, ("index"))
    posteriors.units = units
    posteriors.long_name = "a posteriori_simulated_mole_fraction"
    posteriors[:] = da_all_posterior["mod"]

    priors = ncfile.createVariable("mf_prior", np.float32, ("index"))
    priors.units = units
    priors.long_name = "a priori_simulated_mole_fraction"
    priors[:] = da_all_prior["mod"]

    farfield_prior = ncfile.createVariable("mf_bc_prior", np.float32, ("index"))
    farfield_prior.units = units
    farfield_prior.long_name = "farfield_contribution_simulated_mole_fraction"
    farfield_prior[:] = da_all_farfield["mod"]

    farfield_posterior = ncfile.createVariable("mf_bc_posterior", np.float32, ("index"))
    farfield_posterior.units = units
    farfield_posterior.long_name = "farfield_contribution_simulated_mole_fraction"
    farfield_posterior[:] = da_all_farfield["mod"]

    lats = ncfile.createVariable("latitude", np.float32, ("index"))
    lats.long_name = "latitude"
    lats.units = "degrees_north"
    lats[:] = da_all_posterior["lat"]

    lons = ncfile.createVariable("longitude", np.float32, ("index"))
    lons.long_name = "longitude"
    lons.units = "degrees_east"
    lons[:] = da_all_posterior["lon"]

    alts = ncfile.createVariable("altitude", np.float32, ("index"))
    alts.long_name = "height above ground level"
    alts.units = "m agl"
    alts[:] = da_all_posterior["alt"]

    flags = ncfile.createVariable(
        "assimilation_flag", "i4", ("index"), fill_value=-9999
    )
    flags.long_name = "indicating whether observation was used in inversion/assimilation. 0: not used; 1: used"
    flags.units = "1"
    flags[:] = da_all_posterior["flag"]

    # add global attributes
    ncfile.title = (
        "observed and simulated atmospheric " + str.upper(species) + " concentration"
    )
    ncfile.species = str.upper(species)
    ncfile.inversion = "CarboScope-Regional"
    ncfile.domain = "Europe"
    ncfile.institution = "MPI Biogeochemistry Jena"
    ncfile.close()
    print(path_to_output_conc + "HAS BEEN CREATED FOR: " + str(len(files)) + " STATIONS")
