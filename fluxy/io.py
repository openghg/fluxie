import itertools
import json
import logging
import os
import re
import string
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import yaml
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from fluxy import config
from fluxy.operators.flux_align_dataset import align_time
from fluxy.operators.regions import extract_region_flux
from fluxy.operators.select import slice_flux, get_intake_height, get_site_index
from fluxy.operators.flux_align_dataset import align_time
from fluxy.types import DataType, DataTypes, file_pattern

logger = logging.getLogger(__name__)


# Naming of variables that might have changed in the past
legacy_names: dict[str, str] = {
    "country_flux_total_prior": "flux_total_prior_country",
    "country_flux_total_posterior": "flux_total_posterior_country",
    "percentile_country_flux_total_prior": "percentile_flux_total_prior_country",
    "percentile_country_flux_total_posterior": "percentile_flux_total_posterior_country",
    "covariance_country_flux_total_posterior": "covariance_flux_total_posterior_country",
    "nsite": "number_of_identifier",
    "sitenames": "platform",
    "Yobs": "mf_observed",
    "Yapriori": "mf_prior",
    "Yapost": "mf_posterior",
    "YaprioriBC": "mf_bc_prior",
    "YapostBC": "mf_bc_posterior",
    "Yapriori_bias": "mf_bias_prior",
    "Yapost_bias": "mf_bias_posterior",
    "YaprioriOUTER": "mf_outer_prior",
    "YapostOUTER": "mf_outer_posterior",
    "qYapriori": "percentile_mf_prior",
    "qYapost": "percentile_mf_posterior",
    "uYtotal": "stdev_mf_total",
    "uYobs_repeatability": "stdev_mf_observed_repeatability",
    "uYobs_variability": "stdev_mf_observed_variability",
    "uYmod": "stdev_mf_model",
}


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
        raise FileNotFoundError(f"Cannot find {filepath}.")

    with open(filepath, "r") as f:
        json_data = json.load(f)

    return json_data


def read_yaml(filepath: os.PathLike) -> dict[str, dict]:
    """
    Reads yaml file.

    Args:
        filepath (str or Path):
            Path to yaml file including filename.
    Returns:
        yaml_data (dictionary of dictionaries):
            Dictionary with data read from filepath.
    """

    filepath = Path(filepath)

    if not filepath.is_file():
        raise FileNotFoundError(f"Cannot find {filepath}.")

    with open(filepath, "r") as f:
        yaml_data = yaml.safe_load(f)

    return yaml_data


def read_config_files(
    configs_dir: os.PathLike | None = None,
) -> dict[str, dict]:
    """
    Reads all configuration files.

    Returns:
        data_dict (dictionary of dictionaries):
            Dictionary with keys equal to files basename (without extension).
            Each key points to a dictionary with the data from each config file.
    """

    if configs_dir is None:
        # Get location of config files
        parent_dir = Path(__file__).parent.parent
        configs_dir = parent_dir / "configs"
    else:
        configs_dir = Path(configs_dir)

    logger.info(f"Reading config files from {configs_dir}")

    read_func = {
        ".json": read_json,
        ".yaml": read_yaml,
        ".yml": read_yaml,
    }

    # List of files to be read
    files = itertools.chain(*(configs_dir.glob(f"*{ext}") for ext in read_func.keys()))

    # Read config files
    data_dict = {}
    for file in files:
        data = read_func[file.suffix](file)
        filename = file.stem
        data_dict[filename] = data

    # Join dictionaries from regions_info config
    regions_info = data_dict.get("regions_info", {})
    if "regions" in regions_info.keys():
        if "country_codes" not in regions_info.keys():
            regions_info["country_codes"] = {}
        regions_info["country_codes"].update(regions_info["regions"])

    return data_dict


def make_template(*args: str | tuple[str], suffix: str | None = None) -> str:
    """
    Build filename template based on input arguments.
    Single arguments are expected to indicate directories and are joined with "/".
    Tuple elements are expected to indicate filename parts and are joined with "_".

    Args:
        args (tuple of str or tuples):
            Tuple with all input arguments (folder directories and filename parts).
        suffix (str):
            If provided, a suffix is added to the filename template.

    Returns:
        template (str):
            String composed of template elements (e.g. "{arg1}/{arg2}{suffix}")
    """

    parts = []
    for x in args:
        if isinstance(x, str):
            parts.append(f"{{{x}}}")
        else:
            # tuple (or list etc...)
            part = "_".join(f"{{{y}}}" for y in x)
            parts.append(part)

    # Join directories and filenames
    template = "/".join(parts)

    # Add suffix
    if suffix is not None:
        template += f"{{{suffix}}}"

    return template


def fill_template(template: str, **kwargs: str) -> str:
    """
    Replace filename template with respective input arguments.

    Args:
        template (str):
            Filename template. Output from make_template.
        kwargs (str):
            Elements to fill the template.

    Returns:
        filepath (str):
            Full path to file given by template with elements replaced by kwargs.
    """

    # add place holders for missing args
    template_fields = [x[1] for x in string.Formatter().parse(template)]
    for field in template_fields:
        # check for option params
        if field.endswith("?") and field[:-1] in kwargs:
            kwargs[field] = kwargs.pop(field[:-1])

        # handle missing arguments: remove optional args and
        # create placeholders for the rest
        if field not in kwargs:
            bracket_name = f"{{{field}}}"
            if field.endswith("?"):
                # optional, so remove field, with a bunch of cases to handle separators
                template = template.replace(bracket_name + "_", "")
                template = template.replace("_" + bracket_name, "")
                template = template.replace(bracket_name + "/", "")
                template = template.replace("/" + bracket_name, "")
                template = template.replace(bracket_name, "")
            else:
                kwargs[field] = bracket_name

    return template.format(**kwargs)


def get_filename(
    model: str,
    species: str,
    period: str,
    file_pattern: str,
    config_data: dict[str, dict],
    data_dir: os.PathLike | str,
    read_standard_run: bool,
    filepath_kwargs: dict[str, str],
) -> Path:
    """
    Get complete path to the output file.

    Args:
        model (str):
            Key specifying model name, e.g. 'elris'
        species (str):
            Gas species, e.g. 'ch4'.
        period (str):
            Inversion period as specified in the model filename.
        file_pattern (str):
            String that should be added at the end of the filename.
        config_data (dict of dict):
            Dictionary with settings read from config file.
            Use config filenames as keys.
        data_dir (str):
            Path to top data directory.
        read_standard_run (bool):
            If True, constructs filename from models_info['standard_run'][<model_run_keys>].
            If entry "<model_run_keys>" don't exist, constructs filename from items in "<run_keys>".
            If entry "<run_keys>" don't exist, constructs filename from items in "default".
        filepath_kwargs (dict of str):
            Dictionary with filename parameters (key options: "data_dir", "model_dir", "species_dir", "sub_dir", "model_name")
            If missing, filename parameters are deduced from model.

    Returns:
        filepath (Path):
            Complete path to output file.
    """

    models_info = config_data.get("models_info", {})

    # Get base model name (first keyword of model)
    sub_dir, model_name = os.path.split(model)
    base_model_name, *run_keys = model_name.split("_")
    run_keys = "_".join(run_keys)

    # Get model name
    if model_name not in filepath_kwargs:
        # Get model name from standard_run dictionary
        if read_standard_run:
            logger.info(
                f"Getting {model_name} run key from models_info['standard_run']"
            )

            if (models_info is not None) and (
                all_standard_run_dict := models_info.get("standard_run")
            ):
                standard_run_dict = all_standard_run_dict.get(model_name, {})
                standard_run_dict_key = all_standard_run_dict.get(run_keys, {})
                standard_run_dict_default = all_standard_run_dict.get("default", {})

                if species in standard_run_dict:
                    model_name = f"{base_model_name}_{standard_run_dict[species]}"
                elif species in standard_run_dict_key:
                    model_name = f"{base_model_name}_{standard_run_dict_key[species]}"
                elif species in standard_run_dict_default:
                    model_name = (
                        f"{base_model_name}_{standard_run_dict_default[species]}"
                    )
                else:
                    logger.warning(
                        f"No standard run provided for {species}, neither in '{model_name}', '{run_keys}' nor in 'default'. Please update variable 'standard_run' in models_info.json. Trying to find {model} instead."
                    )
            else:
                logger.warning(
                    f"Config file models_info.json does not exist or variable 'standard_run' is not defined. Please update models_info.json. Trying to find {model} instead."
                )

        # Replace parameter tags by dict values in config
        name_tags = model_name.split("_")
        filename_tags = models_info.get("filename_tags", None)
        if filename_tags is not None:
            for i, param in enumerate(name_tags):
                string_in_file = filename_tags.get(param, None)

                if string_in_file is not None:
                    string_in_file = string_in_file.replace(
                        "<model>", base_model_name.lower()
                    )
                    name_tags[i] = string_in_file

        # Build model name
        model_name = "_".join(name_tags)

    # Get species name
    species_print = species
    if (
        (species_names := models_info.get("species_name"))
        and (model_species := species_names.get(base_model_name))
        and (species_tag := model_species.get(species))
    ):
        species_print = species_tag

    # Get file pattern
    if not period and file_pattern.startswith("_"):
        # Remove leading underscore if no period is given
        file_pattern = file_pattern[1:]

    # Hard-coded filename template
    filepath_temp = make_template(
        "data_dir",
        "model_dir",
        "species_dir",
        "sub_dir?",
        ("model_name", "species_print", "period"),
        suffix="suffix",
    )

    # Define dictionary with template arguments
    fill_kwargs = {
        "data_dir": data_dir,
        "model_dir": base_model_name,
        "species_dir": species,
        "sub_dir": sub_dir,
        "model_name": model_name,
    }

    fill_kwargs.update(
        {
            **filepath_kwargs,
            "species_print": species_print,
            "period": period,
            "suffix": file_pattern,
        },
    )

    # Get full path to file
    filepath = fill_template(
        filepath_temp,
        **fill_kwargs,
    )

    return Path(filepath)


def read_model_output(
    data_dir: os.PathLike,
    file_type: DataType,
    species: str,
    models: list[str],
    config_data: dict[str, dict] = {},
    period: str | list[str | None] | None = None,
    add_sites_to_flux: bool = False,
    read_standard_run: bool = False,
    model_filepath_dict: dict[str, dict] = {},
) -> dict[str, xr.Dataset]:
    """
    Extracts mole fraction or flux timeseries data from each model.

    Args:
        data_dir (str):
            Path to top data directory.
        file_type (DataType):
            DataType that indicate file to read (flux, concentration or eddy_flux)
        species (str):
            Gas species, e.g. 'ch4'.
        models (list of str):
            Model name tags specifying model runs,
            i.e. '<inversionModel>_<optional_identifying_tags>', preceded by subdirectory if applicable,
            e.g. ['InTEM_NAME_EUROPE_EDGAR','ELRIS_NAME_EUROPE_EDGAR']
        config_data (dict of dict):
            Dictionary with settings read from config file.
            Use config filenames as keys.
        period (str or list of str):
            Inversion period as specified in the model filename.
            If it is a string, the same period is considered for all models.
            If it is a list, one value per model must be specified, e.g. ['monthly','yearly']
        add_sites_to_flux (bool):
            If true, add sites variable to flux dataset.
        read_standard_run (bool):
            If True, constructs filename from models_info['standard_run'][<model_run_keys>].
            If entry "<model_run_keys>" don't exist, constructs filename from items in "<run_keys>".
            If entry "<run_keys>" don't exist, constructs filename from items in "default".
        model_filepath_dict (dict of dict):
            Dictionary with the elements of models as keys and a dictionary of filename parameters as values.
            If not provided, the filename parameters are deduced from models.

    Returns:
        ds_all (dictionary of datasets):
            xarray dataset read directly from each model's mole fraction netCDF.
    """
    file_type = DataTypes(file_type)

    if period is None and file_type in (DataTypes.FLUX, DataTypes.CONCENTRATION):
        # Default period
        period = "yearly"

    if isinstance(period, str | None):
        period = [period] * len(models)

    elif len(period) != len(models):
        raise ValueError(
            f"period must be None, a string or a list of the same length as models."
        )

    ds_all = {}

    for i, m in enumerate(models):
        period_str = period[i] or ""
        filepath_kwargs = model_filepath_dict.get(m, {})
        filepath = get_filename(
            m,
            species,
            period_str,
            file_pattern(file_type),
            config_data,
            data_dir,
            read_standard_run,
            filepath_kwargs,
        )

        # Check if file exists
        if not filepath.is_file():
            #  alternative filename with _flux ending
            if file_type == DataTypes.FLUX:
                logger.warning(
                    f"Cannot find {file_type.value} file: {filepath}. Will try alternative name."
                )
                filepath = get_filename(
                    m,
                    species,
                    period_str,
                    file_pattern(file_type, alternative=True),
                    config_data,
                    data_dir,
                    read_standard_run,
                    filepath_kwargs,
                )
            if not filepath.is_file():
                logger.warning(f"Cannot find {file_type.value} file: {filepath}.")
                continue

        # Read file
        logger.info(f"Reading {file_type} file: {filepath}")
        ds_all[m] = xr.open_dataset(filepath)

        # Fix variables and attributes
        ds_all[m] = edit_vars_and_attributes(
            ds_all[m],
            m,
            period_str,
            file_type,
            config_data.get("regions_info", {}),
            config_data.get("site_info", {}),
            species=species,
        )

        # Add sites variable to flux dataset
        if add_sites_to_flux and file_type == DataTypes.FLUX:
            ds_all[m] = add_sites_var(ds_all[m], filepath, m, period[i], config_data)

        # Overwrite species attributes
        current_species = ds_all[m].attrs.get("species", "not set")
        if current_species != species:
            logger.info(
                f"'species' attribute in dataset {m} ({current_species}) differs from species {species}. It is overwritten."
            )
            ds_all[m].attrs["species"] = species

    return ds_all


def read_flux_total_fgases(
    data_dir: str,
    species: str,
    models: str | list,
    config_data: dict[str, dict],
    regions: list | str,
    start_date: str,
    end_date: str,
    period: str = "yearly",
    unit: str = "Tg CO2-eq yr-1",
) -> dict[str, xr.Dataset]:
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
            Model name tags specifying model runs,
            i.e. models[i] = '<inversionModel>_<run_key>', preceded by subdirectory if applicable, e.g. ['InTEM','ELRIS'].
            The model name tag for each species is taken from standard_run[models[i]] in models_info.json, preceded by <inversionModel>.
            If standard_run[models[i]] does not exist in models_info.json, standard_run["default"] is used.
        regions (list of str):
            Region names used to extract fluxes. Only these regions can then be plotted.
        config_data (dict of dict):
            Dictionary with settings read from config file.
            Use config filenames as keys.
        start_date (str):
            Date to slice data from, e.g. '2021-01-01'
        end_date (str):
            Date to slice data to, e.g. '2022-01-01' would include all
            data up to 2021-12-31.
        period (str or list of str):
            Inversion period as specified in the model filename.
            If it is a string, the same period is considered for all models.
            If it is a list, one value per model must be specified, e.g. ['monthly','yearly']
        unit (str):
            unit in which to put the dataset. Must be in CO2-eq

    Returns:
        ds_all (dictionary of datasets):
            xarray dataset read directly from each model's flux netCDF.
    """

    all_species = (
        config_data["species_info"]
        .get(species, {"list_species": None})
        .get("list_species", None)
    )
    if all_species is None:
        raise ValueError(
            f"No list of species was found in the config_data for {species}. "
            + f"If config_data was created with read_config_files from fluxy/io.py, update configs/species_info.json."
        )
    if "CO2-eq" not in unit:
        raise ValueError("Unit should be in CO2-eq.")

    # Update parameters
    date_message = (
        " If this fails with an error message related to region_time dimensions, check the availablility\n"
        + "of data from all models for all timestamps.\n"
        + "To fix this error, set start_date and end_date as lists with the correct start and end times\nfor each model."
    )
    if isinstance(start_date, str):
        logger.info(" Using same start date for all models")
        logger.info(date_message)
        start_date = [start_date] * len(models)
    if isinstance(end_date, str):
        logger.info(" Using same end date for all models")
        logger.info(date_message)
        end_date = [end_date] * len(models)

    if isinstance(period, str):
        period = [period] * len(models)
    if len(period) != len(models):
        raise ValueError(
            f"period must be a string or a list of the same length as models."
        )

    if isinstance(regions, str):
        regions = [regions]

    missing_species = {model: list() for model in models}

    # Extract data by species, model and region
    ds_all = {region: {model: list() for model in models} for region in regions}
    for species_p in all_species:
        # read and slice dataset for each species
        ds_in = read_model_output(
            data_dir,
            "flux",
            species_p,
            models,
            config_data,
            period,
            read_standard_run=True,
        )

        for model in models:
            if model not in ds_in.keys():
                missing_species[model].append(f"{species_p} ({model})")

        ds_in = slice_flux(
            ds_in,
            config_data,
            start_date,
            end_date,
            species_p,
            country_flux_units_print=unit,
        )

        # extract regions
        for region in regions:
            ds_all_region = extract_region_flux(
                ds_in, region, config_data["regions_info"], keep_country_dim=True
            )
            for model in ds_all_region.keys():
                ds_all[region][model].append(ds_all_region[model])

    # Sum species datasets by region and model to create output
    ds_output = create_flux_total_fgases(ds_all, species, regions, models)

    # print messages about used config
    messages_ordered_by_model = list()
    for model in models:
        if missing_species[model]:
            messages_ordered_by_model.append(
                [
                    logger.warning,
                    f" Model {model} is missing species: {missing_species[model]}",
                ]
            )
        else:
            messages_ordered_by_model.append(
                [logger.info, f" All species succesfully read for {model}!"]
            )
    for message in messages_ordered_by_model:
        message[0](message[1])

    logger.info(
        " To change the files used as the standard for each HFC/PFC, edit 'standard_run' in models_info.json"
    )

    return ds_output


def create_flux_total_fgases(ds_all, species, regions, models):
    """
    Sum species datasets by region and model to create output.

    Args:
        ds_all (dictionnary of dictionnary of list of xarray datasets):
            First keys are the regions, second the model, the list contains all the data for the species to be summed.
        species (str):
            'all_hfc' or 'all_pfc'
        models (list of str):
            Model name tags specifying model runs,
            i.e. '<inversionModel>_<optional_identifying_tags>', preceded by subdirectory if applicable,
            e.g. ['InTEM_NAME_EUROPE_EDGAR','ELRIS_NAME_EUROPE_EDGAR']
        regions (list of str):
            Region names used to extract fluxes. Only these regions can then be plotted.

    Returns:
        ds_output (dictionary of datasets):
            dictionnary of xarray datasets ready to be used with fluxy plot methods.
    """
    ds_output = {}
    for model in models:
        ds_list = []
        for region in regions:
            ds_tmp = xr.concat(
                align_time(ds_all[region][model]),
                dim="species",
                combine_attrs="drop_conflicts",
            )

            ds_summed = [
                ds_tmp[["prior", "posterior"]].sum(dim="species", keep_attrs=True),
            ]
            for var in ["prior", "posterior"]:
                ds_unc = np.sqrt(
                    ((ds_tmp[[f"{var}_lower", f"{var}_upper"]] - ds_tmp[var]) ** 2).sum(
                        dim="species", keep_attrs=True
                    )
                )
                ds_unc[f"{var}_lower"] = ds_summed[0][var] - ds_unc[f"{var}_lower"]
                ds_unc[f"{var}_upper"] = ds_summed[0][var] + ds_unc[f"{var}_upper"]
                ds_summed.append(ds_unc)
            ds_list.append(xr.merge(ds_summed, combine_attrs="no_conflicts"))

        ds_tmp = xr.concat(ds_list, dim="country", combine_attrs="no_conflicts")
        ds_tmp.attrs["species"] = species

        ds_output[model] = ds_tmp
    return ds_output


def load_countries_shape(region_bounds: tuple = ()) -> gpd.geodataframe:
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
        "Norway": "NOR",
        "Kosovo": "KOS",
        "France": "FRA",
        "Indian Ocean Ter.": "IOT",
    }

    for name, iso_a3 in name_to_iso_a3_mapping.items():
        gdf.loc[gdf["NAME"] == name, "ISO_A3"] = iso_a3

    # If a region is specified, filter the GeoDataFrame
    if region_bounds:
        min_lon, max_lon, min_lat, max_lat = region_bounds
        gdf = gdf.cx[min_lon:max_lon, min_lat:max_lat]

    return gdf


def edit_vars_and_attributes(
    ds: xr.Dataset,
    model: str,
    frequency: str,
    file_type: DataType,
    regions_info: dict[str, str],
    site_info: dict[str, dict],
    species: str | None = None,
) -> xr.Dataset:
    """
    Edit dataset variables and attributes.
    This function would not be needed if all files complied with the data format.

    Args:
        ds (xarray dataset):
            xarray dataset with model data.
        model (str):
            Model name tag corresponding to ds,
            i.e. '<inversionModel>_<optional_identifying_tags>', preceded by subdirectory if applicable
        frequency (str):
            Frequency of the inversion results present in the dataset.
            Options for "monthly" and "yearly".
        file_type (str):
            Output file type.
            See :py:class:`fluxy.types.DataType` for options.
        regions_info (dict of str):
            Dictionary with country and region names (read from config file).
        species (str, optional):
            Gas species, e.g. 'ch4'. If None, no species attribute is added.

    Returns:
        ds (xarray dataset):
            xarray dataset with updated variables and attributes.
    """

    # Add inversion frequency to global attributes
    if "frequency" not in ds.attrs:
        ds.attrs["frequency"] = frequency

    # Rename legacy variables
    name_dict = {
        var: legacy_names[var]
        for var in itertools.chain(
            ds.data_vars.keys(), ds.coords.keys(), ds.sizes.keys()
        )
        if var in legacy_names
    }

    ds = ds.rename(name_dict)

    # Get model name
    filename_tags = os.path.basename(model)
    m0 = filename_tags.split("_")[0].lower()

    # check the species
    if species is not None:
        if "species" not in ds.attrs:
            ds.attrs["species"] = species
        elif ds.attrs["species"] != species:
            logger.info(
                f"Species {ds.attrs['species']} in dataset does not match species {species} in model {model}."
            )

    file_type = DataTypes(file_type)

    # Fix flux dataset
    if file_type == DataTypes.FLUX:

        # Apply model specific corrections
        if m0 in ("elris", "flexinvert"):
            # Fix for legacy files
            if "countrynumber" in ds.dims.keys():
                ds["country"] = ds["country"].astype("str")
                ds = ds.set_index(countrynumber="country").rename(
                    {"countrynumber": "country"}
                )

        elif m0 == "enkf":
            period = np.median(ds.time.values[1:] - ds.time.values[:-1]).astype(
                "timedelta64[D]"
            )
            if abs(period - np.timedelta64(30, "D")) < 3:
                ds["time"] = ds.time.values + np.timedelta64(15, "D")

        elif m0 == "intem":
            # Easy fix for InTEM ("units" attribute is wrongly set to "unit")
            vars_to_check = [
                "flux_total_prior_country",
                "flux_total_posterior_country",
                "percentile_flux_total_prior_country",
                "percentile_flux_total_posterior_country",
            ]

            for var in vars_to_check:
                if var not in ds:
                    continue
                if (
                    "units" not in ds[var].attrs.keys()
                    and "unit" in ds[var].attrs.keys()
                ):
                    ds[var].attrs["units"] = ds[var].attrs["unit"]
                    ds[var].attrs.pop("unit")

            if "countrynumber" in ds:
                ds = ds.rename({"countrynumber": "country"})

            if "BEL-LUX" in ds.country and (
                "BEL" not in ds.country and "LUX" not in ds.country
            ):
                logger.info(
                    f" InTEM does not estimate separate BELGIUM emissions.\n A population ratio of {config.bel_pop_r} is being used to scale InTEM's total BELGIUM+LUXEMBOURG estimate."
                )

                r = config.bel_pop_r

                variables_with_country = [
                    var for var in ds.data_vars if "country" in ds[var].dims
                ]
                numerical_vars = [
                    var
                    for var in variables_with_country
                    if np.issubdtype(ds[var].dtype, np.number)
                    and var != "country_fraction"
                ]

                ds_bel = r * ds[numerical_vars].sel(country="BEL-LUX")
                ds_lux = (1 - r) * ds[numerical_vars].sel(country="BEL-LUX")

                del ds_bel["country"]
                del ds_lux["country"]

                ds_bel["country_merge"] = xr.DataArray(
                    data=[
                        "BELGIUM",
                    ]
                    * ds_bel.time.size,
                    dims=[
                        "time",
                    ],
                    coords={"time": ds_bel.time},
                    attrs=ds["country"].attrs,
                )

                ds_lux["country_merge"] = xr.DataArray(
                    data=[
                        "LUXEMBOURG",
                    ]
                    * ds_lux.time.size,
                    dims=[
                        "time",
                    ],
                    coords={"time": ds_lux.time},
                    attrs=ds["country"].attrs,
                )

                ds_bellux = xr.concat(
                    [ds_bel, ds_lux], pd.Index(["BEL", "LUX"], name="country")
                )

                ds = xr.merge([ds, ds_bellux], join="outer", compat="no_conflicts")
                ds = ds.drop_vars("country_merge")

        elif m0 == "rhime":
            ds["country"] = [
                regions_info["country_codes"].get(x, x) for x in ds["country"].values
            ]

        elif m0 == "cif-enks":
            # Move time variable to center of the month
            ds["time"] = ds.time.values + np.timedelta64(15, "D")

            # Add "_" to second country dimension in covariance matrix
            ds = ds.rename({"country2": "country_2"})

        # Rename second country dimension in covariance matrix (xarray requirement)
        var_to_change = "covariance_flux_total_posterior_country"
        if var_to_change in ds and ds[var_to_change].dims == (
            "time",
            "country",
            "country",
        ):
            ds[var_to_change] = xr.DataArray(
                data=ds[var_to_change].data,
                dims=["time", "country", "country_2"],
                coords=dict(
                    time=(["time"], ds[var_to_change].time.data),
                    country=(["country"], ds[var_to_change].country.data),
                    country_2=(["country_2"], ds[var_to_change].country.data),
                ),
                attrs=ds[var_to_change].attrs,
            )

    # Fix concentration/eddy flux datasets
    elif file_type in (DataTypes.CONCENTRATION, DataTypes.EDDY_FLUX):
        # Ensure integer dtype
        ds["number_of_identifier"] = ds["number_of_identifier"].astype(int)

        # Ensure string dtype
        ds["platform"] = ds["platform"].astype(str)

        # Fix old format vs new format
        if "index" not in ds.dims:
            platforms = ds["platform"].values
            ds = (
                # Remove the old platform dimension and replace with a new coordinate
                ds.drop_vars("platform")
                .assign_coords({"platform": ("platform", platforms)})
                # Restack the dataset to have a single index dimension
                .stack({"index": ["number_of_identifier", "time"]})
                .reset_index("index")
            )

        if "assimilation_flag" not in ds:
            # Add assimilation_flag if not present
            ds = ds.assign(
                assimilation_flag=("index", np.ones(ds["index"].size, dtype=int))
            )

        # Test that the number of identifiers had valid values
        max_num_id, min_num_id = (
            ds["number_of_identifier"].max(),
            ds["number_of_identifier"].min(),
        )
        if min_num_id == 1 and max_num_id == len(ds["platform"]):
            # 1 based (also called as retarded) indexing, so we need to shift the values
            ds["number_of_identifier"] -= 1
        elif min_num_id < 0:
            raise ValueError(
                "The number of identifiers should be positive. Please check the input data."
            )
        elif max_num_id >= len(ds["platform"]):
            raise ValueError(
                f"The max number of identifiers should be less than the number of platform. "
                "Please check the input data."
            )

        # Set coordinates
        ds = ds.assign_coords(
            {var: ds[var] for var in ["number_of_identifier", "time", "platform"]}
        )

        # Fix for InTEM (units of platform are wrongly set to mol mol-1)
        ds["platform"].attrs.pop("units", None)

        # Apply model specific corrections
        if m0 == "cif-enks":
            # Convert platform names to upper case and drop "_C" for continuos data
            platform_in_caps = [platform.upper() for platform in ds["platform"].values]
            for i, platform in enumerate(platform_in_caps):
                platform_id, dtype = platform.split("_")
                if dtype == "C":
                    platform_in_caps[i] = platform_id

            ds["platform"] = xr.DataArray(
                data=platform_in_caps,
                dims=ds["platform"].dims,
                coords=ds["platform"].coords,
            )
        if m0 == "flexinvert":
            ds["mf_observed"].attrs["units"] = "ppt"
            ds["mf_observed"].attrs["longname"] = "observed_mole_fraction"
            ds["mf_prior"].attrs["units"] = "ppt"
            ds["mf_prior"].attrs["longname"] = "apriori_simulated_mole_fraction"
            ds["mf_posterior"].attrs["units"] = "ppt"
            ds["mf_posterior"].attrs["longname"] = "aposteriori_simulated_mole_fraction"
            ds["mf_bc_prior"] = ds["Ypri_bkg"]
            ds["mf_bc_prior"].attrs["units"] = "ppt"
            ds["mf_bc_prior"].attrs[
                "longname"
            ] = "apriori_simulated_boundary_condition_mole_fraction"
            ds["mf_bc_prior"] = ds["Ypri_bkg"]
            ds["mf_bc_posterior"] = ds["Ypost_bkg"]
            ds["mf_bc_posterior"].attrs["units"] = "ppt"
            ds["mf_bc_posterior"].attrs[
                "longname"
            ] = "aposteriori_simulated_boundary_condition_mole_fraction"

            # Fill intake_height and stdev_mf_model with fake values
            ds["intake_height"][:] = 0
            ds["stdev_mf_model"][:] = 0
        if m0 == "intem":
            # Set assimilation_flag to zero is mf_observed = NaN
            mask = np.isnan(ds["mf_observed"])
            if any(mask):
                ds["assimilation_flag"][mask] = 0
                logger.info(
                    f"Masking out nan values in {model}, as a quick fix for a bug in InTEM concentration files."
                )

    # Fix eddy flux dataset
    if file_type == DataTypes.EDDY_FLUX:
        # check some eddy flux variables
        if "ecflux_observed" not in ds:
            var1, var2 = "ecflux_measured", "ecflux_observed_storage"
            if var1 in ds and var2 in ds:
                if ds[var1].units != ds[var2].units:
                    raise ValueError(
                        f"Units of {var1} and {var2} do not match: "
                        f"{ds[var1].units} != {ds[var2].units}"
                    )
                # Calculate the observed flux
                ds["ecflux_observed"] = ds[var1] + ds[var2]
                ds["ecflux_observed"].attrs["units"] = ds[var1].attrs.get(
                    "units", "umol m-2 s-1"
                )

        if "ecflux_prior" not in ds:
            # Calculate it from sectorial prior
            ds["ecflux_prior"] = ds["ecflux_sectorial_prior"].sum(dim="sector")
            ds["ecflux_prior"].attrs["units"] = ds["ecflux_sectorial_prior"].attrs.get(
                "units", "umol m-2 s-1"
            )
    return ds


def add_sites_var(
    ds_flux: xr.Dataset,
    filepath_flux: Path,
    model: str,
    frequency: str,
    config_data: dict[str, dict],
) -> xr.Dataset:
    """
    Add a 'sites' variable to a flux dataset indicating which sites are used for each flux timestamp,
    using the corresponding concentration dataset.

    Args:
        ds_flux (xarray dataset):
            xarray dataset with model data.
        filepath_flux (Path):
            Path to the flux NetCDF file.
        model (str):
            Model name tag corresponding to ds_flux.
        frequency (str):
            Frequency of the inversion results present in the dataset.
        config_data (dict of str):
            Dictionary with settings read from json file.

    Returns:
        ds_flux (xarray dataset):
            xarray dataset with a new 'sites' variable added if the concentration
            file exists. If not, returns the original ds_flux unchanged.
    """

    # Derive the path for the concentration file
    if re.match(r".*_flux.nc$", str(filepath_flux)):
        filepath_conc = filepath_flux.with_stem(
            filepath_flux.stem.replace("flux", "concentrations")
        )
    else:
        filepath_conc = filepath_flux.with_name(
            filepath_flux.stem + "_concentrations.nc"
        )

    # Check if file exists
    if not filepath_conc.is_file():
        logger.warning(
            f"Cannot find {filepath_conc} to add sites to {model} flux dataset."
        )
        return ds_flux

    # Open the concentration dataset
    ds_conc = xr.open_dataset(filepath_conc)

    # Fix variables and attributes
    ds_conc = edit_vars_and_attributes(
        ds_conc,
        model,
        frequency,
        "concentration",
        config_data.get("regions_info", {}),
        config_data.get("site_info", {}),
    )

    # Get list of observation platforms (sites) and flux time points
    sites_list = ds_conc.platform.values.tolist()
    flux_times = ds_flux.time

    # Initialize empty binary array [time, site] to store presence (1) or absence (0) of data
    sites = xr.DataArray(
        data=np.zeros((len(flux_times), len(sites_list)), dtype=int),
        coords={"time": flux_times, "platform": sites_list},
        dims=["time", "platform"],
        attrs={
            "units": "1",
            "long_name": "Site availability (1 if observations present during this period)",
        },
    )

    if frequency == "yearly":
        flux_keys = flux_times.dt.year.values
    elif frequency == "monthly":
        flux_keys = list(zip(flux_times.dt.year.values, flux_times.dt.month.values))
        flux_keys = np.array(flux_keys, dtype=[("year", "i4"), ("month", "i4")])

    for site in sites_list:
        site_index = get_site_index(ds_conc, site)
        mask = (ds_conc["number_of_identifier"] == site_index) & ds_conc[
            "mf_observed"
        ].notnull()
        # Note: drop=True leads to problems when the platform exists but there is absolutely no data
        valid_times = ds_conc["time"].where(mask, drop=False)

        if frequency == "yearly":
            mf_keys = valid_times.dt.year.values
        elif frequency == "monthly":
            years = valid_times.dt.year.values
            months = valid_times.dt.month.values
            # Mask NaN, otherwise conversion to int won't work
            valid_mask = ~np.isnan(years) & ~np.isnan(months)
            mf_keys = list(zip(years[valid_mask], months[valid_mask]))
            mf_keys = np.array(mf_keys, dtype=[("year", "i4"), ("month", "i4")])

        # Mark time steps in flux where observations from this site exist
        sites.loc[dict(platform=site)] = np.isin(flux_keys, mf_keys).astype(int)

    # Add the 'sites' variable to the flux dataset
    ds_flux["sites"] = sites

    logger.info(f"Adding sites from {filepath_conc} to {model} flux dataset.")

    return ds_flux
