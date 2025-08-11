import xarray as xr
import logging
import re
from collections import Counter
from typing import Literal
from fluxy import config

logger = logging.getLogger(__name__)


def scale_variables(
    model: str,
    ds_model: xr.Dataset,
    species_info: dict[str, str | float] = None,
    **print_units: dict[Literal["mf_unit", "country_flux_unit", "flux_unit"], str],
) -> xr.Dataset:
    """
    Scales mole fractions, country fluxes and fluxes according to print units.

    Args:
        model (str):
            Name tag of the model being scaled.
            i.e. '<inversionModel>_<optional_identifying_tags>', preceded by subdirectory if applicable
        ds_model (xarray dataset):
            Sliced dataset with mf data from model.
        species_info (dictionary of str):
            Dictionary with species-specific settings.
        print_units (dictionary of str):
            Option for keys are: 'mf_unit', 'country_flux_unit', 'flux_unit'
            Keys point to units to which mole fractions/country fluxes/fluxes should be converted to.
            If None, no scaling is applied.
    Returns:
        ds_model (xarray dataset):
            Sliced and scaled dataset.
    """

    # Dictionaries with scaling settings (variable and unit type)
    var_type_name = {
        "mf_unit": "mole fraction",
        "country_flux_unit": "country flux",
        "flux_unit": "flux",
    }

    unit_type = {
        "mf_unit": "mf",
        "country_flux_unit": "mass1 time-1",
        "flux_unit": "amount1 length-2 time-1",
    }
    # NOTE: country fluxes must be in mass units so that conversion to CO2-eq works

    optional_unit_type = {
        "flux_unit": "mass1 length-2 time-1",
    }

    # Create copy to prevent scanning for variables in scaled dataset
    # NOTE: Scaled dataset may contain CO2-eq, which is an acceptable target unit but not
    # an acceptable entry unit (see issue #36)
    ds_scaled = ds_model.copy()

    # Loop over variable types
    for scale_var, print_unit in print_units.items():
        if print_unit is not None:

            if var_type_name.get(scale_var) is None:
                raise ValueError(
                    f"{scale_var} is not implemented. Acceptable keys are: {var_type_name.keys()}."
                )

            # Get list of variables to scale
            var_names, var_unit = get_variables(
                ds_model, unit_type=unit_type[scale_var]
            )

            if (not var_names) and (opt_unit_type := optional_unit_type.get(scale_var)):
                var_names, var_unit = get_variables(ds_model, unit_type=opt_unit_type)

            if not var_names:
                raise ValueError(
                    f"There are no variables in {model} with {var_type_name[scale_var]} units in the attributes. Scaling to {print_unit} cannot be applied."
                )

            if var_unit is None:
                raise ValueError(
                    f"{model} dataset considers different {var_type_name[scale_var]} units. Uniform scaling to {print_unit} cannot be applied."
                )

            # Get scaling factor
            gwp = 1
            target_unit = print_unit
            if (scale_var == "country_flux_unit") and ("CO2-eq" in target_unit):
                gwp = species_info["gwp"]
                target_unit = target_unit.replace("CO2-eq", "")
                logger.info(f"Converting to mass of CO2-eq using GWP = {gwp}.")

            molar_mass = None
            if species_info is not None:
                molar_mass = species_info["molar_mass"]

            scaling_factor = get_units_conversion_factor(
                var_unit, target_unit, molar_mass
            )

            # Apply scaling
            for v in var_names:
                ds_scaled[v] = ds_model[v] * scaling_factor * gwp
                ds_scaled[v].attrs["units"] = print_unit

            logger.info(
                f"Scaling {model} {var_type_name[scale_var]} by {scaling_factor*gwp}."
            )

            # Apply scaling to covariance matrix
            # NOTE: covariance units are assumed consistent with country flux units (see issue #35)
            cov_var = "covariance_flux_total_posterior_country"
            if (scale_var == "country_flux_unit") and (cov_var in ds_model.keys()):
                cov_scaling_factor = scaling_factor**2 * gwp**2
                ds_scaled[cov_var] = ds_model[cov_var] * cov_scaling_factor
                logger.info(f"Scaling covariance in {model} by {cov_scaling_factor}")

    return ds_scaled


def get_variables(ds_model: xr.Dataset, unit_type: str) -> tuple[list[str], str | None]:
    """
    Finds variables of a given type in a dataset.

    Args:
        ds_model (xarray dataset):
            Dataset with data from a model.
        unit_type (str):
            Unit type of interest (e.g. "mf", "mass1 length-2 time-1")
    Returns:
        var_names (list of str):
            Names of variables in the dataset with unit_type units.
        unique_units (str | None):
            unit_type explicit units used in the dataset (e.g. "mol mol-1", "kg m-2 s-1")
            Set to None if different units are present.
    """

    var_names = []
    var_units = []

    # Find variables of type var_type in dataset
    for var, var_data in ds_model.items():
        if "units" in var_data.attrs.keys():
            unit = var_data.attrs["units"]

            # Particular case of mole fractions:
            if unit_type == "mf":
                if unit in config.units_scale["mf"].keys():
                    var_names.append(var)
                    var_units.append(unit)

            else:
                # Get unit type
                x, unit_dims = get_unit_type_and_conversion_to_base(unit)

                # Append var if unit is of the desiared var_type
                if Counter(unit_dims) == Counter(unit_type.split()):
                    var_names.append(var)
                    var_units.append(unit)

    # Get var units if unique
    unique_units = list(set(var_units))
    if len(unique_units) == 1:
        unique_units = unique_units[0]
    else:
        unique_units = None

    return var_names, unique_units


def get_unit_type_and_conversion_to_base(input_unit: str) -> tuple[float, list[str]]:
    """
    Computes conversion factor to base unit.
    Units are expected to have the following format:
        "<letters><(-)integer>" separated by spaces (e.g. "kg m-2 s-1")

    Args:
        input_unit (str):
            Units of a given variable.
    Returns:
        conversion_factor (float):
            Scaling factor to base unit.
        unit_dim_type (list of str):
            Unit type of input_unit (e.g. ["mass1", "length-2", "time-1"])
    """

    # Get individual units
    units_list = input_unit.split()

    # Initialize list of dimensions and exponents
    unit_dim = []
    unit_exponent = [1] * len(units_list)

    # Get unit dimension and exponent
    for i, unit_exp in enumerate(units_list):
        unit_elements = list(re.findall(r"[a-zA-Z]+|[-+]?\d+", unit_exp))
        unit_dim.append(unit_elements[0])
        if len(unit_elements) > 1:
            unit_exponent[i] = float(unit_elements[1])

    conversion_factor = 1
    unit_dim_type = []

    # Get unit family and conversion factor to base units
    for i, unit in enumerate(unit_dim):
        factor_to_base = None
        for unit_family, units in config.units_scale.items():
            if unit in units:
                factor_to_base = units[unit] ** unit_exponent[i]
                unit_dim_type.append(unit_family + f"{unit_exponent[i]:.0f}")
                break

        if factor_to_base is None:
            raise KeyError(f"Unit {unit} does not exist in units_scale dictionary.")

        conversion_factor = conversion_factor * factor_to_base

    return conversion_factor, unit_dim_type


def get_units_conversion_factor(
    from_unit: str, to_unit: str, molar_mass: float | None = None
) -> float:
    """
    Computes conversion factor between two units.
    Units are expected to have the following format:
        "<letters><(-)integer>" separated by spaces (e.g. "kg m-2 s-1")

    Consistency between base and target units are verified.
    Non simplified units (e.g. "kg m-1 m-1 s-1") are assumed different from their simplified form
    and should be avoided.

    Conversion from mass units (g) to amount of substance (mol) is done via molar_mass.
    The current implementation only applies the conversion if the units exponent is equal to 1.

    Args:
        from_unit (str):
            Units of the variable to be converted.
        to_unit (str):
            Target units.
        molar_mass (float):
            Molar mass (g mol-1) to be used in g<->mol conversion.
    Returns:
        conversion_factor (float):
            Scaling factor that guarantees the requested units conversion.
    """

    # Deal with particular case of mol mol-1
    if from_unit == "mol mol-1" or to_unit == "mol mol-1":

        unit_to_base = config.units_scale["mf"].get(from_unit, None)
        if unit_to_base is None:
            raise KeyError(
                f"Conversion factor to/from {from_unit} does not exist in units_scale dictionary."
            )

        target_to_base = config.units_scale["mf"].get(to_unit, None)
        if target_to_base is None:
            raise KeyError(
                f"Conversion factor to/from {to_unit} does not exist in units_scale dictionary."
            )

        return unit_to_base / target_to_base

    # General case
    unit_to_base, unit_dim_type = get_unit_type_and_conversion_to_base(from_unit)
    target_to_base, target_dim_type = get_unit_type_and_conversion_to_base(to_unit)

    # Get g-mol conversion factor if needed
    M_scaling = 1
    if molar_mass is not None:
        non_common_units = set(unit_dim_type) ^ set(target_dim_type)

        if non_common_units == {"mass1", "amount1"}:
            # NOTE: conversion is only applied if exponent == 1
            if "amount1" in unit_dim_type:
                M_scaling = molar_mass  # g mol-1
                # Update unit type so that it passes consistency check
                index = unit_dim_type.index("amount1")
                unit_dim_type[index] = "mass1"
            else:
                M_scaling = 1 / molar_mass  # mol g-1
                # Update unit type so that it passes consistency check
                index = unit_dim_type.index("mass1")
                unit_dim_type[index] = "amount1"
            # NOTE: only one unit element of type amount/mass is expected and replaced

            logger.info(f"Converting g<->mol using M = {molar_mass} g mol-1")

    # Check for consistency
    if Counter(unit_dim_type) != Counter(target_dim_type):
        raise ValueError(
            f"Units {from_unit} ({unit_dim_type}) and {to_unit} ({target_dim_type}) are not consistent."
        )

    return unit_to_base / target_to_base * M_scaling
