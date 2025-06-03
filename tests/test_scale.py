from pathlib import Path
import pytest
import fluxy
from fluxy.io import read_config_files
from fluxy.io import read_model_output
from fluxy.operators.convert import scale_variables

data_dir = Path(fluxy.__path__[0]).parent / "data" / "tests"

species = "hfc134a"
period = "yearly"

# NOTE: if you change the models list, update the decorator
models = [
    "InTEM_NAME_EUROPE_EDGAR_old_format",
    "ELRIS_NAME_EUROPE_EDGAR_old_format",
    "RHIME_NAME_EUROPE_EDGAR_old_format",
]

# NOTE: if you change the target units, update the HARD-CODED scaling
mf_units_print = "ppt"
country_flux_units_print = "Gg CO2-eq yr-1"
flux_units_print = "kg km-2 yr-1"

config_data = read_config_files()

ds_all_flux = read_model_output(
    data_dir, "flux", species, models, config_data, period=period
)
ds_all_mf = read_model_output(
    data_dir, "concentration", species, models, config_data, period=period
)


@pytest.mark.parametrize(
    "m, original_country_flux_unit, original_flux_unit",
    [
        ("InTEM_NAME_EUROPE_EDGAR_old_format", "kg a-1", "mol m-2 s-1"),
        ("ELRIS_NAME_EUROPE_EDGAR_old_format", "kg yr-1", "mol m-2 s-1"),
        ("RHIME_NAME_EUROPE_EDGAR_old_format", "kg a-1", "mol m-2 s-1"),
    ],
)
def test_scale_flux(m, original_country_flux_unit, original_flux_unit):
    # Define test variables and indexes
    test_country_flux_var = "flux_total_posterior_country"
    itime_country_flux = 0
    icountry = 0
    test_flux_var = "flux_total_posterior"
    itime_flux = 0
    ilat = 0
    ilon = 0

    # Save old values
    ds_country_flux = ds_all_flux[m][test_country_flux_var].isel(
        time=itime_country_flux, country=icountry
    )
    ds_flux = ds_all_flux[m][test_flux_var].isel(
        time=itime_flux, latitude=ilat, longitude=ilon
    )

    if ds_country_flux.values == 0:
        raise ValueError("Please select an index with non-zero country fluxes.")

    if ds_flux.values == 0:
        raise ValueError("Please select an index with non-zero fluxes.")

    # Apply scaling
    dss_scaled = {}
    dss_scaled[m] = scale_variables(
        m,
        ds_all_flux[m],
        config_data["species_info"][species],
        country_flux_unit=country_flux_units_print,
        flux_unit=flux_units_print,
    )

    # Save scaled values
    ds_scaled_country_flux = dss_scaled[m][test_country_flux_var].isel(
        time=itime_country_flux, country=icountry
    )
    ds_scaled_flux = dss_scaled[m][test_flux_var].isel(
        time=itime_flux, latitude=ilat, longitude=ilon
    )

    # Check conversion
    assert ds_scaled_country_flux.units == country_flux_units_print
    assert (
        ds_country_flux.units == original_country_flux_unit
    ), "Units scaling cannot be verified. Please update the units according to the test datasets."

    # HARD-CODED scaling: adjust manually according to original_country_flux_unit and country_flux_units_print
    scaling_factor = 1e-6 * config_data["species_info"][species]["gwp"]
    assert (
        1
        - abs(ds_scaled_country_flux.values / (ds_country_flux.values * scaling_factor))
    ) < 1e-6

    assert ds_scaled_flux.units == flux_units_print
    assert (
        ds_flux.units == original_flux_unit
    ), "Units scaling cannot be verified. Please update the units according to the test datasets."

    # HARD-CODED scaling: adjust manually according to original_flux_unit and flux_units_print
    scaling_factor = (
        config_data["species_info"][species]["molar_mass"]  # mol -> g
        * 1e-3  # g -> kg
        * 1e6  # m-2 -> km-2
        * 60
        * 60
        * 24
        * 365  # s-1 -> yr-1
    )
    assert (1 - abs(ds_scaled_flux.values / (ds_flux.values * scaling_factor))) < 1e-6


@pytest.mark.parametrize(
    "m, original_mf_unit",
    [
        ("InTEM_NAME_EUROPE_EDGAR_old_format", "mol mol-1"),
        ("ELRIS_NAME_EUROPE_EDGAR_old_format", "mol mol-1"),
        ("RHIME_NAME_EUROPE_EDGAR_old_format", "mol mol-1"),
    ],
)
def test_scale_mf(m, original_mf_unit):
    # Define test variable and indexes
    test_var = "mf_posterior"
    index = 0

    # Save old value
    ds = ds_all_mf[m][test_var].isel(index=index)

    if ds.values == 0:
        raise ValueError("Please select an index with non-zero mole fractions.")

    # Apply scaling
    dss_scaled = {}
    dss_scaled[m] = scale_variables(
        m,
        ds_all_mf[m],
        mf_unit=mf_units_print,
    )

    # Save scaled value
    ds_scaled = dss_scaled[m][test_var].isel(index=index)

    # Check conversion
    assert ds_scaled.units == mf_units_print
    assert (
        ds.units == original_mf_unit
    ), "Units scaling cannot be verified. Please update the units according to the test datasets."

    # HARD-CODED scaling: adjust manually according to original_mf_unit and mf_units_print
    scaling_factor = 1e12
    assert (1 - abs(ds_scaled.values / (ds.values * scaling_factor))) < 1e-6
