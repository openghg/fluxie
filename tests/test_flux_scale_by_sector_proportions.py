import pytest
from pathlib import Path
import fluxy
from fluxy.operators.flux_scale_by_sector_proportions import scale_by_sector_proportions
from fluxy.io import read_config_files, read_model_output
from fluxy.operators.select import slice_flux

data_dir = Path(fluxy.__path__[0]).parent / "data" / "tests"
config_data = read_config_files()


def test_scale_by_sector_proportions():

    species = "ch4"  # options for individual species, or 'all_hfc' or 'all_pfc'
    models = [
        "InTEM_NAME_EUROPE_EDGAR_old_format",
        "ELRIS_NAME_EUROPE_EDGAR_old_format",
        "RHIME_NAME_EUROPE_EDGAR_old_format",
    ]
    regions = ["GERMANY"]
    period = "monthly"  # use to override standard inversion periods, must be a list the same length as models, e.g. ['monthly','yearly']
    country_flux_units_print = "Gg yr-1"
    start_date = "2022-01-01"  # inclusive. Option to set as list of dates, e.g. ['2018-01-01','2019-01-01'] which is required for total fgases if one model is missing obs for a year
    end_date = "2024-01-01"  # not inclusive. Option to set as list of dates, e.g. ['2023-01-01','2022-01-01'] which is required for total fgases if one model is missing obs for a year
    get_labels_from_file = False
    sector_file = "EUROPE_EDGAR"
    create_region_sector_totals = True  # if True, uses country_fraction variable to sum spatial sector fluxes to region sector fluxes

    ds_all_flux_scaled = {}
    
    ds_all_flux = read_model_output(
        data_dir, "flux", species, models, config_data, period=period
    )

    for m in models:
        ds_all_flux_scaled[m] = slice_flux(
            {m: ds_all_flux[m]},
            config_data,
            start_date,
            end_date,
            species=species,
            country_flux_units_print=country_flux_units_print,
        )[m]

    ds_all_flux_scaled = scale_by_sector_proportions(
        data_dir=data_dir,
        ds_all=ds_all_flux_scaled,
        species=species,
        country_flux_units_print=country_flux_units_print,
        config_data=config_data,
        regions=regions,
        sector_file=sector_file,
        create_region_sector_totals=create_region_sector_totals,
        sectors=["agriculture"],
        cell_area_test_file=True
    )

    # variables flux_agriculture_prior, flux_agriculture_posterior,
    # flux_agriculture_country_prior and flux_agriculture_country_posterior should have been added

    test_variables = [
        "flux_agriculture_prior",
        "flux_agriculture_posterior",
        "flux_agriculture_prior_country",
        "flux_agriculture_posterior_country",
    ]

    var_present = False

    for m in models:
        if all(i in ds_all_flux_scaled[m] for i in test_variables):
            var_present = True
        else:
            var_present = False

    assert var_present == True
