from pathlib import Path
import fluxy
from fluxy.io import read_config_files, read_model_output, read_flux_total_fgases
from fluxy.operators.select import slice_flux, slice_mf


data_dir = Path(fluxy.__path__[0]).parent / "data" / "tests"


def test_read_flux():
    # This test fails sometimes when runned wil all the other tests
    # Because of a xarray cache problem
    config_data = read_config_files()

    species = "hfc134a"  # options for individual species, or 'all_hfc' or 'all_pfc'
    models = [
        "InTEM_NAME_EUROPE_EDGAR_std",
        "ELRIS_NAME_EUROPE_EDGAR_std",
        "RHIME_NAME_EUROPE_EDGAR_std",
    ]
    period = "yearly"
    country_flux_units_print = "Gg yr-1"
    start_date = "2018-01-01"  # inclusive. Option to set as list of dates, e.g. ['2018-01-01','2019-01-01'] which is required for total fgases if one model is missing obs for a year
    end_date = "2024-01-01"  # not inclusive. Option to set as list of dates, e.g. ['2023-01-01','2022-01-01'] which is required for total fgases if one model is missing obs for a year

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


def test_read_mf():
    # This test fails sometimes when runned wil all the other tests
    # Because of a xarray cache problem
    config_data = read_config_files()

    species = "hfc134a"
    site = "MHD"
    models = ["InTEM_NAME_EDGAR_std", "ELRIS_NAME_EDGAR_std", "RHIME_NAME_EDGAR_std"]
    period = "yearly"  # use to override standard inversion periods, must be a list the same length as models, e.g. ['monthly','yearly']
    mf_units_print = "ppt"
    start_date = "2018-01-01"  # inclusive. Option to set as list of dates, e.g. ['2018-01-01','2019-01-01'] which is required for total fgases if one model is missing obs for a year
    end_date = "2024-01-01"  # not inclusive. Option to set as list of dates, e.g. ['2023-01-01','2022-01-01'] which is required for total fgases if one model is missing obs for a year
    baseline_site = (
        None  #'MHD', 'JFJ' or 'CMN'. If None, does not mask by baseline time
    )

    ds_all_mf = read_model_output(
        data_dir, "concentration", species, models, config_data, period=period
    )

    ds_all_mf_sliced = slice_mf(
        ds_all_mf.copy(),
        start_date,
        end_date,
        site,
        baseline_site=baseline_site,
        data_dir=data_dir,
        mf_units_print=mf_units_print,
    )
