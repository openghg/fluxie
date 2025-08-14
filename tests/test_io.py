from pathlib import Path
import fluxy
from fluxy.io import read_config_files, read_model_output, read_flux_total_fgases
from fluxy.operators.select import slice_flux, slice_mf
from fluxy.test_utils import data_dir
from fluxy.test_utils.models import test_models
import pytest


@pytest.mark.parametrize("model", test_models)
@pytest.mark.parametrize("add_sites_to_flux", [True, False])
def test_read_flux(model, add_sites_to_flux):
    # This test fails sometimes when runned with all the other tests
    # Because of a xarray cache problem
    config_data = read_config_files()

    species = "hfc134a"  # options for individual species, or 'all_hfc' or 'all_pfc'

    period = "yearly"

    ds_all_flux = read_model_output(
        data_dir, "flux", species, [model], config_data, period=period, add_sites_to_flux=add_sites_to_flux
    )


@pytest.mark.parametrize("model", test_models)
def test_read_mf(model):
    # This test fails sometimes when runned wil all the other tests
    # Because of a xarray cache problem
    config_data = read_config_files()

    species = "hfc134a"
    period = "yearly"  # use to override standard inversion periods, must be a list the same length as models, e.g. ['monthly','yearly']

    ds_all_mf = read_model_output(
        data_dir, "concentration", species, [model], config_data, period=period
    )


def test_read_config_default():
    config_data = read_config_files()
    assert isinstance(config_data, dict), "Config data should be a dictionary"
    assert len(config_data) > 0, "Config data should not be empty"


def test_read_empty_configs():
    fake_dir = Path("/nowhere/is/this")
    config_data = read_config_files(fake_dir)
    assert (
        config_data == {}
    ), "Config data should be an empty dictionary when no config files are found"
