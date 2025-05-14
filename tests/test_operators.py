import itertools

import pytest

from fluxy.operators.mf import compute_mf_difference, stats_mf
from fluxy.operators.select import slice_flux, slice_mf
from fluxy.test_utils.models import get_loaded_models, test_models

ds_all_mf = get_loaded_models("concentration")
ds_all_flux = get_loaded_models("flux")


# Test the difference between all available models
@pytest.mark.parametrize("model_pair", list(itertools.combinations(test_models, 2)))
def test_mf_difference(model_pair):

    ds_diff = compute_mf_difference(ds_all_mf, model_pair)


@pytest.mark.parametrize("model", test_models)
def test_slice_flux(model):

    country_flux_units_print = "Gg yr-1"
    start_date = "2018-01-01"
    end_date = "2024-01-01"

    dss_flux_scaled = slice_flux(
        {model: ds_all_flux[model]},
        start_date=start_date,
        end_date=end_date,
        country_flux_units_print=country_flux_units_print,
    )
    assert model in dss_flux_scaled


@pytest.mark.parametrize("model", test_models)
def test_slice_mf(model):

    start_date = "2018-01-01"
    end_date = "2024-01-01"

    dss_mf_scaled = slice_mf(
        {model: ds_all_mf[model]},
        start_date,
        end_date,
    )
    assert model in dss_mf_scaled


@pytest.mark.parametrize(
    "stat", ["prior", "posterior", "prior_above_BC", "posterior_above_BC"]
)
def test_stats_mf(stat):

    df = stats_mf(
        ds_all_mf,
        stats_type=stat,
    )
