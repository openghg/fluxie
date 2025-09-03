from datetime import timedelta
import itertools

import pytest

import xarray as xr
import pandas as pd
import numpy as np
from fluxy.operators.mf import compute_mf_difference, stats_mf
from fluxy.operators.select import (
    slice_flux,
    slice_mf,
    clean_timeseries_missing_data,
    slice_height,
    slice_site,
)
from fluxy.test_utils.models import (
    get_loaded_models,
    test_models,
    test_models_with_inlet,
)

ds_all_mf = get_loaded_models(test_models,"concentration")
ds_all_flux = get_loaded_models(test_models,"flux")

ds_all_mf_with_inlet = get_loaded_models(test_models_with_inlet,"concentration")


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


@pytest.mark.parametrize("model", test_models_with_inlet)
def test_slice_height(model):

    ds_all_mf__with_inlet_sliced = slice_site(ds_all_mf_with_inlet[model], site="TAC")

    ds_sliced = slice_height(ds_all_mf__with_inlet_sliced, intake_height=185)
    
    print(ds_sliced)

    assert np.unique(ds_sliced["intake_height"].values == 185.0)

ds_test = xr.Dataset(
    {
        "var1": (["index"], [1, 2, None, 4, 5]),
        "var2": (["index"], [None, 2, None, None, 5]),
    },
    coords={
        "time": ("index", pd.date_range("2020-01-01", periods=5, freq="D")),
    },
)


def test_clean_timeseries_missing_data():

    ds_out = clean_timeseries_missing_data(ds_test)
    # Missing days should have been replaced
    assert len(ds_out.time) == 5


def test_clean_timeseries_missing_data_with_freq_pd():

    ds_out = clean_timeseries_missing_data(ds_test, min_freq="1D")
    # Missing days should have been replaced
    assert len(ds_out.time) == 5


def test_clean_timeseries_missing_data_with_freq_timedelta():

    ds_out = clean_timeseries_missing_data(ds_test, min_freq=timedelta(days=1))
    # Missing days should have been replaced
    assert len(ds_out.time) == 5


def test_clean_timeseries_missing_data_removed():

    ds_out = clean_timeseries_missing_data(
        ds_test, min_freq="10D", variables_nans=["var2"]
    )
    # Missing days should have been removed
    assert len(ds_out.time) == 2
