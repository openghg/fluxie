import pytest
from fluxy.test_utils.models import test_models, get_loaded_models
from fluxy.operators.flux_map_resample import (
    get_flux_mean,
    average_over_period,
)

dss = get_loaded_models("flux")


@pytest.mark.parametrize("model", test_models)
def test_get_flux_mean(model):
    da_mean = get_flux_mean(dss[model])


@pytest.mark.parametrize("model", test_models)
@pytest.mark.parametrize(
    "period",
    [
        "year",
        "month",
        "season",
        # TODO: add test with the list input
    ],
)
def test_average_over_period(model, period):
    # Test the function with a sample dataset
    ds_seasonal = average_over_period(dss[model], chop_by=period)
