from fluxie.operators.regions import format_plot_regions
import xarray as xr

import pytest


def test_format_default():
    """Test the format_plot_regions function with default settings."""

    regions = ["FRANCE", "GERMANY", "ITALY"]
    formatted_regions = format_plot_regions(plot_regions=regions)

    assert isinstance(formatted_regions, list)
    assert formatted_regions == ["FRANCE", "GERMANY", "ITALY"]


def test_no_region():
    """Test that format_plot_regions raises ValueError when called without plot_regions or ds_all."""
    with pytest.raises(
        ValueError, match="ds_all must be provided if plot_regions is None."
    ):
        format_plot_regions()


def test_format_from_ds_all():
    ds_mock = {
        "model1": xr.Dataset(
            {
                "country": (["country"], ["FRA", "DEU", "ITA"]),
            }
        ),
        "model2": xr.Dataset(
            {
                "country": (["country"], ["FRA", "DEU", "ESP"]),
            }
        ),
    }

    formatted_regions = format_plot_regions(ds_all=ds_mock)

    assert isinstance(formatted_regions, list)
    for c in ["FRA", "DEU"]:
        assert c in formatted_regions
    for c in ["ITA", "ESP"]:
        assert c not in formatted_regions
