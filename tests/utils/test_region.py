import xarray as xr

from fluxy.plots.utils import get_map_bounds


def test_get_map_bounds_tuple():
    """Test the get_map_bounds function with a tuple."""

    bounds = (0, 1, 2, 3)
    map_bounds = get_map_bounds(region=bounds)

    assert isinstance(map_bounds, tuple)
    assert map_bounds == (0, 1, 2, 3)


def test_get_map_bounds_list():
    """Test the get_map_bounds function with a list."""

    bounds = [0, 1, 2, 3]
    map_bounds = get_map_bounds(region=bounds)

    assert isinstance(map_bounds, tuple)
    assert map_bounds == (0, 1, 2, 3)


def test_get_map_bounds_known_region():
    """Test the get_map_bounds function with a known region."""

    bounds = "IRELAND"
    map_bounds = get_map_bounds(
        region=bounds,
        config_data={
            "regions_info": {
                "country_codes": {
                    "IRELAND": "IRL",
                    "UK": "GBR",
                    "FRANCE": "FRA",
                }
            },
        },
    )

    assert isinstance(map_bounds, tuple)
    assert len(map_bounds) == 4


def test_get_map_bounds_from_country_fraction():
    """Test the get_map_bounds function different list of regions."""

    # Create a minimum datasets for testing
    import numpy as np

    dims = xr.Dataset(
        {
            "latitude": (["latitude"], [0, 1, 2, 3, 4]),
            "longitude": (["longitude"], [0, 1, 2, 3, 4]),
            "country": (["country"], ["FRA", "GBR", "NLD"]),
        }
    )
    gbr = xr.Dataset(
        {
            "latitude": (["latitude"], [0, 1]),
            "longitude": (["longitude"], [0, 1]),
            "country": (["country"], ["GBR"]),
            "country_fraction": (
                ["country", "longitude", "latitude"],
                np.ones((1, 2, 2)),
            ),
        }
    )
    fra = xr.Dataset(
        {
            "latitude": (["latitude"], [2, 3]),
            "longitude": (["longitude"], [1, 2]),
            "country": (["country"], ["FRA"]),
            "country_fraction": (
                ["country", "longitude", "latitude"],
                np.ones((1, 2, 2)),
            ),
        }
    )
    ndl = xr.Dataset(
        {
            "latitude": (["latitude"], [1, 2]),
            "longitude": (["longitude"], [2, 3]),
            "country": (["country"], ["NDL"]),
            "country_fraction": (
                ["country", "longitude", "latitude"],
                np.ones((1, 2, 2)),
            ),
        }
    )

    dss = [
        xr.merge([dims, gbr, fra, ndl]),
    ] * 2

    # Regions to test and their answer
    regions_lims = {
        "FRA": (1, 2, 2, 3),
        "FRA-UK": (0, 2, 0, 3),
        "FRA-GBR": (0, 2, 0, 3),
        "NW_EU": (0, 3, 0, 3),
        "NW_EU-FRA": (0, 3, 0, 3),
    }

    # Config_data
    config_data = {
        "regions_info": {
            "country_codes": {
                "UK": "GBR",
            },
            "regions": {
                "NW_EU": "FRA-GBR-NDL",
            },
        }
    }

    for region, res in regions_lims.items():
        print(region)
        bounds = get_map_bounds(
            ds_all=dss, region=region, config_data=config_data, zoom_degree=0
        )
        assert isinstance(bounds, tuple)
        assert bounds == res


def test_get_map_from_datasets():
    """Test the get_map_bounds function with datasets."""

    # Create a minimum datasets for testing
    dss = [
        xr.Dataset(
            {
                "latitude": (["latitude"], [0, 1]),
                "longitude": (["longitude"], [0, 1]),
            }
        ),
        xr.Dataset(
            {
                "latitude": (["latitude"], [2, 3]),
                "longitude": (["longitude"], [2, 3]),
            }
        ),
    ]

    bounds = get_map_bounds(ds_all=dss)
    assert isinstance(bounds, tuple)
    assert bounds == (0, 3, 0, 3)
