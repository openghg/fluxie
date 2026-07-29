import numpy as np
import pytest
import xarray as xr

from fluxie.operators.flux_align_dataset import align_lat_lon, align_time


def _make_grid_test_dataset(
    latitudes: np.ndarray, longitudes: np.ndarray
) -> xr.Dataset:
    data = np.arange(latitudes.size * longitudes.size).reshape(
        1, latitudes.size, longitudes.size
    )
    return xr.Dataset(
        {
            "flux": (
                ["time", "latitude", "longitude"],
                data,
            )
        },
        coords={
            "time": np.array([np.datetime64("2024-01-01")]),
            "latitude": latitudes,
            "longitude": longitudes,
        },
    )


def _make_time_test_dataset(time: np.ndarray) -> xr.Dataset:
    return xr.Dataset(
        {
            "flux": (
                ["time"],
                np.arange(time.size, dtype=float),
            )
        },
        coords={"time": time.astype("datetime64[ns]")},
        attrs={"inversion_system": "test", "species": "test"},
    )


ds_ref = _make_grid_test_dataset(
    latitudes=np.array([45.0, 46.0, 47.0]),
    longitudes=np.array([5.0, 6.0]),
)

ds_close = _make_grid_test_dataset(
    latitudes=np.array([45.005, 46.005, 47.005]),
    longitudes=np.array([5.0, 6.0]),
)

ds_short = _make_grid_test_dataset(
    latitudes=np.array([45.0, 46.0]),
    longitudes=np.array([5.0, 6.0]),
)


def test_align_lat_lon_single_dataset_returns_input():

    aligned = align_lat_lon([ds_ref], coord="latitude")

    assert len(aligned) == 1
    assert aligned[0].identical(ds_ref)


def test_align_lat_lon_aligns_close_coordinates():

    aligned = align_lat_lon([ds_ref, ds_close], coord="latitude", rel_tolerance=0.01)

    assert aligned[0]["latitude"].identical(ds_ref["latitude"])
    assert aligned[1]["latitude"].identical(ds_ref["latitude"])


def test_align_lat_lon_aligns_not_close_enough():

    with pytest.raises(ValueError, match="seem to be too different"):
        align_lat_lon([ds_ref, ds_close], coord="latitude", rel_tolerance=1e-6)


def test_align_lat_lon_raises_when_coordinates_cannot_be_aligned():

    ds_far = _make_grid_test_dataset(
        latitudes=np.array([45.5, 46.5, 47.5]),
        longitudes=np.array([5.0, 6.0]),
    )

    with pytest.raises(ValueError, match="dimensions seem to be too different"):
        align_lat_lon([ds_ref, ds_far], coord="latitude")


def test_align_lat_lon_with_different_lengths():

    align_lat_lon([ds_ref, ds_short], coord="latitude", min_rel_overlap=0.5)


def test_align_lat_lon_with_too_different_lengths():

    with pytest.raises(ValueError, match="cover too different ranges "):
        align_lat_lon([ds_ref, ds_short], coord="latitude", min_rel_overlap=0.8)


def test_align_time_allows_missing_whole_periods_when_overlapping():

    ds_missing_year = _make_time_test_dataset(
        np.array(["2018-01-01", "2019-01-01", "2021-01-01"])
    )
    ds_complete = _make_time_test_dataset(
        np.array(["2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01"])
    )

    aligned = align_time([ds_missing_year, ds_complete])

    expected_time = ds_missing_year.time.values
    assert aligned[0].time.identical(ds_missing_year.time)
    assert np.array_equal(aligned[1].time.values, expected_time)


def test_align_time_reindexes_missing_whole_periods_when_not_overlapping():

    ds_missing_year = _make_time_test_dataset(
        np.array(["2018-01-01", "2019-01-01", "2021-01-01"])
    )
    ds_complete = _make_time_test_dataset(
        np.array(["2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01"])
    )

    aligned = align_time([ds_missing_year, ds_complete], only_overlapping=False)

    assert aligned[0].time.size == 4
    assert aligned[1].time.size == 4
    assert aligned[0].flux.isnull().any()


def test_align_time_raises_for_non_periodic_gaps():

    ds_irregular = _make_time_test_dataset(
        np.array(["2018-01-01", "2019-06-01", "2021-01-01"])
    )
    ds_regular = _make_time_test_dataset(
        np.array(["2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01"])
    )

    with pytest.raises(ValueError, match="Unable to infer period"):
        align_time([ds_irregular, ds_regular])
