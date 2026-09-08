import numpy as np
import pytest
import xarray as xr

from fluxie.operators.flux_prepare_inventory import retrieve_inventories
from fluxie.test_utils import data_dir

kwargs = {
    "data_dir": data_dir,
    "country": "UK",
    "species": "ch4",
    "start_date": "2020-01-01",
    "end_date": "2022-01-01",
    "unit": "Gg yr-1",
    "s_data": {"ch4": {"molar_mass": 16.04}},
    "r_data": {},
    "inventory_years": None,
    "inventory_filename": "EUROPE_EDGAR",
}


def test_retrieve_inventories_default_year():

    inventories, uncertainties = retrieve_inventories(**kwargs)

    assert len(inventories) == 1
    assert len(uncertainties) == 1
    assert inventories[0].sizes["time"] == 2


def test_retrieve_inventories_multiple_sectors():

    inventories, uncertainties = retrieve_inventories(
        sectors=["energy", "waste"],
        **kwargs,
    )

    assert inventories[0].sizes["sector"] == 2
    assert uncertainties[0].sizes["sector"] == 2


def test_retrive_non_existing_country():

    with pytest.raises(KeyError, match="Country not found in inventory: XX"):
        inventories, uncertainties = retrieve_inventories(**kwargs | {"country": "XX"})
