from fluxy.io import read_config_files, read_model_output
from fluxy.types import DataTypes
from fluxy.test_utils import data_dir

import pytest

test_models = ["EDDY_HARDAU", "EDDY_HARDAU_STORAGE_2LAYERS"]


@pytest.mark.parametrize("model", test_models)
def test_read_ecflux(model):
    ds = read_model_output(
        data_dir=data_dir / "ecflux",
        file_type=DataTypes.EDDY_FLUX,
        species="co2",
        models=[model],
    )

    assert model in ds, f"Model {model} not found in the loaded datasets"


def test_sectorial_config():
    config_dict = read_config_files()

    assert "sectors" in config_dict, "Sectors should be defined in the config"
