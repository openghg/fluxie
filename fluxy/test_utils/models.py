import xarray as xr

from fluxy.io import read_config_files, read_model_output
from fluxy.types import DataTypes, DataType
from fluxy.test_utils import data_dir

test_models = [
    "InTEM_NAME_EUROPE_EDGAR_old_format",
    "ELRIS_NAME_EUROPE_EDGAR_old_format",
    "RHIME_NAME_EUROPE_EDGAR_old_format",
    "ELRIS_NAME_EUROPE_EDGAR",
]

test_models_with_inlet = ["InTEM_NAME_EUROPE_EDGAR"]


def get_loaded_models(
    test_models: list[str],
    file_type: DataType,
) -> dict[str, xr.Dataset]:
    """
    Returns a list of loaded models.
    """

    config_data = read_config_files()

    ds_all_mf = read_model_output(
        data_dir,
        file_type,
        species="hfc134a",
        models=test_models,
        config_data=config_data,
    )

    return ds_all_mf
