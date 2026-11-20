import xarray as xr

from fluxie.io import read_config_files, read_model_output
from fluxie.types import DataTypes, DataType
from fluxie.test_utils import data_dir

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
    Read the hfc134a test data of the given models.

    Args:
        test_models:
            Names of the models to read.
        file_type:
            Type of the files to read, e.g. 'flux' or 'concentration'.

    Returns:
        ds_all_mf:
            Dictionary of the loaded datasets, with model names as keys.
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
