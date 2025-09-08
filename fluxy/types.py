from enum import Enum
from typing import Literal, NewType, Union
import xarray as xr


class DataTypes(Enum):
    """Enum for different data types used in fluxy."""

    # Flux data (emission maps)
    FLUX = "flux"
    # Concentration data (measurements, model outputs)
    CONCENTRATION = "concentration"
    # Flux measured by eddy covariance methods and eddy flux simulations
    EDDY_FLUX = "eddy_flux"


def file_pattern(file_type: DataTypes) -> str:
    """Returns the ending pattern for the given file type."""
    if file_type == DataTypes.FLUX:
        # Default file type
        return ".nc"
    elif file_type == DataTypes.CONCENTRATION:
        return "_concentrations.nc"
    else:
        return f"_{file_type.value}.nc"


DataType = DataTypes | Literal["flux", "concentration", "eddy_flux"]

VariableType = str | dict[str, str | None] | list[str]


DsAll = dict[str, xr.Dataset]
