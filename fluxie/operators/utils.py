from functools import wraps
from typing import Callable

import xarray as xr

from fluxie.types import DsAll


def apply_to_dict_or_single(
    func: Callable[..., xr.Dataset],
) -> Callable[..., xr.Dataset | DsAll]:
    """
    Decorate a function to apply it to a single dataset or a dictionary of datasets.

    If the first argument is a dictionary of datasets (DsAll), the function is applied
    to each dataset in the dictionary with the same arguments. Otherwise, the function
    is applied to the single dataset directly.

    Args:
        func:
            Function that operates on a single xr.Dataset and returns a modified dataset.
            The first parameter must be the dataset.

    Returns:
        wrapper:
            Decorated function that can handle both single datasets and dictionaries
            of datasets.
    """

    @wraps(func)
    def wrapper(
        ds_all: xr.Dataset | DsAll, *args: object, **kwargs: object
    ) -> xr.Dataset | DsAll:
        if isinstance(ds_all, dict):
            return {key: func(ds, *args, **kwargs) for key, ds in ds_all.items()}
        else:
            # Single dataset case
            return func(ds_all, *args, **kwargs)

    return wrapper
