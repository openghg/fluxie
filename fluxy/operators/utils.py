from functools import wraps

import xarray as xr

from fluxy.types import DsAll


def apply_to_dict_or_single(func):
    """Decorator that applies a function to either a single dataset or a dictionary of datasets.

    If the first argument is a dictionary of datasets (DsAll), the function is applied
    to each dataset in the dictionary with the same arguments. Otherwise, the function
    is applied to the single dataset directly.

    Parameters
    ----------
    func : callable
        Function that operates on a single xr.Dataset and returns a modified dataset.
        The first parameter must be the dataset.

    Returns
    -------
    callable
        Decorated function that can handle both single datasets and dictionaries of datasets.
    """

    @wraps(func)
    def wrapper(ds_all, *args, **kwargs):
        if isinstance(ds_all, dict):
            return {key: func(ds, *args, **kwargs) for key, ds in ds_all.items()}
        else:
            # Single dataset case
            return func(ds_all, *args, **kwargs)

    return wrapper
