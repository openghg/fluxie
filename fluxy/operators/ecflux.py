import logging
from fluxy.types import DsAll
from fluxy.operators.utils import apply_to_dict_or_single
import xarray as xr


@apply_to_dict_or_single
def filter_ecflux(
    ds: xr.Dataset | DsAll,
    min_footprint_ratio: float = 0.8,
    flux_range: tuple[float | None, float | None] = (None, None),
    qa_flag: int | list[int] = 1,
    qa_blh: list[int] = [0],
    min_friction_velocity: float = 0.2,
):
    """Filter the eddy covariance data.

    Parameters
    ----------
    ds :
        Dataset or dictionary of datasets containing eddy covariance flux data.
        The decorator will handle applying the function to each dataset in the dictionary.
    min_footprint_ratio :
        Minimum footprint ratio to consider a flux valid, by default 0.8
    flux_range :
        Minimum and maximum flux values to filter out.
    qa_flag :
        Maximum quality assurance flag value to consider a flux valid, by default 1
        * 0: high quality
        * 1: moderate quality
        * 2: low quality
        if given as a list, keep only the fluxes with the given qa_flag values,
        (useful if you want to keep only the low quality fluxes, e.g. [2])
    qa_blh :
        Boundary layer height quality assurance flag, by default [0]
        * 0: below boundary layer height
        * 1: above boundary layer height
        if given as a list, keep only the fluxes with the given qa_blh values,
        (useful if you want to keep only the fluxes below the boundary layer height, e.g. [0])
    min_friction_velocity :
        Minimum friction velocity to consider a flux valid, by default 0.2 m/s
        The friction velocity threshold helps to avoid low turbulence situation

    Returns
    -------
    xr.Dataset or DsAll
        Filtered dataset(s) with the same structure as input.
    """
    logger = logging.getLogger(__name__)

    mask = xr.ones_like(ds["index"], dtype=bool)

    if "footprint_coverage_fraction" in ds:
        mask &= ds["footprint_coverage_fraction"] >= min_footprint_ratio
    else:
        logger.warning(
            "No `footprint_coverage_fraction` found in the dataset. "
            "Skipping footprint ratio filtering."
        )

    if (
        flux_range[0] is not None
        and flux_range[1] is not None
        and flux_range[0] >= flux_range[1]
    ):
        # Check that min is less than max
        raise ValueError(
            f"Minimum flux value must be less than maximum flux value. {flux_range=}"
        )

    if flux_range[0] is not None:
        mask &= ds["ecflux_observed"] >= flux_range[0]
    if flux_range[1] is not None:
        mask &= ds["ecflux_observed"] <= flux_range[1]

    if "qa_flag" not in ds:
        logger.warning(
            "No `qa_flag` found in the dataset. "
            "Skipping quality assurance flag filtering."
        )
    else:
        if isinstance(qa_flag, list):
            mask &= ds["qa_flag"].isin(qa_flag)
        else:
            mask &= ds["qa_flag"] <= qa_flag

    if "qa_blh" not in ds:
        logger.warning(
            "No `qa_blh` found in the dataset. "
            "Skipping boundary layer height quality assurance flag filtering."
        )
    else:
        mask &= ds["qa_blh"].isin(qa_blh)

    if "friction_velocity" not in ds:
        logger.warning(
            "No `friction_velocity` found in the dataset. "
            "Skipping friction velocity filtering."
        )
    else:
        mask &= ds["friction_velocity"] >= min_friction_velocity

    ds_out = ds.where(mask, drop=True)

    logger.info(
        f"Filtered  {len(ds_out['index'])} / {len(ds['index'])} eddy covariance fluxes"
    )

    return ds_out
