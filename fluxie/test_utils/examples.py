"""Load the test data used in the documentation gallery."""

from dataclasses import dataclass, field

import xarray as xr

from fluxie.config import set_model_colors, set_model_labels, set_print_settings
from fluxie.io import read_config_files, read_model_output
from fluxie.operators.flux_scale_by_sector_proportions import (
    scale_by_sector_proportions,
)
from fluxie.operators.sectors import group_sectors
from fluxie.operators.select import slice_flux, slice_mf
from fluxie.test_utils import data_dir

MODELS = [
    "InTEM_NAME_EUROPE_EDGAR_old_format",
    "ELRIS_NAME_EUROPE_EDGAR_old_format",
    "RHIME_NAME_EUROPE_EDGAR_old_format",
]

ECFLUX_MODELS = ["EDDY_HARDAU", "EDDY_HARDAU_STORAGE_2LAYERS"]


@dataclass
class ExampleData:
    """Datasets and settings commonly passed to the plot functions."""

    ds_all: dict[str, xr.Dataset]
    species: str
    models: list[str]
    config_data: dict[str, dict]
    start_date: str | None = None
    end_date: str | None = None
    model_colors: dict[str, str] = field(init=False)
    model_labels: dict[str, str] = field(init=False)
    annotate_coords: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.model_colors = set_model_colors(self.models)
        self.model_labels = set_model_labels(self.models, self.config_data, False)
        self.annotate_coords = set_print_settings()


def load_flux(
    species: str = "hfc134a",
    period: str = "yearly",
    start_date: str = "2018-01-01",
    end_date: str = "2024-01-01",
    add_sites_to_flux: bool = False,
    models: list[str] = MODELS,
    **slice_kwargs: str | None,
) -> ExampleData:
    """
    Read and slice the flux test data.

    Args:
        species:
            Gas species, e.g. 'ch4'.
        period:
            Inversion period, 'yearly' or 'monthly'.
        start_date:
            Start of the selected period (inclusive).
        end_date:
            End of the selected period (exclusive).
        add_sites_to_flux:
            If True, add the sites of the concentration files to the flux datasets.
        models:
            Model runs to read.
        **slice_kwargs:
            Passed to `fluxie.operators.select.slice_flux`, e.g. the units.

    Returns:
        data:
            The sliced flux datasets with their settings.
    """
    config_data = read_config_files()
    ds_all = read_model_output(
        data_dir,
        "flux",
        species,
        models,
        config_data,
        period=period,
        add_sites_to_flux=add_sites_to_flux,
    )
    ds_all = slice_flux(
        ds_all, config_data, start_date, end_date, species=species, **slice_kwargs
    )
    return ExampleData(ds_all, species, models, config_data, start_date, end_date)


def load_sector_flux(regions: list[str]) -> ExampleData:
    """
    Read the ch4 flux test data and split the country fluxes into sectors.

    Args:
        regions:
            Regions for which the sector totals are computed.

    Returns:
        data:
            The flux datasets with the sector fluxes added, with their settings.
    """
    units = "Gg yr-1"
    data = load_flux(
        species="ch4",
        period="monthly",
        start_date="2022-01-01",
        country_flux_units_print=units,
    )
    data.ds_all = scale_by_sector_proportions(
        data_dir=data_dir,
        ds_all=data.ds_all,
        species=data.species,
        country_flux_units_print=units,
        config_data=data.config_data,
        regions=regions,
        sector_file="EUROPE_EDGAR",
        create_region_sector_totals=True,
        cell_area_test_file=True,
    )
    return data


def load_mf(
    site: str | None = "MHD",
    start_date: str = "2018-01-01",
    end_date: str = "2019-01-01",
    select: bool = True,
) -> ExampleData:
    """
    Read the mole fraction test data.

    Args:
        site:
            Site to select. If None, all sites are kept.
        start_date:
            Start of the selected period (inclusive).
        end_date:
            End of the selected period (exclusive).
        select:
            If False, return the data as read, without selecting site and period.

    Returns:
        data:
            The mole fraction datasets with their settings.
    """
    species = "hfc134a"
    config_data = read_config_files()
    ds_all = read_model_output(
        data_dir, "concentration", species, MODELS, config_data, period="yearly"
    )
    if select:
        ds_all = slice_mf(
            ds_all, start_date, end_date, site, data_dir=data_dir, mf_units_print="ppt"
        )
    return ExampleData(ds_all, species, MODELS, config_data, start_date, end_date)


def load_ecflux(sectors: bool = False) -> ExampleData:
    """
    Read the eddy covariance test data.

    Args:
        sectors:
            If True, group the inventory sectors as defined in the sectors config.

    Returns:
        data:
            The eddy covariance flux datasets with their settings.
    """
    species = "co2"
    config_data = read_config_files()
    ds_all = read_model_output(
        data_dir / "ecflux", "eddy_flux", species, ECFLUX_MODELS, config_data
    )
    if sectors:
        ds_all = group_sectors(ds_all, sectors_config=config_data["sectors"])
    return ExampleData(ds_all, species, ECFLUX_MODELS, config_data)
