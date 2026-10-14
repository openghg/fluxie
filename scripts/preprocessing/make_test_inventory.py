"""
Create the synthetic inventory files of the test data.

Writes `data/tests/inventory/TEST_inventory_<species>_2024.nc` for ch4 and
hfc134a, following the template `data/templates/PAR-AVE-EYE_inventory.cdl`.
The inventory is derived from the prior country fluxes of the InTEM test model,
with a small trend and a 10% uncertainty, so that it differs from the prior.
It is not real data: it is only used in the tests and the documentation gallery.

Usage:

    python scripts/preprocessing/make_test_inventory.py
"""

import numpy as np
import pandas as pd
import xarray as xr

from fluxie.test_utils import data_dir
from fluxie.test_utils.examples import load_flux

MODEL = "InTEM_NAME_EUROPE_EDGAR_old_format"
SPECIES_PERIODS = {"hfc134a": "yearly", "ch4": "monthly"}
YEARS = np.arange(2018, 2024)
INVENTORY_FILENAME = "TEST_inventory"
INVENTORY_YEAR = 2024


def load_priors() -> dict[str, xr.DataArray]:
    """Load the prior country fluxes (Gg yr-1) of the test model for each species."""
    priors = {}
    for species, period in SPECIES_PERIODS.items():
        data = load_flux(
            species=species,
            period=period,
            start_date=f"{YEARS[0]}-01-01",
            models=[MODEL],
            country_flux_units_print="Gg yr-1",
        )
        priors[species] = data.ds_all[MODEL]["flux_total_prior_country"]
    return priors


def make_inventory(prior: xr.DataArray, countries: list[str]) -> xr.Dataset:
    """
    Create a yearly inventory dataset from the prior country fluxes.

    Args:
        prior:
            Prior country fluxes, in Gg yr-1.
        countries:
            ISO 3166-1 alpha-3 codes of the countries to include.

    Returns:
        inventory:
            Inventory dataset following the fluxie inventory template.
    """
    yearly = prior.sel(country=countries).groupby("time.year").mean()
    yearly = yearly.reindex(year=YEARS).transpose("year", "country")
    # Years without model data take the value of the closest year
    yearly = pd.DataFrame(yearly.values).bfill().ffill().values
    # Small trend, so that the inventory differs from the prior
    trend = (1.05 - 0.02 * (YEARS - YEARS[0]))[:, None]
    flux = yearly * trend * 1e6  # Gg yr-1 -> kg yr-1

    return xr.Dataset(
        {
            "flux_total_inventory_country": (
                ("time", "country"),
                flux,
                {
                    "units": "kg/yr",
                    "longname": "Test inventory emissions estimate "
                    "(derived from the InTEM test prior)",
                },
            ),
            "stdev_flux_total_inventory_country": (
                ("time", "country"),
                0.1 * flux,
                {
                    "units": "kg/yr",
                    "longname": "Standard deviation of the test inventory "
                    "emissions estimate",
                },
            ),
        },
        coords={
            # Start of the year, as expected by the inventory bar plots
            "time": (
                "time",
                pd.to_datetime([f"{year}-01-01" for year in YEARS]),
                {"long_name": "Start of the inventory year"},
            ),
            "country": (
                "country",
                np.array(countries, dtype=object),
                {"longname": "Country ISO 3166-1 alpha-3 code"},
            ),
        },
        attrs={
            "missing_data": "[]",
            "comment": "Synthetic inventory for the fluxie tests and documentation, "
            "derived from the InTEM test prior. Not real data.",
        },
    )


def main() -> None:
    """Write the test inventory files."""
    priors = load_priors()
    # Single countries (not aggregated regions) available for all species
    countries = sorted(
        set.intersection(
            *(
                {str(c) for c in prior.country.values if "-" not in str(c)}
                for prior in priors.values()
            )
        )
    )
    for species, prior in priors.items():
        path = (
            data_dir
            / "inventory"
            / (f"{INVENTORY_FILENAME}_{species}_{INVENTORY_YEAR}.nc")
        )
        make_inventory(prior, countries).to_netcdf(
            path,
            encoding={
                "time": {
                    "units": "days since 1970-01-01 00:00:00",
                    "calendar": "proleptic_gregorian",
                }
            },
        )
        print(f"Written {path} ({len(countries)} countries)")


if __name__ == "__main__":
    main()
