from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from fluxie.test_utils import data_dir


def create_test_inventory(
    output_dir: Path,
    species: str = "ch4",
    inventory_filename: str = "EUROPE_EDGAR",
    inventory_year: int = 2021,
    sectors: list[str] = [
        "agriculture",
        "waste",
        "energy",
        "industry",
    ],
    countries=["UK", "DE", "BE", "NL"],
) -> Path:
    """Create a small NetCDF inventory for tests."""

    inventory_dir = output_dir / "inventory"
    inventory_dir.mkdir(parents=True, exist_ok=True)

    times = pd.date_range("2020-01-01", periods=2, freq="YS")

    values = np.array(
        [
            [100.0, 120.0, 80.0, 90.0],
            [110.0, 125.0, 85.0, 95.0],
        ]
    )
    uncertainty = np.full_like(values, 5.0)

    data_vars = {
        "inventory": (
            ("time", "country"),
            values,
            {"units": "Gg/yr"},
        ),
        "stdev_flux_total_inventory_country": (
            ("time", "country"),
            uncertainty,
            {"units": "Gg/yr"},
        ),
    }

    for index, sector in enumerate(sectors, start=1):
        sector_values = values * index / len(sectors)
        data_vars[f"flux_{sector}_inventory_country"] = (
            ("time", "country"),
            sector_values,
            {"units": "Gg/yr"},
        )
        data_vars[f"stdev_flux_{sector}_inventory_country"] = (
            ("time", "country"),
            np.full_like(sector_values, 5.0),
            {"units": "Gg/yr"},
        )

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={"time": times, "country": countries},
        attrs={"missing_data": "[]"},
    )

    output_path = inventory_dir / f"{inventory_filename}_{species}_{inventory_year}.nc"
    dataset.to_netcdf(output_path)
    return output_path


def main() -> None:
    output_path = create_test_inventory(
        output_dir=data_dir,
        species="ch4",
        inventory_filename="EUROPE_EDGAR",
        inventory_year=2021,
    )
    print(f"Created {output_path}")


if __name__ == "__main__":
    main()
