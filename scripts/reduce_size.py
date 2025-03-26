"""Small script to produce test data.

This reduces the size of the data by a large factor,
so that the tests run faster.
"""

# %%
import xarray as xr
from pathlib import Path


input_dir = Path(r"C:\Users\coli\Documents\Data\paris\test_orginial")
out_dir = Path(r"C:\Users\coli\Documents\Data\paris\test_smaller")
# %%
for fp in input_dir.glob("**/*.nc"):
    print(fp)
    try:
        ds = xr.open_dataset(fp)
    except Exception as e:
        raise ValueError(f"Error reading {fp}") from e

    if "latitude" in ds.dims:
        # Reduce the resolution of the lat and lon
        ds_small = ds.coarsen(latitude=20, longitude=20, boundary="trim").mean()
    elif "time" in ds.dims:
        # Reduce the resolution of the time
        ds = ds.drop_vars(
            ["median_poll_uncert_flag"], errors="ignore"
        )  # variable that is not averagable
        ds_small = ds.coarsen(time=37, boundary="trim").mean()
    else:
        raise ValueError("Unknown dimensions")

    out_subdir = out_dir / fp.relative_to(input_dir).parent
    out_subdir.mkdir(parents=True, exist_ok=True)
    ds_small.to_netcdf(out_subdir / fp.name)
