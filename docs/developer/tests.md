# Tests

Each function in fluxie should have a unit test. Tests are run with
[pytest](https://docs.pytest.org) and are located in the `tests/` folder.

```bash
pip install -e ".[dev]"
pytest            # run all the tests
pytest tests/test_plots.py -k flux_map   # run a subset of the tests
```

Tests are also run automatically on every pull request
(see [Continuous integration](contributing.md#continuous-integration)).

## Test data

Small test datasets are stored in the `data/tests` folder of the repository:

| Folder | Content |
|:--|:--|
| `InTEM`, `ELRIS`, `RHIME` | Flux and concentration outputs of three inversion models, for `hfc134a` (yearly) and `ch4` (monthly) |
| `sector_flux` | Prior sector fluxes for `ch4` |
| `ecflux` | Eddy covariance fluxes |
| `configs` | Small cell area file for the tests |

The path to this folder is available as `fluxie.test_utils.data_dir`:

```python
from fluxie.io import read_config_files, read_model_output
from fluxie.test_utils import data_dir

config_data = read_config_files()
ds_all = read_model_output(
    data_dir, "flux", "hfc134a", ["InTEM_NAME_EUROPE_EDGAR"], config_data
)
```

Keep the test files small: they are stored in the git repository.
If you need new test data, discuss it in an issue first.

## Writing tests

- Test the function with its default options, and also with special cases
  (other options, missing variables, several models...).
- Test that wrong inputs raise a clear error, using `pytest.raises`:

    ```python
    def test_country_flux_with_inventory_raises_no_datadir():
        with pytest.raises(ValueError, match="data_dir must be provided"):
            plot_country_flux(ds_all_flux, species, plot_inventory=True)
    ```

- Plot tests (`tests/test_plots.py`) check that the functions run without error.
  The visual result is checked in the [gallery](../gallery.md)
  of the documentation preview of the pull request.
