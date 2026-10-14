# Adding a plot function

In fluxie, each plot corresponds to a function in the `fluxie/plots` package.

## Before you start

Check the [gallery](../gallery.md) and the [API](../api/plots.md)
for a similar function. If one exists, it is usually better to add an option to
it than to create a new function.

## Guidelines

- **Start by copying an existing function** producing a similar plot.
- **Use the same argument names and data structures** as the other functions
  (`ds_all`, `species`, `config_data`, `model_colors`, `model_labels`, ...;
  see [How fluxie works](../user_guide/principles.md#3-plotting-one-function-per-plot)).
- **Make the defaults look good.** The plot should look good when called with
  only the required arguments.
- **Return the figure** (`matplotlib.figure.Figure`) so users can customize or
  save it. Do not call `plt.show()` or save the figure in the function.
- **Put data processing in operators.** If the plot needs data processing that
  could be reused, write it as a function in `fluxie/operators`.
- **Use logging, not print.** fluxie uses the standard
  [logging](https://docs.python.org/3/library/logging.html) module
  (`logger = logging.getLogger(__name__)`).

## Docstrings

The [API documentation](../api/index.md) is generated from the docstrings.
Use the [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings),
and give the types **only in the signature**, not in the docstring:

```python
def plot_something(
    ds_all: dict[str, xr.Dataset],
    species: str,
    region: str | None = None,
) -> plt.Figure:
    """
    Plot something for all models.

    Args:
        ds_all:
            Dictionary of model datasets.
        species:
            Gas species, e.g. 'ch4'.
        region:
            Region to plot, e.g. 'FRANCE'. If None, plots the whole domain.

    Returns:
        fig:
            The matplotlib figure.
    """
```

These rules are checked automatically, locally with the commands below and in
the [continuous integration](contributing.md#continuous-integration):

```bash
ruff check ./fluxie ./docs   # docstring format and type hints in the signature
pydoclint ./fluxie ./docs    # docstring arguments match the signature, no types in the docstring
```

| Check | Tool |
|:--|:--|
| Google style docstrings, summary on the line after the opening quotes | ruff (`D` rules) |
| Type hints for all arguments and return values | ruff (`ANN` rules) |
| All arguments documented, with the names of the signature | pydoclint |
| No types in the docstring | pydoclint |
| A `Returns:` section when the function returns something | pydoclint |

Small helper functions can have a docstring with only a one-line summary; the
arguments are then not checked. `ruff check --fix` corrects most formatting
issues automatically. The configuration of both tools is in `pyproject.toml`.

## Checklist

- [ ] The function is in the appropriate module of `fluxie/plots`.
- [ ] The docstring documents all arguments.
- [ ] The function is added to the [API page](../api/plots.md)
      (`docs/api/plots.md`).
- [ ] A test is added in `tests/test_plots.py` (see [Tests](tests.md)).
- [ ] An example is added to the gallery (see [Documentation](docs.md#adding-a-gallery-example)).
