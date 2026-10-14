# Documentation

The documentation is built with [Zensical](https://zensical.org).
It is hosted on [Read the Docs](https://fluxie.readthedocs.io), which builds
it for every commit on `devel` and every pull request.

## Building the documentation locally

Install the documentation dependencies, generate the gallery and the hall of
fame and start a local server:

```bash
pip install -e ".[docs]"
python docs/build_pages.py
zensical serve
```

The documentation is then available at <http://localhost:8000>
and is rebuilt every time you save a file.
Use `zensical build` to build the static site in the `site/` folder.

!!! note
    `docs/build_pages.py` runs the gallery examples, which takes less than a
    minute. Run it again after changing plotting code or `docs/hall_of_fame.bib`.

## Structure

```
zensical.toml            # Configuration and navigation
.readthedocs.yaml        # Read the Docs configuration
docs/
    index.md             # Home page
    user_guide/          # User guide pages
    api/                 # API pages, generated from the docstrings
    developer/           # Developer guide pages
    gallery.py           # Examples shown in the gallery
    hall_of_fame.bib     # Publications shown in the hall of fame
    build_pages.py       # Generates the gallery and the publications list
    generated/           # Output of build_pages.py (not in git)
```

Some pages include other files with the
[snippets](https://facelessuser.github.io/pymdown-extensions/extensions/snippets/)
extension (`--8<-- "path/to/file"`):

- the credits page includes `CREDITS.md` from the root of the repository, so it
  is not duplicated in `docs/`,
- the gallery and hall of fame pages include the files generated in
  `docs/generated/` by `docs/build_pages.py`.

## Hall of fame

The publications of the [Hall of fame](../hall_of_fame.md) are stored in
`docs/hall_of_fame.bib`. `docs/build_pages.py` reads this file and generates the
list of publications, sorted by year. To add a publication, paste its BibTeX
entry in the file (see [Add your publication](../hall_of_fame.md#add-your-publication)).

The pages are written in Markdown, with the
[Zensical extensions](https://zensical.org/docs/)
(admonitions, tabs, ...).

## API documentation

The API pages use [mkdocstrings](https://mkdocstrings.github.io) to render the
docstrings. To add a function, add a line with its full path in the
corresponding page of `docs/api/`:

```markdown
::: fluxie.plots.flux_map.plot_flux_map
```

You can link to a function from any page with
`` [`plot_flux_map`][fluxie.plots.flux_map.plot_flux_map] ``.

## Adding a gallery example

The [gallery](../gallery.md) is a single page generated before the
documentation is built: the script `docs/build_pages.py` runs the examples defined in
`docs/gallery.py` and shows, for each of them, its title, a link to the API
documentation of the plot function and the resulting figure.

To add an example, add a function to `docs/gallery.py` decorated with
`@example(title, plot_function)` and returning the figure:

```python
section = "Flux maps"


@example("Prior, posterior and difference", plot_flux_map)
def _():
    data = load_flux_map()
    return plot_flux_map(
        data.ds_all,
        data.species,
        region="EUROPE",
        config_data=data.config_data,
        model_labels=data.model_labels,
    )
```

- Examples appear in the order of the file, under the heading given by the
  last `section = ...` assignment.
- `plot_function` is used to create the link to the API documentation.
- To show another option of the same function, add another example with its
  own title.
- If an example fails, the script shows a warning and the gallery shows a
  failure notice instead of the figure.

### Loading the test data

The test data are loaded with the helpers of `fluxie.test_utils.examples`,
which avoids repeating the same reading and selection code in every example:

| Function | Data |
|:--|:--|
| `load_flux(...)` | Fluxes of the three test models (`hfc134a` by default), sliced with `slice_flux` |
| `load_sector_flux(regions)` | `ch4` fluxes split into sectors |
| `load_mf(site="MHD", ...)` | Mole fractions, sliced with `slice_mf` (`site=None` for all sites, `select=False` for the raw data) |
| `load_ecflux(sectors=False)` | Eddy covariance fluxes |

They return an `ExampleData` object with the attributes commonly passed to the
plot functions: `ds_all`, `species`, `models`, `config_data`, `model_colors`,
`model_labels`, `annotate_coords`, `start_date` and `end_date`.
If your plot needs other data, add a helper to this module.
