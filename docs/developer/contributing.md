# Contributing

fluxie is an open-source project and contributions are very welcome:
new plots, bug fixes, documentation, examples...

!!! tip "You used fluxie in a publication?"
    You don't need to be a developer to contribute: add your publication to the
    [Hall of fame](../hall_of_fame.md#add-your-publication) directly from your
    web browser.

## Discussing new features

Before starting to work on a new feature or a bug fix, please
[open an issue](https://github.com/openghg/fluxie/issues) to discuss it with the
maintainers. This avoids duplicated work and helps you find the best way to
integrate your changes. For bugs, include a minimal example and the error message.

## What goes into fluxie

- **Plot functions and operators** are part of the fluxie core. They are tested
  and maintained by the fluxie team.
- **Preprocessing scripts** that convert the output of a specific model into the
  fluxie format can be added to `scripts/preprocessing` as examples, but are not
  maintained by the fluxie team.


## Workflow

1. Fork the repository (or create a branch if you are a member of the
   [openghg](https://github.com/openghg) organization).
2. Install fluxie in editable mode with the development tools:

    ```bash
    pip install -e ".[dev,docs]"     # development and documentation tools
    ```
3. Create a branch from `devel` and make your changes.
4. Add tests for your changes (see [Tests](tests.md)), and a gallery example if
   you add a plot (see [Adding a plot function](new_plot.md)).
5. Format your code with [black](https://black.readthedocs.io), and check the
   docstrings and type hints (see [Docstrings](new_plot.md#docstrings)):

    ```bash
    black ./fluxie ./tests ./docs
    ruff check ./fluxie ./docs
    pydoclint ./fluxie ./docs
    ```
6. Ask an AI to review your code.
7. Open a pull request to the `devel` branch, and link the corresponding issue.

The maintainers will review your pull request and may suggest improvements.


## Continuous integration

Each pull request is checked with GitHub Actions
(`.github/workflows/python-app.yml`). The checks must pass before merging:

- code formatting with `black --check ./tests ./fluxie ./docs`,
- docstrings and type hints with `ruff check ./fluxie ./docs` and
  `pydoclint ./fluxie ./docs` (see [Docstrings](new_plot.md#docstrings)),
- unit tests with `pytest`,
- execution of the example notebooks `scripts/example_basics.ipynb` and
  `scripts/example_ecflux.ipynb`.

Read the Docs also builds a preview of the documentation for each pull request,
including the [gallery](../gallery.md), so reviewers can see the
new plots. The link is shown in the checks of the pull request.


## Credits

If you contribute to fluxie, add yourself to `CREDITS.md` in your pull request
(see [Credits](../credits.md)).
