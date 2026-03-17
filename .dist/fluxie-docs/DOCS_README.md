# fluxie — Sphinx documentation

This directory contains the source files for the fluxie Sphinx documentation,
hosted at https://fluxie.readthedocs.io.

## Directory layout

```
docs/
├── conf.py                 Sphinx configuration
├── index.rst               Root table of contents
├── getting_started.rst     Installation and quick-start guide
├── data_format.rst         Input file formats and naming conventions
├── requirements.txt        Pinned build dependencies for Read the Docs
├── api/
│   └── index.rst           Auto-generated API reference entry point
├── developer/
│   └── index.rst           Developer guide (setup, tests, contributing)
├── tutorials/
│   ├── index.rst           Tutorial listing
│   └── example_basics.nblink  Links to scripts/example_basics.ipynb
└── _static/                Custom CSS / images (add as needed)

.readthedocs.yaml           Read the Docs build configuration (repo root)
pyproject_docs_addition.toml  Snippet to add to pyproject.toml
```

## Building locally

1. Install the documentation dependencies:

   ```bash
   pip install -e ".[docs]"
   ```

2. Build the HTML output:

   ```bash
   cd docs
   make html
   ```

3. Open `docs/_build/html/index.html` in a browser.

To rebuild from scratch:

```bash
make clean html
```

## Connecting to Read the Docs

1. Sign in to https://readthedocs.org with the openghg GitHub account.
2. Click **Import a project** and select `openghg/fluxie`.
3. Read the Docs detects `.readthedocs.yaml` automatically and uses it for all
   subsequent builds.
4. Every push to `devel` triggers a new build.  Git tags (e.g. `v2.1`) appear
   as versioned documentation snapshots.

## Adding content

- **New page:** Create a `.rst` (or `.md`) file in the appropriate
  subdirectory and add its name to the nearest `toctree` directive.
- **New tutorial notebook:** Add the notebook to `scripts/`, create a
  matching `.nblink` stub in `docs/tutorials/`, and add the name to the
  `toctree` in `docs/tutorials/index.rst`.
- **API coverage:** The API reference is built from docstrings.  Any public
  function or class that lacks a docstring will appear as an empty entry.
  Write NumPy-style docstrings to populate it.
