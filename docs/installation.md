# Installation

fluxie requires Python 3.10 or later.

## From source

Clone the repository and install fluxie in editable mode:

```bash
git clone https://github.com/openghg/fluxie.git
cd fluxie
pip install -e .
```

This also gives you the default configuration files (`configs/`), the example
[notebooks](user_guide/notebooks.md) (`scripts/`) and the test data (`data/tests`)
used in the notebooks and in the [gallery](gallery.md).

## From PyPI

```bash
pip install fluxie
```

## On the ICOS Jupyter Hub

Create and activate a virtual environment:

```bash
python -m venv fluxie-env
source fluxie-env/bin/activate
```

Install the IPython kernel package into the environment and register the
environment as a new Jupyter kernel:

```bash
pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name fluxie-env --display-name "fluxie-env"
```

Then install fluxie from source as described above.

When opening a notebook, make sure that the newly created environment
(`fluxie-env`) is selected as kernel in the drop-down list in the upper right
corner. If it does not show up, restart your Jupyter instance following
[these instructions](https://icos-carbon-portal.github.io/jupyter/how_to/#restart-your-jupyter-instance).

## Installing the documentation dependencies

See [Documentation](developer/docs.md) in the developer guide.
