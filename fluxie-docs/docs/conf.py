# docs/conf.py
#
# Sphinx configuration for the fluxie documentation.
# Build locally:  cd docs && make html
# Hosted on:      https://fluxie.readthedocs.io

import os
import sys

# Make the fluxie package importable during the build so autodoc can inspect it.
sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

project = "fluxie"
copyright = "2024, OpenGHG contributors"
author = "OpenGHG contributors"

# Pull the version from the installed package so it stays in sync automatically.
try:
    from importlib.metadata import version as _pkg_version
    release = _pkg_version("fluxie")
except Exception:
    release = "unknown"

version = ".".join(release.split(".")[:2])

# General configuration


extensions = [
    "sphinx.ext.autodoc",           # Extract docstrings from source code
    "sphinx.ext.autosummary",       # Generate summary tables for modules/classes
    "sphinx.ext.napoleon",          # Parse NumPy- and Google-style docstrings
    "sphinx.ext.viewcode",          # Add [source] links next to each symbol
    "sphinx.ext.intersphinx",       # Cross-link to NumPy, xarray, pandas, etc.
    "sphinx_autodoc_typehints",     # Render type hints in the signature line
    "myst_parser",                  # Allow Markdown (.md) source files
    "nbsphinx",                     # Render Jupyter notebooks
]

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

templates_path    = ["_templates"]
exclude_patterns  = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]



html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "titles_only": False,
}

html_static_path = ["_static"]

nbsphinx_execute = "never"
