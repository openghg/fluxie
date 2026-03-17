Getting started
===============

Requirements
------------

Fluxie requires Python 3.9 or later.  All other dependencies are declared in
``pyproject.toml`` and are installed automatically.

Installation
------------

Local machine
~~~~~~~~~~~~~

Clone the repository and install fluxie in editable mode so that any local
changes are immediately reflected without reinstalling::

    git clone https://github.com/openghg/fluxie.git
    cd fluxie
    pip install -e .

ICOS Jupyter Hub
~~~~~~~~~~~~~~~~

On the ICOS Jupyter Hub you must first create and activate a dedicated virtual
environment before installing the package.

Create and activate the environment::

    python -m venv fluxie-env
    source fluxie-env/bin/activate

Register the environment as a Jupyter kernel so you can select it in the
notebook interface::

    pip install --upgrade pip
    pip install ipykernel
    python -m ipykernel install --user --name fluxie-env --display-name "fluxie-env"

Install fluxie::

    git clone https://github.com/openghg/fluxie.git
    cd fluxie
    pip install -e .

.. note::

   After installation, select ``fluxie-env`` from the kernel drop-down in the
   upper-right corner of the Jupyter interface.  If the kernel does not appear,
   restart your Jupyter instance following the
   `ICOS instructions <https://icos-carbon-portal.github.io/jupyter/how_to/#restart-your-jupyter-instance>`_.

Quick start
-----------

The fastest way to explore fluxie is to open and run the bundled example
notebook::

    scripts/example_basics.ipynb

It uses test data shipped with the repository and demonstrates the core
workflow: loading model output, selecting date ranges, and producing the
standard set of comparison plots.

See :doc:`tutorials/index` for a rendered version of this notebook and
additional examples.

Next steps
----------

- Read the :doc:`data_format` guide to learn how to structure your own model
  output files so fluxie can read them.
- Browse the :doc:`api/index` for the complete function and class reference.
- See the :doc:`developer/index` if you want to contribute to the project.
