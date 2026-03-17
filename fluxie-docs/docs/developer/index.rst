Developer guide
===============

This page describes how to set up a development environment, run the test
suite, build the documentation locally, and submit contributions.

.. contents:: On this page
   :local:
   :depth: 2

Setting up a development environment
-------------------------------------

Clone the repository and install fluxie together with all development and
documentation dependencies::

    git clone https://github.com/openghg/fluxie.git
    cd fluxie
    pip install -e ".[docs]"

The ``.[docs]`` extra installs Sphinx, the Read the Docs theme, and all other
packages needed to build the documentation.

Running the tests
-----------------

The test suite uses `pytest <https://docs.pytest.org>`_::

    pip install pytest
    pytest tests/

To run a single test file::

    pytest tests/test_<module>.py -v

Building the documentation locally
------------------------------------

::

    cd docs
    make html

Open ``docs/_build/html/index.html`` in a browser to review the result.

To rebuild from scratch (clears cached output)::

    make clean html

Coding conventions
------------------

- Follow `PEP 8 <https://peps.python.org/pep-0008/>`_ for all Python code.
- Write **NumPy-style docstrings** for every public function, class, and
  module.  The API reference is generated directly from these docstrings, so
  accuracy and completeness matter.

  Minimal example::

      def load_flux(path: str, species: str) -> xr.Dataset:
          """Load a flux NetCDF file into an xarray Dataset.

          Parameters
          ----------
          path : str
              Path to the NetCDF file.
          species : str
              Species identifier used in the filename (e.g. ``"hfc134a"``).

          Returns
          -------
          xr.Dataset
              Dataset containing the flux variables defined in the
              PARIS-AVENGERS-EYECLIMA template.

          Raises
          ------
          FileNotFoundError
              If no file matching ``path`` exists on disk.
          """

- Add or update tests for every new function or bug fix.
- Keep pull requests focused: one logical change per PR makes review faster.

Adding a new tutorial notebook
--------------------------------

1. Create the notebook under ``scripts/``.
2. Add a corresponding ``.nblink`` file in ``docs/tutorials/``::

       {
           "path": "../../scripts/<your_notebook>.ipynb"
       }

3. Add the notebook name (without extension) to the ``toctree`` in
   ``docs/tutorials/index.rst``.
4. Set ``nbsphinx_execute = "never"`` in ``docs/conf.py`` (already the
   default) so the notebook is not re-executed during the documentation build.

Making a release
----------------

1. Update the version string in ``pyproject.toml``.
2. Update ``CHANGELOG.md`` (if present) with a summary of changes.
3. Commit the version bump and push to ``devel``::

       git add pyproject.toml
       git commit -m "Bump version to X.Y"
       git push origin devel

4. Create and push a git tag matching the version::

       git tag vX.Y
       git push origin vX.Y

   Read the Docs will automatically build a versioned documentation snapshot
   for the new tag.

Contributing
------------

Contributions are welcome.  The recommended workflow is:

1. Open an issue on the `GitHub repository <https://github.com/openghg/fluxie/issues>`_
   to describe the proposed change and discuss the approach with the maintainers
   before writing code.
2. Fork the repository and create a feature branch::

       git checkout -b feature/short-description

3. Make your changes, add tests, and ensure the test suite passes.
4. Open a pull request against the ``devel`` branch and link it to the
   relevant issue.

For questions or discussion, use the issue tracker rather than direct contact.
