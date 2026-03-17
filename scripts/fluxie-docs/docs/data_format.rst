Data formatting
===============

This page describes every input file that fluxie can consume, the required
directory layout, and the naming conventions that allow filenames to be
resolved automatically.

All NetCDF templates referenced below are stored in the repository under
``data/templates/``.

.. contents:: On this page
   :local:
   :depth: 2

Required files
--------------

Flux and concentration NetCDF files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These files carry the inversion model output and are the primary input to
fluxie.  They must conform to the **PARIS-AVENGERS-EYECLIMA** template:

- ``data/templates/PAR-AVE-EYE_inversion_flux_output.cdl``
- ``data/templates/PAR-AVE-EYE_inversion_concentration_output.cdl``

A human-readable summary of the most important variables in each template is
provided in ``data/templates/README_templates.md``.

Naming convention
^^^^^^^^^^^^^^^^^

Both files share a common naming scheme based on ``_``-separated tags:

.. code-block:: text

   Flux file:
   <inversionModel>_<optional_tags>_<species>_<frequency>.nc

   Concentration file:
   <inversionModel>_<optional_tags>_<species>_<frequency>_concentration.nc

``<frequency>`` must be exactly ``yearly`` or ``monthly``.

For clear traceability and automatic plot labelling, use the following
pattern for ``<optional_tags>``::

   <transportModel>_<domain>_<prior>

**Example** — InTEM inversion of HFC-134a using NAME transport over Europe
with EDGAR prior:

.. code-block:: text

   InTEM_NAME_EUROPE_EDGAR_hfc134a_yearly.nc
   InTEM_NAME_EUROPE_EDGAR_hfc134a_yearly_concentration.nc

Directory layout
^^^^^^^^^^^^^^^^

Files must be placed under::

   /path/to/data/<inversionModel>/<species>/

For example::

   /path/to/data/InTEM/hfc134a/
       InTEM_NAME_EUROPE_EDGAR_hfc134a_yearly.nc
       InTEM_NAME_EUROPE_EDGAR_hfc134a_yearly_concentration.nc

The path ``/path/to/data/`` is specified in the notebook when you initialise
the analysis.

Optional configuration files
-----------------------------

All configuration files use JSON format.  Working examples for each file are
provided in the ``configs/`` directory of the repository.

Regions information
~~~~~~~~~~~~~~~~~~~

**File:** ``configs/regions_info.json``

Controls how countries are aggregated into named regions and specifies
point-source locations to overlay on maps.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Key
     - Type
     - Description
   * - ``country_codes``
     - ``dict[str, str]``
     - Country names mapped to their ISO 3166-1 alpha-3 codes.
   * - ``regions``
     - ``dict[str, list[str]]``
     - Named regions mapped to lists of country names they contain.
   * - ``point_source``
     - ``dict[str, list[float]]``
     - Named locations mapped to ``[latitude, longitude]`` coordinates.

Models information
~~~~~~~~~~~~~~~~~~

**File:** ``configs/models_info.json``

Controls filename resolution, plot labels, and multi-species aggregation.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Key
     - Type
     - Description
   * - ``filename_tags``
     - ``dict[str, str]``
     - Short alias mapped to a full tag string embedded in filenames.
       For example ``{"std": "4sites_baseline_optimized"}`` means the
       model run ``InTEM_NAME_EDGAR_std`` resolves to
       ``InTEM_NAME_EDGAR_4sites_baseline_optimized``.
       Use ``<model>`` as a placeholder that is replaced by the
       inversion model name in lower case.
   * - ``model_labels``
     - ``dict[str, str]``
     - Model run names mapped to the labels shown on plots.
       If a run is absent from this dictionary the label is constructed
       automatically from the run name.
   * - ``species_name``
     - ``dict[str, dict]``
     - Per-model overrides for the species identifier used in filenames.
       Keys correspond to ``<inversionModel>``.
   * - ``standard_run``
     - ``dict[str, dict]``
     - Identifies the standard run for each model when aggregating over
       all HFCs or PFCs.  Use the ``"default"`` sub-key for the default
       set of name tags; add additional sub-keys (e.g. ``"longrun"``) for
       alternative configurations.

Species information
~~~~~~~~~~~~~~~~~~~

**File:** ``configs/species_info.json``

A dictionary of species identifiers (or group names) mapped to their physical
properties and display settings.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Key
     - Type
     - Description
   * - ``species_print``
     - ``str``
     - Species name for plot axis labels, in LaTeX notation
       (e.g. ``"HFC-134a"``).
   * - ``gwp``
     - ``float``
     - GWP-100 value from IPCC AR5, used to convert country fluxes to
       CO\ :sub:`2`-equivalent mass.
   * - ``molar_mass``
     - ``float``
     - Molar mass in g mol\ :sup:`-1`, used for mol–g conversions.
   * - ``list_species``
     - ``list[str]``
     - List of species identifiers that make up a group.
       Used when plotting the sum of country fluxes across multiple species.

Sites information
~~~~~~~~~~~~~~~~~

**File:** ``configs/site_info.json``

A dictionary of station codes mapped to per-network observation metadata.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Key
     - Type
     - Description
   * - ``latitude``
     - ``float``
     - Latitude of the station in degrees North.
   * - ``longitude``
     - ``float``
     - Longitude of the station in degrees East.
   * - ``height_station_masl``
     - ``float``
     - Station elevation in metres above sea level.
   * - ``long_name``
     - ``str``
     - Full descriptive name of the station.
   * - ``height``
     - ``list[str]``
     - Inlet heights in metres above ground level.
   * - ``height_name``
     - ``list[str]``
     - Human-readable labels for each inlet height.

Optional input files
--------------------

Bottom-up inventory NetCDF
~~~~~~~~~~~~~~~~~~~~~~~~~~

Used to overlay gridded inventory estimates on flux maps.

- **Template:** ``data/templates/PAR-AVE-EYE_inventory.cdl``
- **Location:** ``/path/to/data/inventory/``
- **Naming:** ``<inventory_identifier>_<species>_<year>.nc``

  Example: ``UNFCCC_inventory_hfc134a_2024.nc``

Baseline timestamp NetCDF
~~~~~~~~~~~~~~~~~~~~~~~~~

Contains timestamps that flag background (baseline) periods at each
measurement station.

- **Template:** ``data/templates/PAR-AVE-EYE_baseline_timestamps.cdl``
- **Location:** ``/path/to/data/baseline_timestamps/``
- **Naming:** ``<stationID>_<baseline_identifier>_baseline_timestamps.nc``

  Example: ``JFJ_InTEM_baseline_timestamps.nc``
