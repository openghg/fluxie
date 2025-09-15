import logging
logger = logging.getLogger(__name__)

# Cities to plot
point_markers = {
    "UK": ["london", "edinburgh", "cardiff", "belfast"],
    "SWITZERLAND": ["bern", "zurich", "geneva", "basel", "lausanne"],
    "GERMANY": ["berlin", "hamburg", "munich", "koeln", "frankfurt", "essen"],
    "ITALY": ["rome", "milan", "naples", "turin", "palermo"],
    "NETHERLANDS": ["amsterdam", "rotterdam", "hague", "utrecht", "eindhoven"],
    "IRELAND": ["dublin", "cork", "limerick", "galway", "waterford"],
    "HUNGARY": ["budapest", "debrecen", "miskolc", "szeged", "pecs"],
    "NORWAY": ["oslo", "bergen", "sandnes", "stavanger", "drammen"],
    "BELGIUM": ["brussels", "antwerp", "ghent", "charleroi", "liege"],
}

point_markers["BENELUX"] = (
    point_markers["NETHERLANDS"] + point_markers["BELGIUM"] + ["luxembourg"]
)

# Start date of F-gases country fluxes
start_date_fgases = {
    "UK": "2008-01-01",
    "SWITZERLAND": "2008-01-01",
    "GERMANY": "2013-01-01",
    "ITALY": "2008-01-01",
    "NETHERLANDS": "2013-01-01",
    "BELGIUM": "2013-01-01",
    "BENELUX": "2013-01-01",
    "IRELAND": "2008-01-01",
    "HUNGARY": "2018-01-01",
    "NORWAY": "2018-01-01",
}

# Specify the percentile to use for the color scales in the flux spatial map
fluxlim_percentiles = {
    "UK": {
        "ch4": 0.95,
        "n2o": 0.95,
        "hfc32": 0.99,
        "hfc125": 0.99,
        "hfc134a": 0.99,
        "hfc143a": 0.99,
        "cf4": 0.99,
        "pfc116": 0.95,
        "pfc218": 0.99,
        "pfc318": 0.95,
        "sf6": 0.99,
    },
    "SWITZERLAND": {
        "ch4": 0.96,
        "n2o": 0.96,
        "hfc32": 0.98,
        "hfc125": 0.98,
        "hfc134a": 0.97,
        "hfc143a": 0.97,
        "cf4": 0.98,
        "pfc116": 0.98,
        "pfc218": 0.975,
        "pfc318": 0.96,
        "sf6": 0.99,
    },
    "GERMANY": {
        "ch4": 0.97,
        "n2o": 0.97,
        "hfc32": 0.99,
        "hfc125": 0.99,
        "hfc134a": 0.99,
        "hfc143a": 0.99,
        "cf4": 0.995,
        "pfc116": 0.995,
        "pfc218": 0.98,
        "pfc318": 0.99,
        "sf6": 0.99,
    },
    "ITALY": {
        "ch4": 0.95,
        "n2o": 0.95,
        "hfc32": 0.99,
        "hfc125": 0.99,
        "hfc134a": 0.99,
        "hfc143a": 0.99,
        "cf4": 0.99,
        "pfc116": 0.99,
        "pfc218": 0.95,
        "pfc318": 0.99,
        "sf6": 0.95,
    },
    "NETHERLANDS": {
        "ch4": 0.96,
        "n2o": 0.97,
        "hfc32": 0.97,
        "hfc125": 0.97,
        "hfc134a": 0.97,
        "hfc143a": 0.96,
        "cf4": 0.99,
        "pfc116": 0.99,
        "pfc218": 0.97,
        "pfc318": 0.99,
        "sf6": 0.99,
    },
    "BELGIUM": {
        "ch4": 0.95,
        "n2o": 0.97,
        "hfc32": 0.95,
        "hfc125": 0.95,
        "hfc134a": 0.94,
        "hfc143a": 0.94,
        "cf4": 0.99,
        "pfc116": 0.99,
        "pfc218": 0.94,
        "pfc318": 0.99,
        "sf6": 0.98,
    },
    "BENELUX": {
        "ch4": 0.96,
        "n2o": 0.97,
        "hfc32": 0.98,
        "hfc125": 0.97,
        "hfc134a": 0.97,
        "hfc143a": 0.97,
        "cf4": 0.9925,
        "pfc116": 0.99,
        "pfc218": 0.97,
        "pfc318": 0.99,
        "sf6": 0.99,
    },
    "IRELAND": {
        "ch4": 0.95,
        "n2o": 0.95,
        "hfc32": 0.95,
        "hfc125": 0.95,
        "hfc134a": 0.95,
        "hfc143a": 0.95,
        "cf4": 0.99,
        "pfc116": 0.99,
        "pfc218": 0.95,
        "pfc318": 0.95,
        "sf6": 0.95,
    },
    "HUNGARY": {
        "ch4": 0.99,
        "n2o": 0.99,
        "hfc32": 0.97,
        "hfc125": 0.95,
        "hfc134a": 0.95,
        "hfc143a": 0.95,
        "cf4": 0.99,
        "pfc116": 0.995,
        "pfc218": 0.96,
        "pfc318": 0.965,
        "sf6": 0.98,
    },
    "NORWAY": {
        "ch4": 0.95,
        "n2o": 0.95,
        "hfc32": 0.95,
        "hfc125": 0.95,
        "hfc134a": 0.95,
        "hfc143a": 0.95,
        "cf4": 0.95,
        "pfc116": 0.95,
        "pfc218": 0.95,
        "pfc318": 0.95,
        "sf6": 0.95,
    },
}

class AnnexConfig:
    """
    Class containing all the parameters necessary to make the plots and tables for the annex report.

    Attributes:
        data_dir (str): directory where the data are stored
        monthly_species (list): species with monthly inversions (see documentation of annex_plot_generator.produce_plots to know which plots will be made)
        yearly_species (list): species with yearly inversions (see documentation of annex_plot_generator.produce_plots to know which plots will be made)
        combined_species (list): combined species (e.g. ["all_hfc", "all_pfc"])
        models_monthly_species (list): models used for country flux plots of the monthly species
        models_yearly_species (list): models used for country flux plots of the yearly species
        country_flux_units_print (str): units for the country flux plots
        kwargs_country_flux_general (dict): parameters for all country flux plots
        kwargs_country_flux_monthly_species (dict): parameters for country flux plots of monthly species on extended time windows
        kwargs_country_flux_monthly_species_special (dict): parameters for country flux plots of monthly species on PARIS time windows
        kwargs_country_flux_yearly_species (dict): parameters for country flux plots of yeary species
        models_spatial_maps (list): models used for flux map plots
        flux_units_print (str): units for the flux map plots
        kwargs_maps_general (dict): parameters for all flux map plots
        kwargs_maps_mean (dict): parameters for flux map of total posterior
        kwargs_maps_seasonnal (dict): parameters for flux map of posterior seasonal diff to mean
        fluxlim_percentile (dict): flux limits to use in flux map plots for every species of the selected region
        start_date_fgases (dict): start date to use for every species of the selected region
    """

    ### Path to results directory
    data_dir = "/project/paris/PARISNID2025/"

    ### Species
    monthly_species = ["ch4", "n2o"]

    yearly_species = [
        "hfc23",
        "hfc32",
        "hfc125",
        "hfc134a",
        "hfc143a",
        "hfc152a",
        "hfc227ea",
        "hfc245fa",
        "hfc365mfc",
        "hfc4310mee",
        "cf4",
        "pfc116",
        "pfc218",
        "pfc318",
        "sf6",
    ]

    combined_species = ["all_hfc", "all_pfc"]

    ### Start/end date
    start_date_monthly_species = "2008-01-01"
    start_date_paris_window = "2018-01-01"
    start_date_spatial_maps = "2018-01-01"
    end_date = "2024-01-01"

    ### Settings for country fluxes
    ## Model definitions (list or dict["<period>": list(), "<species>": list()] if different between species)
    # no RHIME N2O results in NID2025
    logger.warning("Excluding RHIME from N2O country fluxes!")
    models_country_flux = {
        "monthly": [
            "InTEM_longrun",
            "InTEM",
            "ELRIS",
            "RHIME",
        ],
        "yearly": [
            "InTEM_longrun",
            "InTEM",
            "ELRIS",
            "RHIME",
        ],
        "n2o": [
            "InTEM_longrun",
            "InTEM",
            "ELRIS",
        ],
    }

    ## Units for plot
    country_flux_units_print = "Tg CO2-eq yr-1"

    ## Kwargs for plot_country_flux
    # for all
    kwargs_country_flux_general = dict(
        data_dir=data_dir,
        annex_mode=True,
        plot_inventory=True,
        fix_y_axes=False,
        add_prior=True,
        add_prior_unc=False,
        set_global_leg=False,
        country_codes_as_titles=None,
        plot_resample_and_original=False,
        return_res=True,
    )

    # for monthly/yearly species on extended time window
    # add entries for specific species if different from monthly/yearly default
    kwargs_country_flux_species_specific = {
        "monthly": dict(
            plot_separate=[True, False, False, False],
            plot_combined=[False, True, True, True],
            resample="year",
            resample_uncert_correlation=False,
            rolling_mean=False,
        ),
        "yearly": dict(
            plot_separate=[True, False, False, False],
            plot_combined=[False, True, True, True],
            resample=None,
            rolling_mean=[False, True, True, True],
        ),
        "n2o": dict(
            plot_separate=[True, False, False],
            plot_combined=[False, True, True],
            resample="year",
            resample_uncert_correlation=False,
            rolling_mean=False,
        ),
    }

    # for monthly species on PARIS time window
    kwargs_country_flux_monthly_species_special = {
        "monthly": dict(
            plot_separate=[True, False, False, False],
            plot_combined=[False, True, True, True],
            resample=None,
            rolling_mean=False,
        ),
        "n2o": dict(
            plot_separate=[True, False, False],
            plot_combined=[False, True, True],
            resample=None,
            rolling_mean=False,
        ),
    }

    ### Settings for spatial maps
    ## Model definitions (list or dict["<period>": list(), "<species>": list()] if different between species)
    # NOTE: it is assumed that models_spatial_maps exist in models_monthly/yearly_species
    # no RHIME N2O results in NID2025
    logger.warning("Excluding RHIME from N2O spatial maps!")
    models_spatial_maps = {
        "yearly": [
            "InTEM",
            "ELRIS",
            "RHIME",
        ],
        "monthly": [
            "InTEM",
            "ELRIS",
            "RHIME",
        ],
        "n2o": [
            "InTEM",
            "ELRIS",
        ],
    }
    flux_units_print = "kg km-2 yr-1"

    ## Kwargs for flux_map functions
    # for all
    kwargs_maps_general = dict(
        set_fluxlim="auto",
        plot_combined=True,
        add_sites=True,
    )

    # for flux total posterior (all species)
    kwargs_maps_mean = dict(
        var="flux_total_posterior_inversion_grid",
        cmap="viridis",
        c_border="floralwhite",
        chop_by="year",
    )

    # for posterior seasonal diff to mean (monthly species)
    kwargs_maps_seasonnal = dict(
        var="posterior_mean_diff_inversion_grid",
        cmap="coolwarm",
        c_border="dimgrey",
        chop_by="season",
    )

    def __init__(self, region, inventory_years):
        """
        Initialize instance and update attributes that depend of region and inventory years.
        """
        ### Settings for country fluxes
        self.kwargs_country_flux_general["plot_regions"] = region
        self.kwargs_country_flux_general["inventory_years"] = inventory_years

        ### Settings for spatial maps
        self.fluxlim_percentile = fluxlim_percentiles.get(region, dict())

        ### Start dates
        self.start_date = {
            "monthly": self.start_date_monthly_species,
            "yearly": start_date_fgases[region],
        }

        ### Settings for spatial maps
        self.kwargs_maps_general["region"] = region
        self.kwargs_maps_general["add_markers"] = point_markers[region]
