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

# Lat/Lon limits in spatial maps
map_limits = {
    "UK": [-12,6,47,65],
    "SWITZERLAND": [4,12,44.5,50],
    "GERMANY": [2.5,15.5,46,57], #NOTE: equal to automatic GERMANY country mask, but needed for proper ITMS comparison
    "ITALY": [5,20,35,49],
    "NETHERLANDS": "BENELUX",
    "BELGIUM": "BENELUX",
    "BENELUX": "BENELUX",
    "IRELAND": [-12,-2,50,58],
    "HUNGARY": [14,25,44.5,50.5],
    "NORWAY": [3,33,55,79],
}

# Limits in posterior spatial map color scale (in units flux_units_print)
# default = "auto"
fluxlim = {
    "UK": {
        "pfc116": [0,0.16],
        "pfc318": [0,0.1],
        "nf3": [0,0.04],
    },
    "IRELAND": {
        "cf4": [0,0.5],
        "pfc218": [0,0.4],
        "pfc318": [0,0.1],
        "sf6": [0,0.4],
        "nf3": [0,0.04],
    },
    "NETHERLANDS": {
        "ch4": [0,40000],
        "cf4": [0,1],
        "pfc218": [0,0.2],
        "sf6": [0,1],
    },
    "BELGIUM": {
        "ch4": [0,40000],
        "cf4": [0,1],
        "pfc218": [0,0.2],
        "sf6": [0,1],
    },
    "BENELUX": {
        "ch4": [0,40000],
        "cf4": [0,1],
        "pfc218": [0,0.2],
        "sf6": [0,1],
    },
    "ITALY": {
        "hfc23": [0,1],
    },
}

# Specify the percentile to use for the color scales in the posterior spatial map
# default = 0.99
fluxlim_percentiles = {
    "UK": {
        "ch4": 0.985,
        "hfc125": 0.995,
        "hfc134a": 0.997,
        "hfc143a": 0.997,
        "hfc32": 0.997,
        "hfc152a": 0.997,
        "hfc227ea": 0.997,
        "hfc236fa": 0.998,
        "hfc245fa": 0.997,
        "hfc4310mee": 0.997,
        "cf4": 0.995,
        "pfc218": 0.999,
        "sf6": 0.998,
    },
    "SWITZERLAND": {
        "n2o": 0.995,
        "hfc125": 0.98,
        "hfc134a": 0.985,
        "hfc32": 0.95,
        "hfc227ea": 0.95,
        "hfc23": 0.95,
        "hfc236fa": 0.98,
        "hfc245fa": 0.95,
        "hfc365mfc": 0.985,      
        "cf4": 0.97,
        "sf6": 0.96,
    },
    "GERMANY": {
        "ch4": 0.97,
        "n2o": 0.985,
        "hfc134a": 0.992,
        "hfc227ea": 0.992,
        "hfc365mfc": 0.98,
        "cf4": 0.995,
        "pfc116": 0.995,
        "sf6": 0.999,
    },
    "ITALY": {
        "ch4": 0.9975,
        "n2o": 0.995,
        "hfc32": 0.995,
        "hfc125": 0.995,
        "hfc134a": 0.995,
        "hfc143a": 0.995,
        "hfc152a": 0.999,
        "hfc227ea": 0.999,
        "hfc236fa": 0.999,
        "hfc245fa": 0.999,
        "hfc365mfc": 0.999,
        "hfc4310mee": 0.999,
        "cf4": 0.996,
        "pfc116": 0.999,
        "pfc218": 0.999,
        "pfc318": 0.999,
        "sf6": 0.99,
        "nf3": 0.999,
    },
    "NETHERLANDS": {
        "n2o": 0.995,
        "hfc125": 0.997,
        "hfc134a": 0.999,
        "hfc143a": 0.995,
        "hfc32": 0.995,
        "hfc152a": 0.995,
        "hfc227ea": 0.995,
        "hfc23": 0.999,
        "hfc236fa": 0.997,
        "hfc245fa": 0.999,
        "hfc365mfc": 0.995,
        "pfc116": 0.995,
        "pfc318": 0.998,
        "nf3": 0.997,
    },
    "BELGIUM": {
        "n2o": 0.995,
        "hfc125": 0.997,
        "hfc134a": 0.999,
        "hfc143a": 0.995,
        "hfc32": 0.995,
        "hfc152a": 0.995,
        "hfc227ea": 0.995,
        "hfc23": 0.999,
        "hfc236fa": 0.997,
        "hfc245fa": 0.999,
        "hfc365mfc": 0.995,
        "pfc116": 0.995,
        "pfc318": 0.998,
        "nf3": 0.997,
    },
    "BENELUX": {
        "n2o": 0.995,
        "hfc125": 0.997,
        "hfc134a": 0.999,
        "hfc143a": 0.995,
        "hfc32": 0.995,
        "hfc152a": 0.995,
        "hfc227ea": 0.995,
        "hfc23": 0.999,
        "hfc236fa": 0.997,
        "hfc245fa": 0.999,
        "hfc365mfc": 0.995,
        "pfc116": 0.995,
        "pfc318": 0.998,
        "nf3": 0.997,
    },
    "IRELAND": {
        "n2o": 0.985,
        "hfc125": 0.96,
        "hfc134a": 0.96,
        "hfc143a": 0.95,
        "hfc32": 0.95,
        "hfc152a": 0.97,
        "hfc227ea": 0.95,
        "hfc236fa": 0.985,
        "hfc245fa": 0.96,
        "hfc365mfc": 0.96,
        "hfc4310mee": 0.98,   
        "pfc116": 0.9999,
    },
    "HUNGARY": {
        "ch4": 0.975,
        "hfc32": 0.97,
        "hfc125": 0.98,
        "hfc134a": 0.95,
        "hfc143a": 0.999,
        "hfc227ea": 0.97,
        "hfc23":0.98,
        "cf4": 0.98,
        "pfc116": 0.995,
        "pfc218": 1,
    },
    "NORWAY": {
        "ch4": 0.9995,
        "n2o": 0.9995,
        "hfc32": 0.999,
        "hfc125": 0.999,
        "hfc134a": 0.999,
        "hfc143a": 0.999,
        "hfc152a": 0.999,
        "hfc227ea": 0.999,
        "hfc236fa": 0.999,
        "hfc245fa": 0.999,
        "hfc365mfc": 0.999,
        "hfc4310mee": 0.999,
        "hfc23": 0.999,
        "cf4": 0.9995,
        "pfc116": 0.9995,
        "pfc218": 0.999,
        "pfc318": 0.9995,
        "sf6": 0.999,
        "nf3": 0.9995,
    },
}

# Specify the percentile to use for the color scales in the seasonal spatial map
# default = 0.99
difflim_percentiles = {
    "UK": {
        "ch4": 0.992,
        "n2o": 0.997,
    },
    "SWITZERLAND": {
        "ch4": 0.997,
        "n2o": 0.999,
    },
    "GERMANY": {
        "n2o": 0.997,
    },
    "ITALY": {
        "ch4": 0.9975,
        "n2o": 0.995,
    },
    "NETHERLANDS": {
        "ch4": 0.998,
        "n2o": 0.999,
    },
    "BELGIUM": {
        "ch4": 0.998,
        "n2o": 0.999,
    },
    "BENELUX": {
        "ch4": 0.998,
        "n2o": 0.999,
    },
    "IRELAND": {
        "ch4": 0.997,
        "n2o": 1,
    },
    "NORWAY": {
        "ch4": 0.9995,
        "n2o": 0.9995,
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
    data_dir = "/project/paris/inverse_modelling/"

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
        "hfc236fa",
        "hfc245fa",
        "hfc365mfc",
        "hfc4310mee",
        "cf4",
        "pfc116",
        "pfc218",
        "pfc318",
        "sf6",
        "nf3",
    ]

    combined_species = ["all_hfc", "all_pfc"]

    ### Start/end date
    start_date_monthly_species = "2008-01-01"
    start_date_paris_window = "2018-01-01"
    start_date_spatial_maps = "2020-01-01"
    start_date_table = start_date_spatial_maps
    end_date = "2025-01-01"

    ### Settings for country fluxes
    ## Model definitions (list or dict["<period>": list(), "<species>": list()] if different between species)
    # NOTE: InTEM_longrun is equal to InTEM_NAME.
    #       The results are read twice because rolling_mean is applied to the single line but not to the combined line.
    models_country_flux = {
        "monthly": [
            "InTEM_longrun",
            "InTEM_NAME",
            "InTEM_FLEXPART",
            "ELRIS_NAME",
            "ELRIS_FLEXPART",
            "RHIME_NAME",
            "RHIME_FLEXPART",
        ],
        "yearly": [
            "InTEM_NAME",
            "InTEM_FLEXPART",
            "ELRIS_NAME",
            "ELRIS_FLEXPART",
            "RHIME_NAME",
            "RHIME_FLEXPART",
        ],
        "sf6": [
            "InTEM_NAME",
            "InTEM_FLEXPART",
            "ELRIS_NAME",
            "ELRIS_FLEXPART",
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
            plot_separate=[True, False, False, False, False, False, False],
            plot_combined=[False, True, True, True, True, True, True],
            resample="year",
            resample_uncert_correlation=False,
            rolling_mean=[True, False, False, False, False, False, False],
        ),
        "yearly": dict(
            plot_separate=[True, False, False, False, False, False],
            plot_combined=True,
            resample=None,
            rolling_mean=True,
        ),
        "sf6": dict(
            plot_separate=[True, False, False, False],
            plot_combined=True,
            resample=None,
            rolling_mean=True,
        ),
    }

    # for monthly species on PARIS time window
    kwargs_country_flux_monthly_species_special = {
        "monthly": dict(
            plot_separate=[True, False, False, False, False, False, False],
            plot_combined=[False, True, True, True, True, True, True],
            resample=None,
            rolling_mean=False,
        )
    }

    ### Settings for spatial maps
    ## Model definitions (list or dict["<period>": list(), "<species>": list()] if different between species)
    # NOTE: in produce_plots, it is assumed that models_spatial_maps exist in models_country_flux
    models_spatial_maps = {
        "monthly": [
            "InTEM_NAME",
            "InTEM_FLEXPART",
            "ELRIS_NAME",
            "ELRIS_FLEXPART",
            "RHIME_NAME",
            "RHIME_FLEXPART",
        ],
        "yearly": [
            "InTEM_NAME",
            "InTEM_FLEXPART",
            "ELRIS_NAME",
            "ELRIS_FLEXPART",
            "RHIME_NAME",
            "RHIME_FLEXPART",
        ],
        "sf6": [
            "InTEM_NAME",
            "InTEM_FLEXPART",
            "ELRIS_NAME",
            "ELRIS_FLEXPART",
        ],
    }
    flux_units_print = "kg km-2 yr-1"

    ## Kwargs for flux_map functions
    # for all
    kwargs_maps_general = dict(
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
        set_fluxlim="auto",
    )

    def __init__(self, region, inventory_years):
        """
        Initialize instance and update attributes that depend of region and inventory years.
        """
        ### Settings for country fluxes
        self.kwargs_country_flux_general["plot_regions"] = region
        self.kwargs_country_flux_general["inventory_years"] = inventory_years

        ### Start dates
        self.start_date = {
            "monthly": self.start_date_monthly_species,
            "yearly": start_date_fgases[region],
        }

        ### Settings for spatial maps
        self.kwargs_maps_general["region"] = map_limits[region]
        self.kwargs_maps_general["add_markers"] = point_markers[region]
        self.fluxlim = fluxlim.get(region, dict())
        self.fluxlim_percentile = fluxlim_percentiles.get(region, dict())
        self.difflim_percentile = difflim_percentiles.get(region, dict())
