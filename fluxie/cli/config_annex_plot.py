import logging
import copy

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
    "POLAND": [
        (21.0122, 52.2297),
        (19.9450, 50.0647),
        (19.4560, 51.7592),
    ],  # Warschau, Krakau, Łódź
    "CZECHIA": [
        "prague",
        (16.6068, 49.1951),
        (18.2625, 49.8209),
    ],  # Prag, Brno, Ostrava
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
    "POLAND": "2018-01-01",
    "CZECHIA": "2018-01-01",
}

# Models plotted as separate lines in the country flux plots (not included in the means),
# only for the listed regions and species (model domain must cover the region)
separate_models_country_flux = {
    "GERMANY": {"ch4": ["ITMS-D"],
                "sf6": ["ITMS-D"]},
    "NETHERLANDS": {"ch4": ["ITMS-D"],
                    "sf6": ["ITMS-D"]},
}

# Models plotted as separate spatial maps (same colour scale as the PARIS mean map).
# The key is the file name suffix, e.g. ch4_posterior_map_GERMANY_ITMS.png
separate_models_spatial_maps = {
    "GERMANY": {"ch4": {"ITMS": ["ITMS-D"]}, "sf6": {"ITMS": ["ITMS-D"]}},
    "NETHERLANDS": {"ch4": {"ITMS": ["ITMS-D"]}, "sf6": {"ITMS": ["ITMS-D"]}},
}

# Species-specific start dates (the later one of this and the region start date is used)
start_date_species = {
    "nf3": "2012-01-01",
}

# Start date of the HFC and PFC tables, if different from start_date_table
start_date_table_fgases = {
    "NORWAY": "2024-01-01",
}

# Lat/Lon limits in spatial maps
map_limits = {
    "UK": [-12, 6, 47, 65],
    "SWITZERLAND": [4, 12, 44.5, 50],
    # NOTE: equal to automatic GERMANY country mask, but needed for proper ITMS comparison
    "GERMANY": [2.5, 15.5, 46, 57],
    "ITALY": [5, 20, 35, 49],
    "NETHERLANDS": "BENELUX",
    "BELGIUM": "BENELUX",
    "BENELUX": "BENELUX",
    "IRELAND": [-12, -2, 50, 58],
    "HUNGARY": [14, 25, 44.5, 50.5],
    "NORWAY": [3, 33, 55, 79],
    "POLAND": [13.5, 24.5, 48.5, 55.5],
    "CZECHIA": [11.5, 19.5, 48, 51.5],
}

# Limits in posterior spatial map color scale (in units flux_units_print)
# default = "auto"
fluxlim = {
    "UK": {
        "pfc116": [0, 0.16],
        "pfc318": [0, 0.1],
        "nf3": [0, 0.04],
    },
    "IRELAND": {
        "cf4": [0, 0.5],
        "pfc218": [0, 0.4],
        "pfc318": [0, 0.1],
        "sf6": [0, 0.4],
        "nf3": [0, 0.04],
    },
    "NETHERLANDS": {
        "ch4": [0, 40000],
        "cf4": [0, 1],
        "pfc218": [0, 0.2],
        "sf6": [0, 1],
    },
    "BELGIUM": {
        "ch4": [0, 40000],
        "cf4": [0, 1],
        "pfc218": [0, 0.2],
        "sf6": [0, 1],
    },
    "BENELUX": {
        "ch4": [0, 40000],
        "cf4": [0, 1],
        "pfc218": [0, 0.2],
        "sf6": [0, 1],
    },
    "ITALY": {
        "hfc23": [0, 1],
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
        "hfc23": 0.98,
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
        kwargs_country_flux_seasonal_cycle (dict): parameters for the mean seasonal cycle plots of monthly species on PARIS time windows
        models_spatial_maps (list): models used for flux map plots
        flux_units_print (str): units for the flux map plots
        kwargs_maps_general (dict): parameters for all flux map plots
        kwargs_maps_mean (dict): parameters for flux map of total posterior
        kwargs_maps_seasonnal (dict): parameters for flux map of posterior seasonal diff to mean
        network_comparisons (list): comparisons of two inversion runs (e.g. different site networks) shown as flux maps
        kwargs_maps_network (dict): parameters for the network comparison flux maps
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
    # start_date_monthly_species = "2008-01-01"
    start_date_monthly_species = "2012-01-01"
    start_date_paris_window = "2018-01-01"
    start_date_spatial_maps = "2020-01-01"
    start_date_network_maps = "2024-01-01"  # not all sites measured before 2024
    start_date_table = start_date_spatial_maps
    end_date = "2026-01-01"

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

    ## Combined means shown in the country flux plots (key = legend label)
    # NOTE: keep "Mean" as label for the PARIS mean (renamed to "PARIS mean" in the tables)
    combined_models_country_flux = {
        "PARIS mean": [
            "InTEM_NAME",
            "InTEM_FLEXPART",
            "ELRIS_NAME",
            "ELRIS_FLEXPART",
            "RHIME_NAME",
            "RHIME_FLEXPART",
        ],
        "NAME mean": ["InTEM_NAME", "ELRIS_NAME", "RHIME_NAME"],
    }
    combined_models_country_flux_sf6 = {
        "PARIS mean": ["InTEM_NAME", "InTEM_FLEXPART", "ELRIS_NAME", "ELRIS_FLEXPART"],
        "NAME mean": ["InTEM_NAME", "ELRIS_NAME"],
    }

    # Colors of the combined means, in the order of combined_models_country_flux
    combined_colors_country_flux = ["black", "dodgerblue"]

    # Fixed colors of models in the country flux plots (otherwise assigned by order of the inversion systems)
    # NOTE: a list of colors is needed (first color used for the line)
    model_colors_country_flux = {"ITMS-D": ["darkorange", "orange"]}

    # Combined means plotted with a thin line (same width as the separate lines, e.g. ITMS-D)
    thin_lines_country_flux = ["NAME mean"]

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
            plot_separate=False,
            plot_combined=True,
            combined_models_dict=combined_models_country_flux,
            resample="year",
            resample_uncert_correlation=False,
            rolling_mean=False,
        ),
        "yearly": dict(
            plot_separate=False,
            plot_combined=True,
            combined_models_dict=combined_models_country_flux,
            resample=None,
            rolling_mean=True,
        ),
        "sf6": dict(
            plot_separate=False,
            plot_combined=True,
            combined_models_dict=combined_models_country_flux_sf6,
            resample=None,
            rolling_mean=True,
        ),
    }

    kwargs_country_flux_monthly_species_special = {
        "monthly": dict(
            plot_separate=False,
            plot_combined=True,
            combined_models_dict=combined_models_country_flux,
            resample=None,
            rolling_mean=False,
        )
    }

    # for mean seasonal cycle of monthly species on PARIS time window (each calendar month averaged over all years)
    # NOTE: the inventory (annual values) is not plotted, and resample/rolling_mean cannot be combined with aggreg_month.
    #       Set to None to skip this plot.
    kwargs_country_flux_seasonal_cycle = {
        "monthly": dict(
            plot_separate=False,
            plot_combined=True,
            combined_models_dict=combined_models_country_flux,
            resample=None,
            rolling_mean=False,
            aggreg_month=True,
            plot_inventory=False,
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
        cmap="cividis",
        c_border="floralwhite",
        chop_by="year",
        marker_color="red",
    )

    # for posterior seasonal diff to mean (monthly species)
    kwargs_maps_seasonnal = dict(
        var="posterior_mean_diff_inversion_grid",
        cmap="coolwarm",
        c_border="dimgrey",
        chop_by="season",
        set_fluxlim="auto",
    )

    # kwargs for plot_flux_map_combined_models_comparison
    kwargs_maps_network = dict(
        var="flux_total_posterior_inversion_grid",
        cmap="cividis",
        c_border="floralwhite",
        add_sites=True,
        marker_color="red",
    )

    ### Settings for network comparison maps
    # Comparison of the mean of two groups of models, e.g. all models with the PARIS network
    # vs. all models with a reduced network.
    # For every entry and species, maps of the mean of group 1, the mean of group 2 and their
    # difference (group 1 - group 2) are made over the spatial maps time window.
    #   name: short name used in the output filename.
    #   species: list of species to compare.
    #   groups: dictionary {<label>: <list of models>} with exactly two entries.
    #           Both lists must have the same length: the models are compared pairwise (1st with 1st, ...).
    #           As for the other plots, the run of each model is defined in models_info.json ("standard_run"),
    #           e.g. "InTEM_NAME-reduced": {"hfc134a": "NAME_EUROPE_EDGAR_PARISNID2027_nonPARISsites"}.
    #           For each species, a pair is only used if both models have an explicit "standard_run"
    #           entry for this species (no fallback to "default") and both files exist.
    network_comparisons = [
        dict(
            name="reduced",
            species=[
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
                "nf3",
            ],
            groups={
                "PARIS Network": [
                    "InTEM_NAME",
                    "InTEM_FLEXPART",
                    "ELRIS_NAME",
                    "ELRIS_FLEXPART",
                    "RHIME_NAME",
                    "RHIME_FLEXPART",
                ],
                "Reduced Network": [
                    "InTEM_NAME-reduced",
                    "InTEM_FLEXPART-reduced",
                    "ELRIS_NAME-reduced",
                    "ELRIS_FLEXPART-reduced",
                    "RHIME_NAME-reduced",
                    "RHIME_FLEXPART-reduced",
                ],
            },
        ),
    ]

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
        self.start_date_table_fgases = start_date_table_fgases.get(
            region, self.start_date_table
        )
        for species, date in start_date_species.items():
            self.start_date[species] = max(date, start_date_fgases[region])

        ### Settings for spatial maps
        self.kwargs_maps_general["region"] = map_limits[region]
        self.kwargs_maps_general["add_markers"] = point_markers[region]
        self.fluxlim = fluxlim.get(region, dict())
        self.fluxlim_percentile = fluxlim_percentiles.get(region, dict())
        self.difflim_percentile = difflim_percentiles.get(region, dict())

        ### Add separate models (e.g. ITMS-D) for this region
        # deepcopy: class attributes must not be changed for the next region
        kwargs_names = [
            "kwargs_country_flux_species_specific",
            "kwargs_country_flux_monthly_species_special",
            "kwargs_country_flux_seasonal_cycle",
        ]
        self.models_country_flux = copy.deepcopy(self.models_country_flux)
        for name in kwargs_names:
            setattr(self, name, copy.deepcopy(getattr(self, name)))

        for species, extra_models in separate_models_country_flux.get(
            region, {}
        ).items():
            period = "monthly" if species in self.monthly_species else "yearly"
            base_models = self.models_country_flux.get(
                species, self.models_country_flux[period]
            )
            self.models_country_flux[species] = base_models + extra_models
            n_base, n_extra = len(base_models), len(extra_models)

            for name in kwargs_names:
                settings = getattr(self, name)
                if not settings:
                    continue
                base_kwargs = settings.get(species, settings.get(period))
                if base_kwargs is None:  # plot type not made for this species (e.g. seasonal cycle for yearly species)
                    continue
                kwargs = dict(base_kwargs)
                rolling_mean = kwargs.get("rolling_mean", False)
                if not isinstance(rolling_mean, list):
                    rolling_mean = [rolling_mean] * n_base
                kwargs["plot_separate"] = [False] * n_base + [True] * n_extra
                kwargs["rolling_mean"] = rolling_mean + [False] * n_extra
                settings[species] = kwargs

        ### Add separate models for the spatial maps (e.g. ITMS-D) for this region
        self.separate_models_spatial_maps = separate_models_spatial_maps.get(region, {})
