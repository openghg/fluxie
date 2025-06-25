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


class Annex_config:
    def __init__(self, region, inventory_years):
        ### Path to results directory
        self.data_dir = "/project/paris/inverse_modelling/"

        ### Species
        self.monthly_species = []  # "ch4", "n2o"

        self.annual_species = [
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

        self.combined_species = ["all_hfc", "all_pfc"]

        ### Settings for country fluxes
        ## Model definitions
        # for monthly species
        self.models_monthly_species = [
            "InTEM_longrun",
            "InTEM",
            "ELRIS",
            "RHIME",
        ]  # NOTE: only options are basic model names w/ and w/o longrun

        # for annual species
        self.models_yearly_species = [
            "InTEM",
            "ELRIS",
            "RHIME",
        ]

        ## Units for plot
        self.country_flux_units_print = "Tg CO2-eq yr-1"

        ## Kwargs for plot_country_flux
        # for all
        self.kwargs_country_flux_general = dict(
            plot_regions=region,
            inventory_years=inventory_years,
            data_dir=self.data_dir,
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

        # for monthly species on extended time window
        self.kwargs_country_flux_monthly_species = dict(
            plot_separate=[True, False, False, False],
            plot_combined=[False, True, True, True],
            resample=[None, "year", "year", "year"],
            resample_uncert_correlation=False,
            rolling_mean=False,
        )

        # for monthly species on PARIS time window
        self.kwargs_country_flux_monthly_species_special = dict(
            plot_separate=[True, False, False],
            plot_combined=[True, True, True],
            resample=None,
            rolling_mean=False,
        )

        # for yearly species
        self.kwargs_country_flux_yearly_species = dict(
            plot_separate=[True, False, False],
            plot_combined=[True, True, True],
            resample=None,
            rolling_mean=True,
        )

        ### Settings for spatial maps
        self.models_spatial_maps = ["InTEM", "ELRIS", "RHIME"]
        self.flux_units_print = "kg km-2 yr-1"

        self.fluxlim_percentile = fluxlim_percentiles.get(region, dict())

        self.start_date_fgases = start_date_fgases[region]

        ## Kwargs for flux_map functions
        # for all
        self.kwargs_maps_general = dict(
            region=region,
            set_fluxlim="auto",
            plot_combined=True,
            add_sites=True,
            add_markers=point_markers[region],
        )

        # for flux total posterior (all species)
        self.kwargs_maps_mean = dict(
            var="flux_total_posterior_inversion_grid",
            cmap="viridis",
            c_border="floralwhite",
            chop_by="year",
        )

        # for posterior seasonal diff to mean (monthly species)
        self.kwargs_maps_seasonnal = dict(
            var="posterior_mean_diff_inversion_grid",
            cmap="coolwarm",
            c_border="dimgrey",
            chop_by="season",
            dt=[[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        )
