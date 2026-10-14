"""
Plots shown in the gallery page of the documentation.

Each function decorated with `@example` returns a figure, which is rendered by
`docs/build_pages.py`. Examples are shown in the order of this file,
grouped by the current `section`.
"""

from fluxie.operators.mf import compute_mf_difference, stats_mf
from fluxie.operators.stats import stats_observed_vs_simulated
from fluxie.plots.correlation import plot_correlation
from fluxie.plots.ec_flux.sectorial_stack import plot_stacked
from fluxie.plots.flux_map import (
    plot_flux_map,
    plot_flux_map_combined_models_comparison,
    plot_flux_map_model_comparison,
    plot_flux_map_over_time,
    plot_flux_map_period_comparison,
)
from fluxie.plots.flux_timeseries import (
    plot_country_flux,
    plot_country_sector_flux_bar,
)
from fluxie.plots.mf_stats import plot_stats_mf, plot_taylor_diagram
from fluxie.plots.mf_timeseries import (
    plot_sites_list_mf,
    plot_sites_timeseries,
    plot_timeseries,
)
from fluxie.plots.stats import plot_stats
from fluxie.test_utils.examples import (
    load_ecflux,
    load_flux,
    load_mf,
    load_sector_flux,
)

# List of (section, title, plot function, example function)
EXAMPLES = []
section = None


def example(title, plot_function):
    """Register an example producing a plot with `plot_function`."""

    def register(func):
        EXAMPLES.append((section, title, plot_function, func))
        return func

    return register


def load_country_flux():
    return load_flux(country_flux_units_print="Gg yr-1")


def load_flux_map():
    return load_flux(add_sites_to_flux=True, flux_units_print="kg km-2 yr-1")


def mf_timeseries_kwargs(data):
    return dict(
        species=data.species,
        site="MHD",
        model_colors=data.model_colors,
        model_labels=data.model_labels,
        config_data=data.config_data,
        annotate_coords=data.annotate_coords,
    )


section = "Flux timeseries"


@example("Country fluxes", plot_country_flux)
def _():
    data = load_country_flux()
    return plot_country_flux(
        data.ds_all,
        data.species,
        plot_regions=["GERMANY", "UK", "BENELUX", "NW_EU2"],
        config_data=data.config_data,
        model_colors=data.model_colors,
        model_labels=data.model_labels,
        start_date=data.start_date,
        end_date=data.end_date,
    )


@example("Country fluxes, combined models", plot_country_flux)
def _():
    data = load_country_flux()
    return plot_country_flux(
        data.ds_all,
        data.species,
        plot_regions=["GERMANY", "UK"],
        config_data=data.config_data,
        model_colors=data.model_colors,
        model_labels=data.model_labels,
        start_date=data.start_date,
        end_date=data.end_date,
        add_prior=False,
        plot_separate_unc=False,
        plot_combined=True,
        plot_combined_unc=True,
        combined_models_dict={
            "All models": data.models,
            "InTEM + ELRIS": data.models[:2],
        },
    )


@example("Sectoral country fluxes", plot_country_sector_flux_bar)
def _():
    data = load_sector_flux(regions=["UK"])
    return plot_country_sector_flux_bar(
        data.ds_all,
        data.species,
        "UK",
        config_data=data.config_data,
        model_colors=data.model_colors,
        model_labels=data.model_labels,
        plot_inventory_or_prior="prior",
    )


section = "Flux maps"


@example("Prior, posterior and difference", plot_flux_map)
def _():
    data = load_flux_map()
    return plot_flux_map(
        data.ds_all,
        data.species,
        region="EUROPE",
        config_data=data.config_data,
        model_labels=data.model_labels,
        add_sites=True,
        add_markers=["paris", "london"],
    )


@example("Posterior flux at the inversion resolution", plot_flux_map)
def _():
    data = load_flux_map()
    return plot_flux_map(
        data.ds_all,
        data.species,
        region="UK",
        config_data=data.config_data,
        model_labels=data.model_labels,
        only="posterior",
        plot_inversion_grid_flux=True,
    )


@example("Model comparison", plot_flux_map_model_comparison)
def _():
    data = load_flux_map()
    return plot_flux_map_model_comparison(
        data.ds_all,
        var="flux_total_posterior",
        models=[data.models[0], data.models[2]],
        species=data.species,
        region="UK",
        config_data=data.config_data,
        model_labels=data.model_labels,
        add_sites=True,
    )


@example("Flux maps over time", plot_flux_map_over_time)
def _():
    data = load_flux_map()
    return plot_flux_map_over_time(
        data.ds_all,
        var="flux_total_posterior",
        species=data.species,
        region="UK",
        config_data=data.config_data,
        model_labels=data.model_labels,
        chop_by="year",
        dt=2,
        plot_combined=True,
        add_sites=True,
    )


@example("Comparison of two groups of models", plot_flux_map_combined_models_comparison)
def _():
    data = load_flux_map()
    return plot_flux_map_combined_models_comparison(
        data.ds_all,
        group_a_models=data.models[:2],
        group_b_models=data.models[2:],
        var="flux_total_posterior",
        species=data.species,
        region="UK",
        config_data=data.config_data,
        group_a_label="InTEM + ELRIS",
        group_b_label="RHIME",
        add_sites=True,
    )


@example("Comparison of two periods", plot_flux_map_period_comparison)
def _():
    data = load_flux_map()
    return plot_flux_map_period_comparison(
        data.ds_all,
        var="flux_total_posterior",
        species=data.species,
        start_dates=["2018-01-01", "2021-01-01"],
        end_dates=["2020-01-01", "2024-01-01"],
        region="UK",
        config_data=data.config_data,
        model_labels=data.model_labels,
        add_sites=True,
    )


section = "Mole fraction timeseries"


@example("Mole fractions at a site", plot_timeseries)
def _():
    data = load_mf()
    return plot_timeseries(
        data.ds_all,
        include={"mf_observed": None, "mf_posterior": "percentile_mf_posterior"},
        plot_type="separate",
        diff_include=["mf_posterior"],
        **mf_timeseries_kwargs(data),
    )


@example("Mole fractions at a site, all models together", plot_timeseries)
def _():
    data = load_mf()
    return plot_timeseries(
        data.ds_all,
        include={"mf_observed": None, "mf_posterior": "percentile_mf_posterior"},
        plot_type="together",
        diff_include=["mf_posterior"],
        histogram_type="violin",
        **mf_timeseries_kwargs(data),
    )


@example("Monthly mean mole fractions", plot_timeseries)
def _():
    data = load_mf()
    return plot_timeseries(
        data.ds_all,
        include={"mf_observed": None, "mf_posterior": None, "posterior_above_BC": None},
        aggreg_month=True,
        **mf_timeseries_kwargs(data),
    )


@example("Mole fraction difference between two models", plot_timeseries)
def _():
    data = load_mf()
    ds_diff = compute_mf_difference(data.ds_all, data.models[:2])
    return plot_timeseries(
        ds_diff,
        include={"mf_observed": None},
        plot_type="diff",
        **mf_timeseries_kwargs(data),
    )


@example("Mole fractions at all sites", plot_sites_timeseries)
def _():
    data = load_mf(select=False)
    return plot_sites_timeseries(
        data.ds_all,
        "mf_posterior",
        data.species,
        data.start_date,
        data.end_date,
        data.model_colors,
        data.model_labels,
        data.config_data,
        separate_by_height=False,
    )


@example("Sites used by each model", plot_sites_list_mf)
def _():
    return plot_sites_list_mf(load_mf(select=False).ds_all)


section = "Statistics"


@example("Statistics per site", plot_stats_mf)
def _():
    data = load_mf(site=None)
    return plot_stats_mf(
        stats=stats_mf(data.ds_all, stats_type="posterior_above_BC"),
        stats_to_plot=["pearson", "bias", "crmse"],
        species=data.species,
        model_colors=data.model_colors,
        model_labels=data.model_labels,
        config_data=data.config_data,
        mf_units_print="ppt",
        stats_type="posterior_above_BC",
        start_date=data.start_date,
        end_date=data.end_date,
    )


@example("Taylor diagram", plot_taylor_diagram)
def _():
    data = load_mf(site=None)
    include = ["prior", "posterior"]
    return plot_taylor_diagram(
        stats={stat: stats_mf(data.ds_all, stats_type=stat) for stat in include},
        model_colors=data.model_colors,
        model_labels=data.model_labels,
        include=include,
        std_range=[0, 1.5],
        std_unit="ppt",
    )


section = "Eddy covariance fluxes"


@example("Eddy covariance flux timeseries", plot_timeseries)
def _():
    data = load_ecflux()
    return plot_timeseries(
        data.ds_all,
        include={"ecflux_observed": None, "ecflux_prior": None},
        species=data.species,
        site="HARDAU",
        config_data=data.config_data,
        n_bins=100,
        y_lim=(-20, 100),
        histogram_type="violin",
    )


@example("Sectoral fluxes, diurnal cycle", plot_stacked)
def _():
    data = load_ecflux(sectors=True)
    return plot_stacked(
        data.ds_all["EDDY_HARDAU"], sectors_config=data.config_data["sectors"]
    )


@example("Sectoral fluxes, diurnal cycle in summer", plot_stacked)
def _():
    data = load_ecflux(sectors=True)
    return plot_stacked(
        data.ds_all["EDDY_HARDAU"],
        sectors_config=data.config_data["sectors"],
        season="JJA",
        area=True,
    )


@example("Sectoral fluxes by wind direction", plot_stacked)
def _():
    data = load_ecflux(sectors=True)
    return plot_stacked(
        data.ds_all["EDDY_HARDAU"],
        sectors_config=data.config_data["sectors"],
        group_format="wind",
        area=True,
        wind_bins=12,
    )


@example("Correlation between two datasets", plot_correlation)
def _():
    data = load_ecflux()
    return plot_correlation(
        data.ds_all, variable="ecflux_observed_storage", style="density"
    )


@example("Statistics of simulated versus observed fluxes", plot_stats)
def _():
    data = load_ecflux(sectors=True)
    df_stats = stats_observed_vs_simulated(
        data.ds_all, sim_var="ecflux_prior", obs_var="ecflux_observed"
    )
    return plot_stats(
        df_stats,
        species=data.species,
        plot_type="mean_and_std",
        variable="co2 flux [μmol m-2 s-1]",
    )
