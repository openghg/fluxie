from pathlib import Path
import fluxy
from fluxy.config import set_print_settings
from fluxy.config import set_model_colors
from fluxy.config import set_model_labels
from fluxy.io import read_config_files, read_model_output, read_flux_total_fgases
from fluxy.operators.mf import stats_mf
from fluxy.operators.select import slice_flux, slice_mf
from fluxy.plots.flux_map import (
    plot_flux_map,
    plot_flux_map_model_comparison,
    plot_flux_map_over_time,
)
from fluxy.plots.flux_timeseries import plot_country_flux
from fluxy.plots.mf_timeseries import (
    plot_mf_timeseries,
    plot_sites_timeseries,
)
from fluxy.operators.mf import compute_mf_difference
from fluxy.plots.mf_stats import plot_stats_mf, plot_taylor_diagram

data_dir = Path(fluxy.__path__[0]).parent / "data" / "tests"

config_data = read_config_files()
annotate_coords = set_print_settings()

species = "hfc134a"  # options for individual species, or 'all_hfc' or 'all_pfc'
models = [
    "InTEM_NAME_EUROPE_EDGAR_old_format",
    "ELRIS_NAME_EUROPE_EDGAR_old_format",
    "RHIME_NAME_EUROPE_EDGAR_old_format",
]
regions = ["GERMANY", "UK", "BENELUX", "NW_EU2"]
period = "yearly"  # use to override standard inversion periods, must be a list the same length as models, e.g. ['monthly','yearly']
country_flux_units_print = "Gg yr-1"
start_date = "2018-01-01"  # inclusive. Option to set as list of dates, e.g. ['2018-01-01','2019-01-01'] which is required for total fgases if one model is missing obs for a year
end_date = "2024-01-01"  # not inclusive. Option to set as list of dates, e.g. ['2023-01-01','2022-01-01'] which is required for total fgases if one model is missing obs for a year
get_labels_from_file = False

ds_all_flux_scaled = {}

if "all" in species:
    ds_all_flux_scaled = read_flux_total_fgases(
        data_dir,
        species,
        models,
        config_data,
        regions,
        start_date,
        end_date,
        period=period,
    )
else:
    ds_all_flux = read_model_output(
        data_dir, "flux", species, models, config_data, period=period
    )

    for m in models:
        ds_all_flux_scaled[m] = slice_flux(
            {m: ds_all_flux[m]},
            config_data,
            start_date,
            end_date,
            species=species,
            country_flux_units_print=country_flux_units_print,
        )[m]


site = "MHD"
baseline_site = None
mf_units_print = "ppt"
ds_all_mf = read_model_output(
    data_dir,
    file_type="concentration",
    species=species,
    models=models,
    config_data=config_data,
    period=period,
)

ds_all_mf_sliced = slice_mf(
    ds_all_mf.copy(),
    start_date,
    end_date,
    site,
    baseline_site=baseline_site,
    data_dir=data_dir,
    mf_units_print=mf_units_print,
)

model_colors = set_model_colors(models)
model_labels = set_model_labels(models, config_data, get_labels_from_file)


region = "UK"
cmap = "viridis"
cmap_diff = "coolwarm"
c_border = "floralwhite"
add_sites = True
add_markers = ["paris", "london"]
season = None
set_fluxlim = "auto"
set_fluxlim_percentile = None
plot_inversion_grid_flux = False

stats_to_plot = ["pearson", "bias", "crmse"]
what_to_compare = "posterior_above_BC"
taylor_stats2include = ["prior", "posterior"]
stats_ylim = {"pearson": [0, 1], "bias": [-1.5, 0.5], "crmse": [0, 1.5]}


def test_flux_timeseries():
    plot_inventory = False
    inventory_years = None
    fix_y_axes = False
    add_prior = True
    add_prior_unc = False
    set_global_leg = True
    country_codes_as_titles = False
    plot_separate = True
    plot_combined = False
    resample = None
    resample_uncert_correlation = False
    plot_resample_and_original = False
    annex_mode = False
    rolling_mean = False
    
    fig = plot_country_flux(
        ds_all_flux_scaled,
        species,
        regions,
        config_data,
        model_colors,
        model_labels,
        start_date,
        end_date,
        annex_mode,
        plot_inventory,
        inventory_years,
        data_dir,
        fix_y_axes,
        add_prior,
        add_prior_unc,
        set_global_leg,
        country_codes_as_titles=country_codes_as_titles,
        plot_separate=plot_separate,
        plot_combined=plot_combined,
        resample=resample,
        resample_uncert_correlation=resample_uncert_correlation,
        plot_resample_and_original=plot_resample_and_original,
        rolling_mean=rolling_mean,
    )


def test_mf_timeseries():

    fig = plot_sites_timeseries(
        ds_all_mf,
        "mf_posterior",
        start_date,
        end_date,
        model_colors,
        model_labels,
    )


def test_obs_modelled_separate():
    fig = plot_mf_timeseries(
        ds_all_mf_sliced,
        species,
        site,
        model_colors,
        model_labels,
        config_data,
        annotate_coords,
        plot_type="separate",
        include={"mf_observed": None, "mf_posterior": "percentile_mf_posterior"},
        diff_include=["mf_posterior"],
        y_lim=None,
    )


def test_obs_modelled_together():

    fig = plot_mf_timeseries(
        ds_all_mf_sliced,
        species,
        site,
        model_colors,
        model_labels,
        config_data,
        annotate_coords,
        plot_type="together",
        include={"mf_posterior": "percentile_mf_posterior"},
        diff_include=["mf_posterior"],
        y_lim=None,
    )


def test_mole_fraction_diff():

    ds_diff = compute_mf_difference(ds_all_mf_sliced.copy(), models[:2])

    fig = plot_mf_timeseries(
        ds_diff,
        species,
        site,
        model_colors,
        model_labels,
        config_data,
        annotate_coords,
        plot_type="diff",
        include={"mf_observed": None},
        diff_include=None,
        y_lim=None,
    )


def test_plot_stats():
    ds_all_allsites = slice_mf(
        ds_all_mf.copy(),
        start_date,
        end_date,
        site=None,
        baseline_site=baseline_site,
        data_dir=data_dir,
        mf_units_print=mf_units_print,
    )

    stats = stats_mf(ds_all_allsites, stats_type=what_to_compare)

    fig = plot_stats_mf(
        stats=stats,
        stats_to_plot=stats_to_plot,
        species=species,
        model_colors=model_colors,
        model_labels=model_labels,
        config_data=config_data,
        mf_units_print=mf_units_print,
        stats_type=what_to_compare,
        stats_ylim=stats_ylim,
        start_date=start_date,
        end_date=end_date,
    )


def test_plot_taylor_diagram():
    ds_all_allsites = slice_mf(
        ds_all_mf.copy(),
        start_date,
        end_date,
        site=None,
        baseline_site=baseline_site,
        data_dir=data_dir,
        mf_units_print=mf_units_print,
    )

    stats = {}
    for stat in taylor_stats2include:
        stats[stat] = stats_mf(ds_all_allsites, stats_type=stat)

    fig = plot_taylor_diagram(
        stats=stats,
        model_colors=model_colors,
        model_labels=model_labels,
        include=taylor_stats2include
    )


def test_plot_flux_map():

    fig = plot_flux_map(
        ds_all=ds_all_flux_scaled,
        species=species,
        region=region,
        config_data=config_data,
        model_labels=model_labels,
        cmap=cmap,
        cmap_diff=cmap_diff,
        c_border=c_border,
        add_sites=add_sites,
        add_markers=add_markers,
        season=season,
        set_fluxlim=set_fluxlim,
        set_fluxlim_percentile=set_fluxlim_percentile,
        plot_inversion_grid_flux=plot_inversion_grid_flux,
    )


def test_plot_flux_map_model_comparison():

    var = "flux_total_posterior"
    models_comparison = [models[0], models[2]]

    fig = plot_flux_map_model_comparison(
        ds_all=ds_all_flux_scaled,
        var=var,
        models=models_comparison,
        species=species,
        region=region,
        config_data=config_data,
        model_labels=model_labels,
        cmap=cmap,
        cmap_diff=cmap_diff,
        c_border=c_border,
        add_sites=add_sites,
        add_markers=add_markers,
        season=season,
        set_fluxlim=set_fluxlim,
        set_fluxlim_percentile=set_fluxlim_percentile,
    )


def test_plot_flux_map_over_time():

    var = "flux_total_posterior"
    plot_combined = True
    chop_by = "year"
    dt = 2

    fig = plot_flux_map_over_time(
        ds_all=ds_all_flux_scaled,
        var=var,
        species=species,
        region=region,
        config_data=config_data,
        model_labels=model_labels,
        chop_by=chop_by,
        dt=dt,
        plot_combined=plot_combined,
        cmap=cmap,
        cmap_diff=cmap_diff,
        c_border=c_border,
        add_sites=add_sites,
        add_markers=add_markers,
        set_fluxlim=set_fluxlim,
        set_fluxlim_percentile=set_fluxlim_percentile,
    )
