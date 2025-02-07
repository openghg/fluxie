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
    plot_spatial_flux_per_timestamp,
)
from fluxy.plots.flux_timeseries import plot_country_flux
from fluxy.plots.mf_timeseries import (
    plot_mf_timeseries,
    plot_sites_timeseries,
)
from fluxy.operators.mf import compute_mf_difference
from fluxy.plots.mf_stats import plot_stats_mf

data_dir = Path(fluxy.__path__[0]).parent / "data" / "tests"

config_data = read_config_files()
annotate_coords = set_print_settings()

specie = "hfc134a"  # options for individual species, or 'all_hfc' or 'all_pfc'
models = ['InTEM_NAME_EDGAR_std','ELRIS_NAME_EDGAR_std','RHIME_NAME_EDGAR_std']
regions = ["GERMANY", "UK", "BENELUX", "NW_EU2"]
period = 'yearly'  # use to override standard inversion periods, must be a list the same length as models, e.g. ['monthly','yearly']
country_flux_units_print = 'Gg yr-1'
start_date = "2018-01-01"  # inclusive. Option to set as list of dates, e.g. ['2018-01-01','2019-01-01'] which is required for total fgases if one model is missing obs for a year
end_date = "2024-01-01"  # not inclusive. Option to set as list of dates, e.g. ['2023-01-01','2022-01-01'] which is required for total fgases if one model is missing obs for a year
get_labels_from_file = False

ds_all_flux_scaled = {}

if "all" in specie:
    ds_all_flux_scaled = read_flux_total_fgases(
        data_dir,
        specie,
        models,
        config_data,
        regions,
        start_date,
        end_date,
        period=period,
    )
else:
    ds_all_flux = read_model_output(
        data_dir, "flux", specie, models, config_data, period=period
    )

    for m in models:
        ds_all_flux_scaled[m] = slice_flux(
            {m: ds_all_flux[m]},
            config_data,
            start_date,
            end_date,
            specie=specie,
            country_flux_units_print=country_flux_units_print,
        )[m]


site = "MHD"
baseline_site = None
mf_units_print = 'ppt'
ds_all_mf = read_model_output(
    data_dir,
    file_type='concentration',
    specie=specie,
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
model_labels = set_model_labels(models,config_data,get_labels_from_file)


plot_area = "UK"
cmap = "viridis"
cmap_diff = "coolwarm"
c_border = "floralwhite"
plot_site_locations = True
plot_point_markers = ["paris", "london"]
season = None
set_fluxlim = "auto"
set_fluxlim_percentile = None
plot_inversion_grid_flux = False

stats_to_plot = ['pearson','nrmse','rmse']

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
        specie,
        regions,
        config_data["species_info"],
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
        "Yapost",
        start_date,
        end_date,
        model_colors,
        model_labels,
    )


def test_obs_modelled_separate():
    fig = plot_mf_timeseries(
        ds_all_mf_sliced,
        specie,
        site,
        model_colors,
        model_labels,
        config_data,
        annotate_coords,
        plot_type='separate',
        include={"Yobs": None,
                 "Yapost": 'qYapost'},
        diff_include=["Yapost"],
        y_lim=None,
    )


def test_obs_modelled_together():

    fig = plot_mf_timeseries(
        ds_all_mf_sliced,
        specie,
        site,
        model_colors,
        model_labels,
        config_data,
        annotate_coords,
        plot_type='together',
        include={"Yapost": 'qYapost'},
        diff_include=["Yapost"],
        y_lim=None,
    )


def test_mole_fraction_diff():

    ds_diff = compute_mf_difference(ds_all_mf_sliced.copy(), models[:2])

    fig = plot_mf_timeseries(
        ds_diff,
        specie,
        site,
        model_colors,
        model_labels,
        config_data,
        annotate_coords,
        plot_type='diff',
        include={'Yobs': None},
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

    stats = stats_mf(ds_all_allsites)

    fig = plot_stats_mf(
        stats,
        stats_to_plot,
        specie,
        model_colors,
        model_labels,
        config_data,
        start_date=start_date,
        end_date=end_date,
    )


def deactivate_test_plot_flux_map():

    fig = plot_flux_map(
        ds_all_flux_scaled,
        specie,
        plot_area,
        config_data,
        cmap=cmap,
        cmap_diff=cmap_diff,
        c_border=c_border,
        period_override=period_override,
        add_sites=plot_site_locations,
        add_markers=plot_point_markers,
        season=season,
        set_fluxlim=set_fluxlim,
        set_fluxlim_percentile=set_fluxlim_percentile,
        plot_inversion_grid_flux=plot_inversion_grid_flux,
    )


def deactivate_test_plot_flux_map_model_comparison():

    var = 'flux_total_posterior'
    model_1 = 'intem_name_edgar'
    model_2 = 'elris_name_edgar'

    fig = plot_flux_map_model_comparison(
        ds_all_flux_scaled,
        var,
        model_1,
        model_2,
        specie,
        plot_area,
        config_data,
        presentation_mode=True,
        cmap=cmap,
        cmap_diff=cmap_diff,
        c_border=c_border,
        period_override=period_override,
        add_sites=plot_site_locations,
        add_markers=plot_point_markers,
        season=season,
        set_fluxlim=set_fluxlim,
        set_fluxlim_percentile=set_fluxlim_percentile,
    )


def deactivate_test_spatial_flux_per_timestamp():

    var = "flux_total_posterior"
    plot_combined = False
    annex_mode = False
    chop_by = "year"
    dt = 1
    fig = plot_spatial_flux_per_timestamp(
        ds_all_flux_scaled,
        specie,
        plot_area,
        end_date,
        config_data["species_info"],
        config_data["models_info"],
        cmap=cmap,
        c_border=c_border,
        var=var,
        plot_combined=plot_combined,
        annex_mode=annex_mode,
        chop_by=chop_by,
        dt=dt,
        period_override=period_override,
        plot_site_locations=plot_site_locations,
        plot_point_markers=plot_point_markers,
        set_fluxlim=set_fluxlim,
        set_fluxlim_percentile=set_fluxlim_percentile,
        plot_inversion_grid_flux=plot_inversion_grid_flux,
    )
