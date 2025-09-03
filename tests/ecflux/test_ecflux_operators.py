from fluxy.io import read_model_output, read_config_files
from fluxy.operators.sectors import group_sectors
from fluxy.operators.stats import stats_observed_vs_simulated
from fluxy.plots.ec_flux.sectorial_stack import plot_stacked
from fluxy.test_utils import data_dir


test_models = ["EDDY_HARDAU", "EDDY_HARDAU_STORAGE_2LAYERS"]


dss = read_model_output(
    data_dir / "ecflux",
    "eddy_flux",
    "co2",
    test_models,
)

config = read_config_files()


def test_stats():
    stats_observed_vs_simulated(
        dss,
        sim_var="ecflux_prior",
        obs_var="ecflux_observed",
    )


def test_grouping():
    group_sectors(dss, sectors_config=config["sectors"])


def test_plot():

    plot_stacked(
        dss["EDDY_HARDAU_STORAGE_2LAYERS"],
    )
