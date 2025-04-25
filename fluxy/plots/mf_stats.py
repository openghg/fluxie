import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from fluxy import config
import pandas as pd


def plot_stats_mf(
    stats: pd.DataFrame,
    stats_to_plot: list[str],
    species: str,
    model_colors: dict[str, str],
    model_labels: dict[str, str],
    config_data: dict[dict],
    mf_units_print: str,
    stats_type: str,
    stats_ylim: dict[list] = None,
    start_date: str = None,
    end_date: str = None,
) -> Figure:
    """
    Plots statistics for all sites, for all models.

    Args:
        stats (pandas.DataFrame):
            Statistical measures, for each site and for each model.
        stats_to_plot (list of str):
            Statistical measures to plot.
        species (str):
            Gas species, e.g. 'ch4'.
        model_colors (dict of str):
            Models and corresponding colours used to plot the model.
        model_labels (dict of str):
            Models and corresponding labels used to plot the stats.
        config_data (dict of dict):
            Dictionary with settings read from json file.
            Use json filenames as keys.
        mf_units_print (str):
            Mole fraction units used in plots
        stats_type (str):
            Type of statistics to be plotted. Should be the same as used in call to stats_mf().
        stats_ylim (dict of lists):
            Limits for y-axis of individual statistic plots. Can be given for selected
            statistics only or passed as None for automatic axis range.
        start_date (str) and end_date (str):
            Dates used to title the plot.
    Returns:
        fig (figure):
            Plot showing each model's fit statistics, for each site.
    """

    models = np.unique(stats["model"].to_numpy())
    # make sure model_labels are in the correct order
    model_str = [model_labels[k] for k in models]
    # plot colors from model names
    colors = [model_colors[k][0] for k in models]

    # determine strings used for plot subtitle
    stats_str = stats_type
    mf_str = ""
    if stats_type not in ["prior", "posterior"]:
        mf_str = " above BC"
        stats_str = stats_type.split("_")[0]

    long_stats = pd.melt(stats, id_vars=["model", "site"], value_vars=stats_to_plot)
    nrows = len(stats_to_plot)
    fig, ax = plt.subplots(nrows, 1, figsize=(10, 3 * nrows), tight_layout=True)
    for i, stat in enumerate(stats_to_plot):
        df_this_stats = long_stats[long_stats["variable"] == stat].pivot(
            index="site", columns="model", values="value"
        )

        df_this_stats.plot(
            kind="bar",
            ax=ax[i],
            stacked=False,
            color=colors,
            xlabel="",
            legend=False,
            zorder=3,
        )
        ax[i].grid(zorder=0)
        if stats_ylim is not None:
            if stat in stats_ylim.keys():
                ax[i].set_ylim(stats_ylim[stat][0], stats_ylim[stat][1])

        ylabel = config.stat_labels[stat]
        if stat not in ["pearson", "nrmse", "nn"]:
            ylabel = ylabel + " (" + mf_units_print + ")"
        ax[i].set_ylabel(ylabel)

    leg = ax[0].legend(
        ncol=3,
        borderpad=0.2,
        columnspacing=1.0,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        labels=model_str,
    )

    species_info = config_data["species_info"][species]
    fig.suptitle(
        (
            f'{species_info["species_print"]} {stats_str} model performance versus mole fraction observations{mf_str}'
            f"\n{start_date} to {end_date}"
        )
    )

    return fig
