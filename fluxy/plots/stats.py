from enum import Enum
import logging
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd


class StatsPlotTypes(Enum):
    MEAN_AND_STD = "mean_and_std"
    BOX_PLOT = "box_plot"


logger = logging.getLogger(__name__)


def plot_stats(
    df_stats: pd.DataFrame,
    species: str = "",
    variable: str = "",
    site_span: float = 0.7,
    plot_type: StatsPlotTypes = StatsPlotTypes.MEAN_AND_STD,
    config_data: dict[str, dict] = {},
):
    """Plot statistics for each site and model.

    The xaxis is the site name, but in each site there are multiple models.

    Args:
        df_stats: DataFrame containing the statistics to plot.
            Can be calculated with the functions
            :py:func:`fluxy.operators.stats.stats_observed_vs_simulated`
        species: Name of the species, used in the plot title.
        variable: Name of the variable, used in the y-axis label.
        site_span: The width of the site span in the plot.
            Should be between 0.1 and 1.0.
        plot_type: Type of the plot to create.
            One of :py:class:`fluxy.operators.stats.StatsPlotTypes`.
        config_data: Configuration data, used to get species information.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    if site_span < 0.1 or site_span > 1:
        raise ValueError("site_span must be between 0.1 and 1.0")

    unique_sites, site_number = np.unique(df_stats["site"], return_inverse=True)
    unique_models, model_number = np.unique(df_stats["model"], return_inverse=True)

    model_offset = site_span / (len(unique_models) + 1)
    x = site_number + model_offset * (model_number + 1)
    sim_obs_offset = model_offset * 0.15

    plot_type = StatsPlotTypes(plot_type)
    match plot_type:
        case StatsPlotTypes.MEAN_AND_STD:
            # Mean values and std deviation
            fmt_kwargs = dict(
                markersize=5,
                capsize=3,
                elinewidth=1,
                markeredgewidth=1,
                linestyle="",
                marker="o",
            )

            ax.errorbar(
                x - sim_obs_offset,
                df_stats["mean_sim"],
                yerr=df_stats["std_sim"],
                label="Simulated",
                **fmt_kwargs,
            )
            ax.errorbar(
                x + sim_obs_offset,
                df_stats["mean_obs"],
                yerr=df_stats["std_obs"],
                label="Observed",
                **fmt_kwargs,
            )
        case StatsPlotTypes.BOX_PLOT:
            # We need to do the box plot manually, use vertical bars
            width = sim_obs_offset * 0.8
            # Horizontal positions for the median bar
            for simobs in ["sim", "obs"]:
                this_x = x + (simobs == "obs") * sim_obs_offset

                medians = df_stats[f"median_{simobs}"]
                q25 = df_stats[f"q25_{simobs}"]
                q75 = df_stats[f"q75_{simobs}"]
                ax.bar(
                    this_x,
                    height=q75 - q25,
                    width=width,
                    bottom=q25,
                    label=f"{simobs.capitalize()} median",
                    alpha=0.5,
                )
                # Add min and max lines
                min_values = df_stats[f"min_{simobs}"]
                max_values = df_stats[f"max_{simobs}"]
                ax.vlines(
                    this_x,
                    min_values,
                    max_values,
                    color="black",
                    linewidth=1,
                    label=f"{simobs.capitalize()} range",
                )
                # horizontal line for median and for the top and bottom of the box
                for ys in [q25, medians, q75]:
                    ax.hlines(
                        ys,
                        this_x - width / 2,
                        this_x + width / 2,
                        color="black",
                        linewidth=1,
                    )

        case _:
            raise ValueError(f"Unknown plot type: {plot_type}")

    species_info = config_data.get("species_info", {}).get(species, {})
    specie_print = species_info.get("species_print", species)
    units = df_stats["unit"].unique()
    if len(units) != 1:
        logger.warning(f"Multiple units found: {units}")
    ax.set_title(f"Statistics for {specie_print}" if specie_print else "Statistics")
    ax.set_ylabel(f"variable [{units[0]}]")
    ax.legend()

    ax.set_xticks(x + sim_obs_offset / 2, np.repeat(unique_models, len(unique_sites)))
    sec_sites = ax.secondary_xaxis(location=0)
    sec_sites.set_xticks(
        np.arange(len(unique_sites)) + site_span / 2, [f"\n\n{s}" for s in unique_sites]
    )
    return fig, ax
