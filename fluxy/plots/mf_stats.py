import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pprint
import numpy as np
from fluxy.plots.utils import set_min_decimal_points
from fluxy import config
import pandas as pd

def print_stats(stats_all: dict[str, dict], stats_to_print: list[str]) -> None:
    """
    Prints statistics to screen.

    Args:
        stats_all (dictionary of dictionaries):
            Statistical measures, for each site and for each model.
        stats_to_print (list of str):
            Statistical measures to print.
    """

    # Round values
    for stat in stats_to_print:
        for site in stats_all[stat].keys():
            for m in stats_all[stat][site]:
                stats_all[stat][site][m] = set_min_decimal_points(
                    stats_all[stat][site][m], sig_fig=3, dec_points=2
                )

    # Print dictionary to screen
    for stat in stats_to_print:
        print(f"\n{config.stat_labels[stat]}:")
        pprint.pprint(stats_all[stat])

    return None


def plot_stats_mf(
    stats_all: dict[str, dict],
    stats_to_plot: list[str],
    species: str,
    model_colors,
    model_labels,
    config_data,
    start_date=None,
    end_date=None,
) -> Figure:
    """
    Plots fit statistics for all sites, for all models.

    Args:
        stats_all (dictionary of dictionaries):
            Statistical measures, for each site and for each model.
        stats_to_plot (list of str):
            Statistical measures to plot.
        species (str):
            Gas species, e.g. 'ch4'.
        model_colors (dict of str):
            Models and corresponding colours used to plot the model.
        config_data (dict of dict):
            Dictionary with settings read from json file.
            Use json filenames as keys.
        start_date (str) and end_date (str):
            Dates used to title the plot.
    Returns:
        fig (figure):
            Plot showing each model's fit statistics, for each site.
    """

    x_val = []
    x_label = []

    species_info = config_data["species_info"][species]

    # Create figure
    nrows = len(stats_to_plot)
    fig, ax = plt.subplots(nrows, 1, figsize=(10, 3 * nrows), tight_layout=True)

    # Expand axis dimension if 1x1
    if nrows == 1:
        ax = np.expand_dims(ax, axis=0)

    for k, stat in enumerate(stats_to_plot):
        if stat not in stats_all.keys():
            raise KeyError(
                f"{stat} is not a valid key. Options are: pearson, nrmse, rmse, std."
            )
        
        for i, site in enumerate(stats_all[stat].keys()):
            for m, model in enumerate(stats_all[stat][site]):
                if i == 0:
                    label = model_labels[model]
                else:
                    label = None
                
                # Make scatter plot
                index = i + m * 0.2
                ax[k].scatter(
                    index,
                    stats_all[stat][site][model],
                    color=model_colors[model][0],
                    marker="x",
                    s=150,
                    label=label,
                )

            x_val.append(i)
            x_label.append(site)

    x_lim0 = -0.2
    x_lim1 = index + 0.2

    for i, stat in enumerate(stats_to_plot):
        # Make x-axis
        ax[i].set_xticks(x_val)
        ax[i].set_xticklabels(x_label, rotation=45)
        ax[i].set_xlim(x_lim0, x_lim1)

        # Add horizontal line and y-axis label
        y_hline = 0
        if stat == "pearson":
            ax[i].invert_yaxis()
            y_hline = 1

        ax[i].hlines(y_hline, x_lim0, x_lim1, linestyle="dotted", color="grey")
        ax[i].set_ylabel(config.stat_labels[stat])

    # Add legend to top subplot
    leg = ax[0].legend(ncol=2, borderpad=0.2, columnspacing=1.0)
    try:
        for l in leg.legend_handles:
            l.set_linewidth(5.0)
    except:
        for l in leg.legendHandles:
            l.set_linewidth(5.0)

    fig.suptitle(
        (
            f'{species_info["species_print"]} Modelled mole fraction statistical fit to obs'
        )
        + f" \n{start_date} to {end_date}"
    )

    # Print stats to screen
#    print_stats(stats_all.copy(), stats_to_plot)

    return fig


def plot_stats_pd_mf(
    stats: pd.DataFrame,
    stats_to_plot: list[str],
    species: str,
    model_colors: dict[str],
    model_labels: dict[str],
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
        stats_all (dictionary of dictionaries):
            Statistical measures, for each site and for each model.
        stats_to_plot (list of str):
            Statistical measures to plot.
        species (str):
            Gas species, e.g. 'ch4'.
        model_colors (dict of str):
            Models and corresponding colours used to plot the model.
        config_data (dict of dict):
            Dictionary with settings read from json file.
            Use json filenames as keys.
        mf_units_print (str):
            Mole fraction units used in plots
        stats_type (str):
            Type of statistics to be plotted. Should be the same as used in call to stats_mf().
        stats_ylim (dict of lists) limits for y-axis of individual statistic plots. Can be given for selected statistics only or passed as None for automatic axis range. 
        start_date (str) and end_date (str):
            Dates used to title the plot.
    Returns:
        fig (figure):
            Plot showing each model's fit statistics, for each site.
    """
    from fluxy import config
    
    import matplotlib.pyplot as plt
    import numpy as np

    models = np.unique(stats['model'].to_numpy())
    # make sure model_labels remain in correct order
    model_labels = [model_labels[k] for k in models]
    # plot colors from model names
    colors = [model_colors[k][0] for k in models]

    # determine strings used for plot subtitle
    if stats_type in ['prior', 'posterior']: 
        mf_str = ""
    else: 
        mf_str = " above BC"
        stats_type = stats_type.split("_")[0]
        
    long_stats = pd.melt(stats, id_vars=['model', 'site'], value_vars=stats_to_plot)
    nrows = len(stats_to_plot)
    fig, ax = plt.subplots(nrows, 1, figsize=(10, 3 * nrows), tight_layout=True)
    for i, stat in enumerate(stats_to_plot): 
        df_this_stats = long_stats[long_stats['variable']==stat].pivot(index='site', columns='model', values='value')

        df_this_stats.plot(kind="bar", 
             ax=ax[i], 
             stacked=False, 
             color=colors,              
             xlabel="", 
             legend=False,
             zorder=3
            )
        ax[i].grid(zorder=0)
        if stats_ylim is not None:
            if stat in stats_ylim.keys():
                ax[i].set_ylim(stats_ylim[stat][0], stats_ylim[stat][1])
                
        ylabel = config.stat_labels[stat]
        if stat not in ['pearson', 'nrmse', 'nn']:            
            ylabel = ylabel+" ("+mf_units_print+")"
        ax[i].set_ylabel(ylabel)
    
    leg = ax[0].legend(ncol=3, borderpad=0.2, columnspacing=1.0, loc="upper center", 
                       bbox_to_anchor=(0.5, 1.25), labels=model_labels)

    species_info = config_data["species_info"][species]
    fig.suptitle(
        (
            f'{species_info["species_print"]} {stats_type} model performance versus mole fraction observations{mf_str}'        
            f"\n{start_date} to {end_date}"
        )
    )
    
    return fig



