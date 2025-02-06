import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pprint
import numpy as np
from fluxy.plots.utils import set_min_decimal_points
from fluxy import config

def print_stats(
        stats_all: dict[str, dict],
        stats_to_print: list[str]
) -> None:
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
                stats_all[stat][site][m] = set_min_decimal_points(stats_all[stat][site][m],
                                                                  sig_fig=3,
                                                                  dec_points=2)

    # Print dictionary to screen
    for stat in stats_to_print:
        print(f'\n{config.stat_labels[stat]}:')
        pprint.pprint(stats_all[stat])

    return None

def plot_stats_mf(
        stats_all: dict[str,dict],
        stats_to_plot: list[str],
        species: str,
        model_colors,
        model_labels,
        config_data,
        start_date=None,
        end_date=None
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

    species_info = config_data['species_info'][species]

    # Create figure
    nrows = len(stats_to_plot)
    fig,ax = plt.subplots(nrows,1,figsize=(10,3*nrows),tight_layout=True)

    # Expand axis dimension if 1x1
    if nrows == 1: ax = np.expand_dims(ax, axis=0)
    
    for k,stat in enumerate(stats_to_plot):
        if stat not in stats_all.keys():
            raise KeyError(f'{stat} is not a valid key. Options are: pearson, nrmse, rmse, std.')
        
        for i,site in enumerate(stats_all[stat].keys()):
            for m,model in enumerate(stats_all[stat][site]):
                if i == 0:
                    label = model_labels[model]
                else:
                    label = None

                # Make scatter plot
                index = i+m*0.2
                ax[k].scatter(index,stats_all[stat][site][model],color=model_colors[model][0],marker='x',s=150,label=label)
                    
            x_val.append(i)
            x_label.append(site)
       
    x_lim0 = -0.2
    x_lim1 = index+0.2                
        
    for i,stat in enumerate(stats_to_plot):
        # Make x-axis 
        ax[i].set_xticks(x_val)
        ax[i].set_xticklabels(x_label,rotation=45)
        ax[i].set_xlim(x_lim0,x_lim1)
    
        # Add horizontal line and y-axis label
        y_hline = 0
        if stat == 'pearson':
            ax[i].invert_yaxis()
            y_hline = 1
       
        ax[i].hlines(y_hline,x_lim0,x_lim1,linestyle='dotted',color='grey')       
        ax[i].set_ylabel(config.stat_labels[stat])

    # Add legend to top subplot
    leg = ax[0].legend(ncol=2,borderpad=.2,columnspacing=1.0)
    try:
        for l in leg.legend_handles:
            l.set_linewidth(5.0)
    except:
        for l in leg.legendHandles:
            l.set_linewidth(5.0)

    fig.suptitle((f'{species_info["species_print"]} Modelled mole fraction statistical fit to obs')+
                 f' \n{start_date} to {end_date}')
    
    # Print stats to screen
    print_stats(stats_all.copy(),stats_to_plot)
    
    return fig
