import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.projections import PolarAxes

import mpl_toolkits.axisartist.grid_finder as gf
import mpl_toolkits.axisartist.floating_axes as fa

import numpy as np
from fluxy import config
import pandas as pd
from typing import Literal


class TaylorDiagram:
    """
    Taylor Diagram class for visualizing model performance.
    It compares standard deviation, correlation coefficient, and RMSE between observations and model predictions.
    """

    def __init__(
        self,
        sd_range: tuple[float, float],
        sd_unit: str,
        sd_obs: float = None,
        fig: Figure = None,
        position: tuple[int, int, int] = (1, 1, 1),
        markersize: int = 100,
        normalized: bool = True,
    ):
        """
        Initialize the axis for the Taylor Diagram.

        Args:
            sd_range (tuple):
                Two floats (min, max). It indicates the range for the radial coordinate.
            sd_unit (str):
                Unit for the standard deviation. It is only used for the axis label.
            sd_obs (float):
                Observed standard deviation.
                It makes sense to have it only if all markers share the same observed standard deviation.
            fig (Figure):
                Instance of a matplotlib figure. The axis for the Taylor Diagram will be added as as
                subplot to this figure.
            position (tuple):
                Three integers (nrows, ncols, index). The subplot will take the index position
                on a grid with nrows rows and ncols columns.
            markersize (int):
                Size of the markers.
            normalized (bool):
                If True, the standard deviation is normalized to the observed standard deviation.
                If False, the standard deviation is plotted in absolute units.
        """

        self.sd_obs = sd_obs  # Standard deviation of the reference
        self.markersize = markersize
        self.smin = sd_range[0]  # Minimum standard deviation
        self.smax = sd_range[1]  # Maximum standard deviation

        # Polar transformation for the Taylor diagram
        tr = PolarAxes.PolarTransform()

        # Correlation coefficient labels and positions
        rlocs = np.concatenate(((np.arange(11.0) / 10.0), [0.95, 0.99]))  # Correlation values
        tlocs = np.arccos(rlocs)  # Convert correlations to polar angles
        gl1 = gf.FixedLocator(tlocs)  # Position of ticks
        tf1 = gf.DictFormatter(dict(zip(tlocs, map(str, rlocs))))  # Format tick labels

        # Define the grid helper with axis limits and labels

        gh = fa.GridHelperCurveLinear(
            tr,
            extremes=(0, np.pi / 2, self.smin, self.smax),  # (theta_min, theta_max, r_min, r_max)
            grid_locator1=gl1,
            tick_formatter1=tf1,
        )

        # Create figure and subplot
        if fig is None:
            fig = plt.figure()
        ax = fa.FloatingSubplot(fig, *position, grid_helper=gh)
        fig.add_subplot(ax)

        # Set the label for standard deviation
        label_sd = (
            "Normalized standard deviation [unitless]" if normalized else "Standard deviation [{}]".format(sd_unit)
        )

        # Customize the axes
        dict_axes = {
            "top": {
                "axis_direction": "bottom",
                "major_tick_labels_axis_direction": "top",
                "label_axis_direction": "top",
                "label": "Correlation coefficient",
            },
            "left": {
                "axis_direction": "bottom",
                "major_tick_labels_axis_direction": "bottom",
                "label_axis_direction": "bottom",
                "label": label_sd,
            },
            "right": {
                "axis_direction": "top",
                "major_tick_labels_axis_direction": "left",
                "label_axis_direction": "top",
                "label": label_sd,
            },
        }

        for key, params in dict_axes.items():
            ax.axis[key].set_axis_direction(params["axis_direction"])
            ax.axis[key].toggle(ticklabels=True, label=True)
            ax.axis[key].major_ticklabels.set_axis_direction(params["major_tick_labels_axis_direction"])
            ax.axis[key].label.set_axis_direction(params["label_axis_direction"])
            ax.axis[key].label.set_text(params["label"])

        # Hide bottom axis (not used)
        ax.axis["bottom"].set_visible(False)

        # Draw constant correlation lines
        for angle in tlocs:  # Use tlocs for exact positions of correlation labels
            x = [0, self.smax * np.cos(angle)]  # Line endpoints (x-coordinates)
            y = [0, self.smax * np.sin(angle)]  # Line endpoints (y-coordinates)
            ax.plot(x, y, color="gray", linestyle="--", linewidth=0.5)  # Dashed gray lines

        # Set main axes for plotting
        self._ax = ax  # Main axes
        self.ax = ax.get_aux_axes(tr)  # Polar coordinates

        # Add reference line and RMSE contours
        if self.sd_obs != None:

            # Plot the reference line and point
            l = self.ax.scatter([0], self.sd_obs, color="k", marker="*", s=self.markersize)  # Reference point
            t = np.linspace(0, np.pi / 2)  # Angles for sd_obs contour
            r = np.zeros_like(t) + self.sd_obs
            self.ax.plot(t, r, "k--", label="_")  # Reference sd_obs line

            # Add RMSE countours
            contours = self.add_contours(colors="0.5")
            self.ax.clabel(contours, inline=1, fontsize=10)

    def add_samples(self, sds: list[float], pearsons: list[float], label: str, *args, **kwargs):
        """
        Add markers representing sample points to the Taylor diagram.

        Args:
            sds (list of floats):
                Standard deviations of the sample points
            pearsons (list of floats):
                Pearson's correlation coefficiens of the sample points
            label (str):
                Label for the sample points, used in the legend.
            *args, **kwargs: Additional plotting parameters (e.g., color, marker).
        """

        ms = kwargs.pop("markersize") if "markersize" in kwargs else self.markersize

        lines = self.ax.scatter(
            np.arccos(pearsons), sds, s=ms, zorder=10, label=label, *args, **kwargs
        )  # Plot in polar coordinates

        return lines

    def add_contours(self, levels: int = 10, **kwargs):
        """
        Add centered RMSE contours to the Taylor diagram.

        Args:
            levels (int):
                Number of contour levels
            **kwargs: Additional contour parameters.
        """

        rs, ts = np.meshgrid(
            np.linspace(self.smin, self.smax),
            np.linspace(0, np.pi / 2),
        )
        xs = rs * np.cos(ts)
        ys = rs * np.sin(ts)
        crmse = np.sqrt(self.sd_obs**2 + rs**2 - 2 * self.sd_obs * rs * np.cos(ts))
        contours = self._ax.contour(xs, ys, crmse, levels=levels, **kwargs)
        return contours


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
        df_this_stats = long_stats[long_stats["variable"] == stat].pivot(index="site", columns="model", values="value")

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


def plot_taylor_diagram(
    stats: dict[Literal["prior", "posterior", "prior_above_BC", "posterior_above_BC"], pd.DataFrame],
    model_colors: dict[str, str],
    model_labels: dict[str, str],
    stat_markers: list[str] = ["o"],
    normalize: bool = True,
    plot_type_model: Literal["separate", "together"] = "separate",
    plot_type_stat: Literal["separate", "together"] = "separate",
    include: list[Literal["prior", "posterior", "prior_above_BC", "posterior_above_BC"]] = ["prior", "posterior"],
    sd_range: tuple[float, float] = (0, 2.5),
    sd_unit: str = "ppb",
    check_sites: bool = False,
) -> Figure:
    """
    Plots statistics for all sites, for all models.

    Args:
        stats (dict of pd.DataFrame):
            Dictionary containing statistics for each model and each site.
            The keys should match the entries in the `include` parameter.
        model_colors (dict of str):
            Models and corresponding colours used to plot the markers.
        model_labels (dict of str):
            Models and corresponding labels used to label the markers.
        stat_markers (list of str):
            Marker styles used to plot the stats specified in the `include` parameter, sharing the same index.
            If only one marker is provided, it will be applied to all stats.
        normalize (bool):
            If True, normalizes the data by the standard deviation of the observations.
            If False, plots the absolute data.
        plot_type_model (str):
            If 'separate', plots each model in a separate subplot.
            If 'together', plots all models in the same subplot.
        plot_type_stat (str):
            If 'separate', plots each statistic in a separate subplot.
            If 'together', plots all statistics in the same subplot.
        include (list of str):
            List of statistics to include in the plot.
            Options are 'prior' and 'posterior'.
        sd_range (tuple of float):
            Range for the standard deviation axis, in units provided with `sd_unit` if `normalize` = True.
        sd_unit (str):
            Unit for the standard deviation axis label.
        check_sites (bool):
            If True, color different sites with different colors.
    Returns:
        fig (figure):
            Plot showing each model's fit statistics, for each site.
    """

    # Check that include is a subset of stats keys
    if not set(include).issubset(stats.keys()):
        raise ValueError(f"include {include} must be a subset of stats keys {list(stats.keys())}")
    
    # Ensure stats keys are valid
    type_options = ["prior", "posterior", "prior_above_BC", "posterior_above_BC"]
    for s in stats:
        assert s in type_options, f"'{s}' is not in {type_options}"

    # Get model names
    models = np.unique(stats[include[0]]["model"].to_numpy())

    # Get site names
    sites = np.unique(stats[include[0]]["site"].to_numpy())
    num_colors = len(sites)
    colormap = plt.get_cmap("Spectral", num_colors)  # Using 'hsv' colormap for more distinct colors
    site_colors = {
        s: colormap(i / num_colors) for i, s in enumerate(sites)
    }  # Normalize index for better color distribution

    # Set the number of rows and columns
    NCOLS_MAX = 3
    ncols = nrows = 1

    if plot_type_model != "together" or plot_type_stat != "together":

        if plot_type_model == "separate" and plot_type_stat == "separate":
            nsubplots = len(models) * len(stats)

        elif plot_type_model == "together" and plot_type_stat == "separate":
            nsubplots = len(stats)

        elif plot_type_model == "separate" and plot_type_stat == "together":
            nsubplots = len(models)

        ncols = NCOLS_MAX if nsubplots >= NCOLS_MAX else nsubplots
        nrows = nsubplots // ncols + 1

    # Create the figure
    fig = plt.figure(figsize=(10 * ncols, 8 * nrows), tight_layout=False)

    # Create a dictionary to save positions of diagrams
    dict_diags_pos = {}

    # Initalize common features for legend
    combined_handles = []
    combined_labels = []

    # Loop over statistics (i.e., prior, posterior)
    for i, (s, stat) in enumerate(stats.items()):

        # Skip if stat is not in include
        if s not in include:
            continue

        # Fetch statistical data
        long_stat = pd.melt(stat, id_vars=["model", "site"], value_vars=["pearson", "sd_obs", "sd_sim"])
        fetch_stat = lambda s: long_stat[long_stat["variable"] == s].pivot(
            index="site", columns="model", values="value"
        )
        df_pearson = fetch_stat("pearson")
        df_sds_obs = fetch_stat("sd_obs")
        df_sds_sim = fetch_stat("sd_sim")

        # Loop over models
        for j, m in enumerate(models):

            # Get the position of the subplot depending on the case
            if plot_type_model == "separate" and plot_type_stat == "separate":
                index_pos = i * ncols + j + 1

            elif plot_type_model == "together" and plot_type_stat == "together":
                index_pos = 1

            elif plot_type_model == "together" and plot_type_stat == "separate":
                index_pos = i + 1

            elif plot_type_model == "separate" and plot_type_stat == "together":
                index_pos = j + 1

            # Get the statistics for the model m
            sds_obs = df_sds_obs[m].values
            sds_sim = df_sds_sim[m].values
            pearsons = df_pearson[m].values

            # Normalize the data if needed
            if normalize:
                sds_sim /= sds_obs
                sds_obs /= sds_obs

            # Remove observation reference if multiple sd_obs and normalize = False
            sd_obs = sds_obs[~np.isnan(sds_obs)][0] if (len(sds_obs) == 1 or normalize) else None

            # Create the Taylor diagram using the corresponding class or fetch it
            if index_pos not in dict_diags_pos:
                diag = TaylorDiagram(
                    sd_range=sd_range,
                    sd_unit=sd_unit,
                    sd_obs=sd_obs,
                    fig=fig,
                    position=(nrows, ncols, index_pos),
                    markersize=150,
                    normalized=normalize,
                )

            else:
                diag = dict_diags_pos[index_pos]

            # Define labels, colors and markers
            label = f"{model_labels[m]} - {s}"
            color = model_colors[m][1] if s.split('_')[0] == "prior" else model_colors[m][0]
            edgecolor = "k"
            marker = stat_markers[include.index(s)] if len(stat_markers) > 1 else stat_markers[0]

            # Add samples to the diagram
            if check_sites:

                for (
                    site,
                    sd_sim,
                    pearson,
                ) in zip(sites, sds_sim, pearsons):

                    # Add samples for each site with different colors
                    site_handles = diag.add_samples(
                        [sd_sim], [pearson], c=site_colors[site], edgecolor="k", marker=marker, label=site
                    )

                    # Add all site handles to the combined legend
                    if i == 0 and j == 0:
                        combined_handles += [site_handles]
                        combined_labels += [site]

                    # Add inner circles to indicate model/statistic
                    inner_handles = diag.add_samples(
                        [sd_sim], [pearson], c=color, edgecolor="k", marker="o", label=label, markersize=40
                    )

                # Add only last inner_handle to the combined legend
                combined_handles += [inner_handles]
                combined_labels += [label]

            else:
                handles = diag.add_samples(sds_sim, pearsons, c=color, edgecolor=edgecolor, marker=marker, label=label)

                # Check if diag.add_samples returns artists or not
                if not isinstance(handles, list):
                    handles = [handles]

                # If this is the first iteration, initialize combined handles and labels
                if i == 0 and j == 0:
                    combined_handles = handles.copy()
                    combined_labels = [label] * len(handles)

                # Add the new marker(s) and label(s) to combined legend
                else:
                    combined_handles += handles
                    combined_labels += [label] * len(handles)

            # Save the position of the diagram's subplot
            dict_diags_pos[index_pos] = diag

    # Add the combined legend to the diagram (once, at the end)
    fig.legend(
        combined_handles,
        combined_labels,
        fontsize=10,
        loc="lower left",
        bbox_to_anchor=(0.1, 1.05),
        ncols=7 if check_sites else 2,
    )

    return fig
