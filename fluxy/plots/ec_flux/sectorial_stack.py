import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fluxy.plots.utils import stack_plot

WIND_LABELS = {
    4: ["N", "E", "S", "W"],
    8: ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
    12: ["N", "NNE", "ENE", "E", "ESE", "SSE", "S", "SSW", "WSW", "W", "WNW", "NNW"],
}


def _get_wind_bins_and_labels(n_bins: int) -> tuple[np.ndarray, list[str]]:
    """Get wind bins and labels for a given number of bins.
    
    Args:
        n_bins: Number of wind bins.
    Returns:
        A tuple containing:
            - bins: Array of wind bin edges in radians.
                Bins start and end at 0 and 2pi (duplicated bin for the start and end)
            - labels: List of wind bin labels.
    """
    if n_bins in WIND_LABELS:
        labels = WIND_LABELS[n_bins]
    else:
        raise ValueError(
            f"Number of wind bins {n_bins} not supported. "
            f"Supported values are {list(WIND_LABELS.keys())}."
        )
    bins = np.deg2rad(np.linspace(0, 360, n_bins + 1))

    return bins, labels


def raise_var_missing(var_name, ds):
    if var_name in ds:
        return
    raise ValueError(
        f"Variable {var_name} not found in the dataset. "
        "Please check the variable name."
    )


def raise_var_dims(var_name, ds, expected_dims):
    if ds[var_name].dims != expected_dims:
        raise ValueError(
            f"Variable {var_name} must have dimensions {expected_dims}. "
            f"Current dimensions: {ds[var_name].dims}"
        )


def plot_stacked(
    ds: xr.Dataset,
    variable_simulated: str = "ecflux_sectorial_prior",
    variable_observed: str = "ecflux_observed",
    season: str = None,
    group_format: str = "%H",
    species: str = " ",
    area: bool = False,
    y_lims: tuple[float, float] = (None, None),
    plot_observation_counts: bool = False,
    sectors_config: dict[str, str] = None,
    errorbar_kwargs: dict = None,
    wind_bins: int = 8,
):
    if sectors_config is None:
        sectors_config = {}
    if errorbar_kwargs is None:
        errorbar_kwargs = {
            "color": "black",
            "marker": "x",
            "label": "Measurements",
            "linestyle": "None",
            "alpha": 0.7,
        }
    """Plot stacked bar chart for sectorial fluxes.

    Args:
        ds: xarray Dataset containing the data to plot.
        variable_simulated: Name of the simulated variable.
            Simulated variable is expected to have a sectorial dimension.
        variable_observed: Name of the observed variable.
            Observed variable is expected to not have sectorial dimension.
        season: Season to filter the data.
        group_format: Format for grouping the data.
            Typical values are:
                * "%X" where x is a time identifer
                    ex: "%H" for hour of the day
                * "wind_direction" to group by wind direction sectors
                    This will require a "wind_direction" variable in the dataset.
                    The output will be a wind rose like plot.

        species: Name of the species to plot.
        area: Whether to plot the data as an area chart.
        y_lims: Limits for the y-axis.
        plot_observation_counts: Whether to plot observation counts.
        sectors_config: Configuration for the sectors.
        wind_bins: Number of wind bins to use if group_format is "wind_direction".

    """

    # Check that the variable is on the sector and index dimensions

    for var in [variable_simulated, variable_observed]:
        raise_var_missing(var, ds)
    raise_var_dims(variable_simulated, ds, ("sector", "index"))
    raise_var_dims(variable_observed, ds, ("index",))

    if season is not None:
        ds = ds.sel(index=ds["time"].dt.season == season)

    df_sim = (
        ds[variable_simulated]
        .swap_dims({"index": "time"})
        .drop(["index", "number_of_identifier"])
        .transpose("time", "sector")
        .to_pandas()
    )
    serie_obs = ds[variable_observed].swap_dims({"index": "time"}).to_series()

    if group_format.startswith("wind"):
        if "wind_direction" not in ds:
            raise ValueError(
                "Variable 'wind_direction' not found in the dataset. "
                "Please check the variable name or `group_format` to another value."
            )

        wind_bins, wind_labels = _get_wind_bins_and_labels(wind_bins)

        fmt_index = lambda x: pd.cut(
            np.deg2rad(x),
            bins=wind_bins,
            labels=wind_bins[:-1] + (wind_bins[1] - wind_bins[0]) / 2,
            include_lowest=True,
        )
        wind_plot = True
        wind_dir = ds["wind_direction"].to_series().values
        df_sim.index = wind_dir
        serie_obs.index = wind_dir
    else:
        fmt_index = lambda x: x.strftime(group_format)
        wind_plot = False

    df_sim.index = fmt_index(df_sim.index)
    df_sim = df_sim.groupby(df_sim.index).mean()

    serie_obs.index = fmt_index(serie_obs.index)
    serie_obs_groupped = serie_obs.groupby(serie_obs.index)
    serie_obs = serie_obs.groupby(serie_obs.index).mean()
    counts = serie_obs.groupby(serie_obs.index).count()

    # Rename the months
    # df_to_plot.index = pd.to_datetime(df_to_plot.index, format="%m").strftime("%b")

    # Calculate the mean and std of the measurements
    df_obs = pd.concat(
        {
            "mean": serie_obs_groupped.mean(),
            "std": serie_obs_groupped.std(),
            "count": serie_obs_groupped.count(),
        },
        axis=1,
    )

    subplots_kwargs = {
        "figsize": (12, 6),
    }
    if wind_plot:
        subplots_kwargs["subplot_kw"] = {"projection": "polar"}

    fig, ax = plt.subplots(**subplots_kwargs)

    if "sector_ordering" in sectors_config:
        sector_order = [
            # Set to presciribed order
            sector
            for sector in sectors_config["sector_ordering"]
            if sector in df_sim.columns
        ] + [
            # Sectors not specified in the ordering
            sector
            for sector in df_sim.columns
            if sector not in sectors_config["sector_ordering"]
        ]
        df_sim = df_sim[sector_order]

    if wind_plot and area:
        # Add extra row, to close the circle
        old_index = df_sim.index
        df_sim = pd.concat([df_sim, df_sim.iloc[[0]]])
        df_sim.index = np.append(old_index, 2 * np.pi + old_index[0])

    ax = stack_plot(
        df_sim,
        ax=ax,
        area=area,
        colors_of_category=sectors_config.get("colors_of_sector", {}),
        width=0.8 if not wind_plot else (wind_bins[1] - wind_bins[0]),
    )

    if "yerr" not in errorbar_kwargs:
        errorbar_kwargs = errorbar_kwargs.copy()
        yerr = df_obs["std"].values.reshape(-1)
        yerr[np.isnan(yerr)] = 0.0
        errorbar_kwargs["yerr"] = np.array(yerr)
    ax.errorbar(
        df_obs.index,
        df_obs["mean"].values.reshape(-1),
        **errorbar_kwargs,
    )
    # scatter the total simulated
    ax.scatter(
        df_sim.index,
        df_sim.sum(axis=1),
        color="black",
        marker="*",
        label="Total simulated fluxes",
    )

    offset = df_obs["mean"].max() * 0.03
    for _, row in df_obs.iterrows():
        kwargs = {"color": "black", "ha": "left", "va": "bottom"}
        if wind_plot:
            # Plot away from the center
            angle = float(row.name)
            # Correct the angle for meteorological convention
            angle = np.pi / 2 - angle
            kwargs["va"] = "top" if np.sin(angle) < 0 else "bottom"
            kwargs["ha"] = "right" if np.cos(angle) < 0 else "left"
        ax.text(
            row.name,
            row["mean"] + offset,
            f"{int(row['count'])}",
            **kwargs,
        )

        if plot_observation_counts:
            ax.text(
                row.name,
                row["mean"].value - offset,
                f"{int(counts[_])}",
                **kwargs,
            )

    handles, labels = ax.get_legend_handles_labels()
    # add the text to the existing legend
    handles, labels = handles[::-1], labels[::-1]
    handles.append(plt.scatter([], [], color="black", marker="$12$"))

    labels.append("Count of valid\ncomparisons")
    x_offset = 1.0 if not wind_plot else 1.06
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(x_offset, 0.5))

    # if wind_plot:
    #     y_lims = (0, y_lims[1])  # No negative values in wind rose
    ax.set_ylim(y_lims)

    season_str = season if season else ""
    title = f"Footprint and measured fluxes {season_str} "
    x_labels = {
        "%H": "Hour of the day (UTC)",
        "%m": "Month of the year",
        "%Y_%m": "Year and month",
        "%m_%H": "Month and hour of the day",
        "%H_%M": "Hour and minute of the day",
    }
    x_label = "Wind direction" if wind_plot else x_labels[group_format]
    y_label = f"{species} Flux " " [ µmol m$^{-2}$ s$^{-1}$ ]"
    if not wind_plot:
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
    else:
        title += f" - {y_label}"
    if group_format == "%Y_%m":
        # Rotate the x labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
    if wind_plot:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(wind_bins[:-1])
        ax.set_xticklabels(wind_labels)
        ax.set_rlabel_position(5)
        # Keep only the tick before the last one
        y_labels = ax.get_yticklabels()
        y_labels = ["" if i != len(y_labels) - 1 else l for i, l in enumerate(y_labels)]
        ax.set_yticklabels(y_labels)

    ax.set_title(title)
    # Make sure to save all the figure and also what is around it
    fig.tight_layout()

    return fig, ax
