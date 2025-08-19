import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from fluxy.plots.utils import stack_plot


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
    sectors_config: dict[str, str] = {},
):
    """Plot stacked bar chart for sectorial fluxes.

    Args:
        ds: xarray Dataset containing the data to plot.
        variable_simulated: Name of the simulated variable.
            Simulated variable is expected to have a sectorial dimension.
        variable_observed: Name of the observed variable.
            Observed variable is expected to not have sectorial dimension.
        season: Season to filter the data.
        group_format: Format for grouping the data.
        species: Name of the species to plot.
        area: Whether to plot the data as an area chart.
        y_lims: Limits for the y-axis.
        plot_observation_counts: Whether to plot observation counts.
        sectors_config: Configuration for the sectors.

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

    fmt_time = lambda x: x.index.strftime(group_format)

    df_sim.index = fmt_time(df_sim)
    df_sim = df_sim.groupby(df_sim.index).mean()
    serie_obs.index = fmt_time(serie_obs)
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

    fig, ax = plt.subplots(figsize=(12, 6))
    ef_kwargs = {
        "color": "black",
        "marker": "x",
        "label": "Measurements",
    }

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

    ax = stack_plot(
        df_sim,
        ax=ax,
        area=area,
        colors_of_category=sectors_config.get("colors_of_sector", {}),
    )
    ax.scatter(
        df_obs.index,
        df_obs["mean"].values.reshape(-1),
        **ef_kwargs,
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
        kwargs = {"color": "black", "ha": "center", "va": "bottom"}
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
                ha="center",
                va="top",
            )

    handles, labels = ax.get_legend_handles_labels()
    # add the text to the existing legend
    handles, labels = handles[::-1], labels[::-1]
    handles.append(plt.scatter([], [], color="black", marker="$12$"))

    labels.append("Count of valid\ncomparisons")
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))

    ax.set_ylim(y_lims)

    ax.set_ylabel(f"{species} Flux " " [ µmol m$^{-2}$ s$^{-1}$ ]")
    season_str = season if season else ""
    ax.set_title(f"Footprint and measured fluxes {season_str} ")
    # ax.set_xlabel()
    x_labels = {
        "%H": "Hour of the day (UTC)",
        "%m": "Month of the year",
        "%Y_%m": "Year and month",
        "%m_%H": "Month and hour of the day",
        "%H_%M": "Hour and minute of the day",
    }
    x_label = x_labels[group_format]
    ax.set_xlabel(x_label)
    if group_format == "%Y_%m":
        # Rotate the x labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
    # Make sure to save all the figure and also waht is around it
    fig.tight_layout()

    return fig, ax
