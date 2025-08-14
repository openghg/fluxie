"""Correlation plot for Fluxy."""

import logging
from typing import Literal
import xarray as xr
import numpy as np

from fluxy import config
from fluxy.operators.select import slice_site
from fluxy.types import VariableType
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

LimsTuple = tuple[float, float]


def plot_correlation(
    ds_all: dict[str, xr.Dataset],
    variable: str | tuple[str, str],
    species: str | None = None,
    site: str | None = None,
    models_to_plot: list[str] | None = None,
    oppose: Literal["variables", "models"] = "models",
    style: Literal["scatter", "density"] = "scatter",
    model_colors: dict[str, str] | None = None,
    linear_fit: bool = True,
    model_labels: dict[str, dict] = {},
    config_data: dict[str, dict] = {},
    presentation_mode: bool = False,
    scatter_size: float = 1.5,
    lims: LimsTuple | tuple[LimsTuple, LimsTuple] | None = None,
    figsize: tuple[float, float] = (10, 10),
    cmap: str = "YlOrRd",
):
    """Plot correlation between two timeseries.

    Args:
        ds_all (dict): Dictionary of xarray datasets for each model.
        variable (str | tuple):
            Variable(s) to plot. If `oppose='variables'`,
            2 variables must be provided as a tuple.
        species (str, optional): Species to plot. Defaults to None.
        site (str, optional): Site to plot. Defaults to None.
        models_to_plot (list[str], optional): Models to plot. Defaults to None.
            If not given, will plot all models.
        oppose (Literal['variables', 'models'], optional): Whether to oppose variables or models. Defaults to 'models'.
        style (Literal['scatter', 'density'], optional): Plotting style.
            If 'scatter', a scatter (points) plot is created.
            If 'density', a density (colorscale) plot is created.
        model_colors (dict, optional): Dictionary of model colors. Defaults to None.
        model_labels (dict, optional): Dictionary of model labels. Defaults to {}.
        config_data (dict, optional): Configuration data. Defaults to {}.
        presentation_mode (bool, optional): Whether to use presentation mode. Defaults to False.
        linear_fit (bool, optional): Whether to plot a linear fit line. Defaults to True.
        scatter_size (float, optional): Size of scatter points. Defaults to 1.5.
        lims (LimsTuple | tuple[LimsTuple, LimsTuple] | None, optional):
            Limits for the x and y axes. If None, defaults to matplotlib limits.
            If a single tuple is provided, it is used for both axes.
            If two tuples are provided, they are used for x and y axes respectively.
        figsize (tuple[float, float], optional):
            Size of the figure. Defaults to (10, 10).
        cmap (str, optional): Colormap to use for the density plot. Defaults to "YlOrRd".

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """

    models = models_to_plot or list(ds_all.keys())

    # Checking consistency of input data
    if oppose == "variables":
        if not isinstance(variable, (list, tuple)) or len(variable) != 2:
            raise ValueError(
                "When oppose='variables', variable must be a list or tuple of two variable names."
            )
        if len(models) != 1:
            raise ValueError("When oppose='variables', only one model can be provided.")
    elif oppose == "models":
        if isinstance(variable, (list, tuple)) and len(variable) == 1:
            variable = variable[0]
        if not isinstance(variable, str):
            raise ValueError(
                "When oppose='models', variable must be a single variable name."
            )
        if len(models) != 2:
            raise ValueError(
                "When oppose='models', exactly two models must be provided."
            )

        # Check common time axis
        ds_x, ds_y = ds_all[models[0]], ds_all[models[1]]
        if site is not None:
            ds_x = slice_site(ds_x, site)
            ds_y = slice_site(ds_y, site)
        else:
            is_single_platform = lambda ds: len(ds["platform"]) == 1
            if not (is_single_platform(ds_x) and is_single_platform(ds_y)):
                raise ValueError(
                    "When oppose='models', both datasets must have a single platform (site)."
                )

        make_da = lambda ds: xr.DataArray(
            ds[variable].values,
            dims=["time"],
            coords={"time": ds["time"].dt.round("15min").values},
        )
        da_x, da_y = make_da(ds_x).dropna("time"), make_da(ds_y).dropna("time")
        da_x, da_y = xr.align(da_x, da_y, join="inner")
    else:
        raise ValueError("Oppose must be either 'variables' or 'models'.")

    # Accessing the variables to plot
    if oppose == "variables":
        ds = ds_all[models[0]]
        x, y = ds[variable[0]].values, ds[variable[1]].values
        # Remove the nans
        mask = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[mask], y[mask]
        x_label, y_label = variable[0], variable[1]
    elif oppose == "models":
        x, y = da_x.values, da_y.values
        x_label, y_label = models[0], models[1]
    else:
        raise ValueError("Oppose must be either 'variables' or 'models'.")

    fig, ax = plt.subplots(figsize=figsize)

    x_lims: LimsTuple | None = None
    y_lims: LimsTuple | None = None
    match lims, oppose:
        case None, "variables":
            # Use default matplotlib limits
            x_lims, y_lims = None, None
        case None, "models":
            # Use the same x and y limits
            min_xy, max_xy = (
                min(min(x), min(y)),
                max(max(x), max(y)),
            )
            lims_xy = (
                min_xy - 0.04 * (max_xy - min_xy),
                max_xy + 0.04 * (max_xy - min_xy),
            )
            x_lims, y_lims = lims_xy, lims_xy
        case ((x_min, x_max), (y_min, y_max)), _:
            # Use the provided limits
            x_lims, y_lims = lims[0], lims[1]
        case (lim_min, lim_max), _:
            # Use the provided limits for both x and y
            x_lims, y_lims = lims, lims
        case _, _:
            raise ValueError(
                "lims must be a tuple of two tuples or a single tuple."
                f" Got {lims=} instead."
            )

    if style == "scatter":
        ax.scatter(
            x,
            y,
            label=f"{x_label} vs {y_label}",
            alpha=0.6,
            s=scatter_size,
        )
    elif style == "density":
        # Calculate the 2D histogram
        hist, xedges, yedges = np.histogram2d(x, y, bins=100)
        # Create a meshgrid for the edges
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
        # Plot the density as a contour plot
        ax.contourf(X, Y, hist.T, levels=20, cmap=cmap, alpha=0.7)
    else:
        raise ValueError("Style must be either 'scatter' or 'density'.")
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    if oppose == "models":
        if x_lims is None:
            x_lims = (min(x), max(x))
        if y_lims is None:
            y_lims = (min(y), max(y))
        ax.plot(
            [x_lims[0], x_lims[1]],
            [y_lims[0], y_lims[1]],
            "k--",
            lw=2,
            label="1:1 line",
        )
    if linear_fit:
        # Fit a linear regression line

        coeffs = np.polyfit(x, y, 1)
        fit_line = np.polyval(coeffs, x)
        ax.plot(x, fit_line, "r-", lw=2, label="Linear fit")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Correlation between {x_label} and {y_label}")
    ax.legend()

    return fig
