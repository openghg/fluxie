import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import colorsys
import logging
from typing import List

logger = logging.getLogger(__name__)

default_color_palette = {
    0: [["#2c74a7", "#9ecae1"]],
    1: [["#cf4c0c", "#fdae6b"]],
    2: [["#2b934a", "#a1d99b"]],
    3: [["#695fa0", "#bcbddc"]],
    4: [["#783632", "#d6616b"]],
    5: [["#7e612c", "#e7ba52"]],
    6: [["#32346d", "#6b6ecf"]],
    7: [["#596a33", "#b5cf6b"]],
    8: [["#6f3b68", "#ce6dbd"]],
}

# population from 2018 to 2023 (at Jan 1 each year)
bel_pop = np.array([11.399, 11.455, 11.522, 11.555, 11.618, 11.723])
lux_pop = np.array([0.602, 0.614, 0.626, 0.635, 0.645, 0.661])
bel_pop_r = np.round(np.mean(bel_pop / (bel_pop + lux_pop)), 3)

mf_labels = {
    "mf_prior": "prior",
    "mf_posterior": "posterior",
    "mf_bc_prior": "prior baseline",
    "mf_bc_posterior": "posterior baseline",
    "mf_bias_prior": "prior bias",
    "mf_bias_posterior": "posterior bias",
    "mf_outer_prior": "prior outer region mf",
    "mf_outer_posterior": "posterior outer region mf",
    "mf_observed": "observed",
    "stdev_mf_observed_repeatability": "obs repeatability mf uncertainty",
    "stdev_mf_observed_variability": "obs variability mf uncertainty",
    "stdev_mf_model": "model uncertainty",
    "stdev_mf_total": "total uncertainty",
}

mf_color_index = {
    "mf_prior": 1,
    "mf_posterior": 0,
    "mf_bc_prior": 1,
    "mf_bc_posterior": 0,
    "mf_bias_prior": 1,
    "mf_bias_posterior": 0,
    "mf_outer_prior": 1,
    "mf_outer_posterior": 0,
    "mf_observed": 1,
    "stdev_mf_observed_repeatability": 0,
    "stdev_mf_observed_variability": 0,
    "stdev_mf_model": 1,
    "stdev_mf_total": 1,
}

flux_labels = {
    "flux_total_prior": "Prior",
    "flux_total_posterior_inversion_grid": "Posterior",
    "flux_total_posterior": "Posterior",
    "posterior_prior_diff": "Posterior - Prior",
    "posterior_prior_diff_inversion_grid": "Posterior - Prior",
    "posterior_mean_diff": "Posterior Anomaly",
    "posterior_mean_diff_inversion_grid": "Posterior Anomaly",
}

stat_labels = {
    "pearson": "Pearson correlation coefficient",
    "rmse": "RMSE",
    "bias": "Bias",
    "crmse": "Centered RMSE",
    "sd_sim": "Simulated StDev",
    "sd_obs": "Observed StDev",
    "nrmse": "Normalised RMSE",
    "sd_res": "Standard deviation of residuals",
    "nn": "Number of observations",
}

# Acceptable units and conversion factor to base unit
units_scale = {
    "mf": {"mol mol-1": 1, "ppm": 1e-6, "ppb": 1e-9, "ppt": 1e-12},  # mf base unit
    "amount": {"kmol": 1e3, "mol": 1},  # amount of substance base unit
    "mass": {"Tg": 1e12, "Gg": 1e9, "Mg": 1e6, "kg": 1e3, "g": 1},  # mass base unit
    "time": {
        "yr": 60 * 60 * 24 * 365,
        "a": 60 * 60 * 24 * 365,
        "s": 1,  # time base unit
    },
    "length": {"km": 1e3, "m": 1},  # length base unit
    "nd": {"1": 1},  # non-dimensional
}


# -----------------------------------------------------------------------------------------------------------
# -- PRINT SETTINGS
# -----------------------------------------------------------------------------------------------------------


def set_print_settings(presentation_mode: bool = False) -> dict[int, list]:
    """
    Sets font size and annotation coordinates.

    Args:
        presentation_mode (logical) (optional):
            If True, use bigger fonts (ideal for presentation slides)

    Returns:
        annotate_coords (dict of lists):
            Coordinates to annotate histogram.
    """

    if presentation_mode:
        # Set big font size (ideal for presentation slides)
        plt.rc("font", size=15)
        plt.rc("axes", titlesize=18)
        plt.rc("axes", labelsize=16)
        plt.rc("xtick", labelsize=15)
        plt.rc("ytick", labelsize=15)
        plt.rc("legend", fontsize=14)

        annotate_coords = {0: [0.58, 0.7], 1: [0.58, 0.4], 2: [0.58, 0.1]}

        logger.warning("Using big fonts when plotting. You might need to define shorter labels.")

    else:
        # Set small font size (ideal for text documents)
        plt.rc("font", size=11)
        plt.rc("axes", titlesize=11)
        plt.rc("axes", labelsize=10)
        plt.rc("xtick", labelsize=11)
        plt.rc("ytick", labelsize=11)
        plt.rc("legend", fontsize=10)

        annotate_coords = {0: [0.65, 0.80], 1: [0.65, 0.60], 2: [0.65, 0.40]}

    return annotate_coords


# -----------------------------------------------------------------------------------------------------------
# -- LABELS
# -----------------------------------------------------------------------------------------------------------


def set_model_labels(models: list[str], config_data: dict[str, dict], get_labels_from_file: bool) -> dict[str, str]:
    """
    Sets the label of each model.

    Args:
        models (list of str):
            Keys specifying model names, e.g. ['intem','elris']
        config_data (dict of dict):
            Dictionary with settings read from json file.
            Use json filenames as keys.
        get_labels_from_file (bool):
            If True, tries to retrieve model labels from models_info.json.
            If False, buids labels automatically from model names.

    Returns:
        model_labels (dict of str):
            Label to be used in plot, per model.
    """

    model_labels = {}

    for m in models:
        # Get model name components
        name_tags = m.split("_")

        # Get labels
        label = None
        if get_labels_from_file and "model_labels" in config_data["models_info"]:
            label = config_data["models_info"]["model_labels"].get(m, None)

        if label is None:
            label = " ".join(name_tags)
        model_labels[m] = label

    return model_labels


# -----------------------------------------------------------------------------------------------------------
# -- COLORS
# -----------------------------------------------------------------------------------------------------------


def set_model_colors(models: list[str]) -> tuple[dict[str, str], dict[int, list[list[str]]]]:
    """
    Sets plotting colors for each model.
    It adds hue-shifted color pairs to the default color palette if needed.

    Args:
        models (list of str): Keys specifying model names, e.g. ['intem','elris']

    Returns:
        model_colors (dict of lists): List of colors to be used by each model.
        color_palette (dict): Dictionary of grouped color pairs.
    """

    model_colors = dict()

    # Get unique inversion systems
    dict_invs = {inv: [m for m in models if inv in m] for inv in dict.fromkeys(m.split("_")[0] for m in models)}

    if len(default_color_palette) < len(dict_invs):
        raise ValueError(
            f"The number of inversion systems to plot is greater than the number of pre-defined colors."
            f"Add more keys to default_color_palette."
        )

    # Add hue-shifted pairs to the color palette and to the model dict
    color_palette = {inv: default_color_palette[k].copy() for k, inv in enumerate(dict_invs)}
    for inv, models in dict_invs.items():
        for imod, m in enumerate(models):
            sign = 1 if imod % 2 == 0 else -1
            shift = (15 + (imod + 1) * 10) * sign 
            add_new_color_pair(color_palette, group_key=inv, hue_shift_degrees=shift)
            model_colors[m] = color_palette[inv][-1]

    return model_colors, color_palette


def plot_color_palette(color_palette: dict[int, list[list[str]]]) -> None:
    """
    Plots the generated color palette.

    Args:
        color_palette (dict): Dictionary of grouped color pairs.
    Returns:
        None
    """

    n_groups = len(color_palette)
    pairs_per_group = max(len(pairs) for pairs in color_palette.values())

    fig_height = n_groups * 0.8 + 1
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.set_xlim(0, pairs_per_group * 2)
    ax.set_ylim(0, n_groups)
    ax.axis("off")

    for row, (group_id, pairs) in enumerate(color_palette.items()):
        y = n_groups - row - 1
        for col, (dark, light) in enumerate(pairs):
            # Draw dark rectangle
            ax.add_patch(mpatches.Rectangle((col * 2, y), 1, 0.8, color=dark))
            # Draw light rectangle
            ax.add_patch(mpatches.Rectangle((col * 2 + 1, y), 1, 0.8, color=light))

        # Label the group
        ax.text(-0.5, y + 0.4, f"{group_id}", va="center", ha="right", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.show()


def hex_to_rgb(hex_color: str) -> tuple:
    return mcolors.to_rgb(hex_color)


def rgb_to_hsl(rgb: tuple) -> tuple:
    return colorsys.rgb_to_hls(*rgb)  # H, L, S


def hsl_to_rgb(hsl: tuple) -> tuple:
    return colorsys.hls_to_rgb(*hsl)


def shift_hue(h: float, degrees: float) -> float:
    """Shift hue (0–1) by degrees (0–360 scale)."""
    return (h + degrees / 360.0) % 1.0


def get_lightness_delta(pair: List[str]) -> float:
    rgb1, rgb2 = map(hex_to_rgb, pair)
    _, l1, _ = rgb_to_hsl(rgb1)
    _, l2, _ = rgb_to_hsl(rgb2)
    return l2 - l1  # Could be negative or positive


def add_new_color_pair(
    color_palette: dict[int, list[list[str]]], group_key: int, hue_shift_degrees: float = 10.0
) -> None:
    """
    Adds a new color pair to `color_palette[group_key]` by hue-shifting
    the first pair and applying the same lightness delta.
    """
    if group_key not in color_palette or not color_palette[group_key]:
        raise ValueError(f"No valid color pair in group {group_key}.")

    base_cpair = color_palette[group_key][0]  # Get the first color pair in the group
    base_rgb = hex_to_rgb(base_cpair[0])
    h, l, s = rgb_to_hsl(base_rgb)
    delta_l = get_lightness_delta(base_cpair)

    # Apply hue shift
    new_h = shift_hue(h, hue_shift_degrees)

    # Generate new color pair
    dark_rgb = hsl_to_rgb((new_h, l, s))
    light_l = l + delta_l
    light_l = max(0.0, min(1.0, light_l))  # clamp to [0,1]
    light_rgb = hsl_to_rgb((new_h, light_l, s))

    # Convert to hex and add to the group
    new_pair = [mcolors.to_hex(dark_rgb), mcolors.to_hex(light_rgb)]
    color_palette[group_key].append(new_pair)


def get_default_colors() -> list[str]:
    """
    Returns the colors from the current matplotlib color cycle.

    Returns:
        color (str):
            Color to be used in plot.
    """

    return plt.rcParams["axes.prop_cycle"].by_key()["color"]
