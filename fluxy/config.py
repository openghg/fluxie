import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

color_palette = {
    0: [["blue", "dodgerblue"], ["dodgerblue", "skyblue"], ["deepskyblue", "cyan"]],
    1: [["purple", "mediumpurple"], ["deeppink", "pink"], ["darkorange", "red"]],
    2: [["darkgreen", "green"], ["limegreen", "palegreen"], ["olive", "lightgreen"]],
    3: [["darkorange", "orange"], ["gold", "khaki"], ["yellow", "lightyellow"]],
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

        logger.warning(
            "Using big fonts when plotting. You might need to define shorter labels."
        )

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


def set_model_colors(models: list[str]) -> dict[str, list]:
    """
    Sets plotting colors for each model.

    Args:
        models (list of str):
            Keys specifying model names, e.g. ['intem','elris']

    Returns:
        model_colors (dict of lists):
            List of colors to be used by each model.
    """

    model_colors = dict()
    max_color_groups = len(color_palette)

    # Get unique inversion systems
    # dict.fromkeys() is used because it conserves the order of the models
    unique_models = list(dict.fromkeys(m.split("_")[0] for m in models))
    n_unique_models = len(unique_models)

    if n_unique_models == 1:
        # The results to plot are from a single inversion system
        i = 0
        index_colors = 0

        # Use color_palette in order
        for m in models:
            if index_colors == len(color_palette[i]):
                i = i + 1
                index_colors = 0

            if i == max_color_groups:
                raise ValueError(
                    f"Number of models to plot is greater than number of pre-defined colors. Add more colors to color_palette."
                )

            model_colors[m] = color_palette[i][index_colors]
            index_colors = index_colors + 1

    else:
        # The results to plot are from multiple inversion systems
        index_colors = [0] * n_unique_models
        index_colors_max = [len(color_palette[i]) for i in color_palette]

        for m in models:
            model_name = m.split("_")[0]
            i = unique_models.index(model_name)

            if i == max_color_groups:
                raise KeyError(
                    f"color_palette has only {i} keys. Add more keys to plot results from more than {i} distinct inversion models."
                )

            if index_colors[i] == index_colors_max[i]:
                raise KeyError(
                    f"There are more than {index_colors[i]} results from {model_name} but only {index_colors[i]} elements in color_palette[{i}]. Add more pairs of colors to the list."
                )

            # For each inversion system, get plotting colors from a single color_palette key
            model_colors[m] = color_palette[i][index_colors[i]]
            index_colors[i] = index_colors[i] + 1

    return model_colors


def set_model_labels(
    models: list[str], config_data: dict[str, dict], get_labels_from_file: bool
) -> dict[str, str]:
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


def get_default_colors() -> list[str]:
    """
    Returns the colors from the current matplotlib color cycle.

    Returns:
        color (str):
            Color to be used in plot.
    """

    return plt.rcParams["axes.prop_cycle"].by_key()["color"]