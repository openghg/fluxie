import pandas as pd
import numpy as np
from pathlib import Path


def get_species_specific_settings(species: str, period: str, settings: list | dict) -> list | dict:
    """
    Get species-specific setting from dictionary.
    Returns the input settings if it is already a list.

    Args:
        species:
            Gas species, e.g. 'ch4'.
        period:
            Inversion period, "yearly" ou "monthly".
        settings:
            List or dictionary with <species> or <period> as keys.


    Returns:
        settings_species:
            List or dictionary with species-specific settings.
    """

    if isinstance(settings, list):
        return settings

    settings_species = settings.get(species, None)
    if settings_species is None:
        settings_species = settings.get(period, {})

    return settings_species


def dict_to_str_dataframe(
    res: dict, inventory_years: list | str | int, species: str
) -> pd.DataFrame:
    """
    Transform the dictionnary outputed by plot_flux_timeseries into a pandas.DataFrame of string that will be used in the latex tables for the annex reports.

    Args:
        res :
            dictionnary outputed by plot_flux_timeseries
        inventory_years :
            Inventory year to use. If a list is given, only the first will be used. The data will be lloked at in the `res` dictionnary with the key f"inventory_{inventory_years}"
        species :
            Gas species.

    Returns:
        pd.DataFrame(output) :
            Dataframe with columns ["species","source", *<years present in res>] and two rows : one for the PARIS mean estimates and one for the UNFCCC inventory estimates.
    """
    if isinstance(inventory_years, list):
        inventory_years = inventory_years[0]

    comb = res["combined"]

    inv_default = {
        "time": comb["time"],
        "value": np.array(
            [
                np.nan,
            ]
            * len(comb["time"])
        ),
    }
    inv = res.get(f"inventory_{inventory_years}", inv_default)

    if species in ["n2o", "ch4"]:
        n_digits = 0
    elif species in ["all_hfc", "all_pfc", "sf6"]:
        n_digits = 1
    else:
        n_digits = 2

    output = {
        "species": [
            species,
        ]
        * 2,
        "source": ["NIR " + inventory_years, "PARIS mean"],
    }
    for it, time in enumerate(comb["time"].astype("datetime64[Y]")):
        paris_val = f"{comb['mean'][it]:.{n_digits}f} \\pm {(comb['max'][it]-comb['min'][it])/2:.{n_digits}f}"
        inv_val = inv["value"][inv["time"].astype("datetime64[Y]") == time]
        if len(inv_val) == 1:
            output[str(time)] = [f"{inv_val[0]:.{n_digits}f}", paris_val]
        else:
            output[str(time)] = [None, paris_val]

    return pd.DataFrame(output)


def make_table(
    df: pd.DataFrame,
    output_path: Path,
    descriptive_cols: list[str] = ["species", "source"],
    hline_place: dict[str] = {"source": "PARIS mean"},
):
    if "hfc" in str(output_path):
        species = "HFCs"
    elif "pfc" in str(output_path):
        species = "PFCs"
    if "main_gases" in str(output_path):
        species = "the main greenhouse gases of focus"
    # Set latex Table env and number of cols
    tmp = str(output_path).split("/")[-1].split(".")[0]
    label = "\n \\label{" + tmp + "}"
    tmp = (
        "Emissions estimation for "
        + species
        + " in $\\rm{TgCO}_{2}\\rm{-eq} \\cdot \\rm{yr}^{-1}$ according to the National Inventory Report (NIR) 2024 and the inversions done in the PARIS project. For the PARIS estimation, the mean of the 3 inversion models is displayed, along with a range of uncertainty estimated via the half distance between the maximum and minimum uncertainties of the different models."
    )
    caption = "\n \\caption{" + tmp + "}"
    begin = (
        "\\begin{table}[H]\n \\small"
        + label
        + caption
        + "\n \\begin{center}\n  \\begin{tabular}{ "
        + len(descriptive_cols) * "l "
        + (len(df.columns) - len(descriptive_cols)) * "l "
        + "}"
    )

    # Set first line with columns title
    header = "     " + len(descriptive_cols) * " & "
    for y in df.columns[len(descriptive_cols) :]:
        header += y
        if y != df.columns[-1]:
            header += " & "

    table = begin + "\n" + header + " \\\\ \hline" + "\n"

    # Iterate over lines of dataframe
    prev_species = ""
    for idRow, row in df.iterrows():
        # Indentation
        l = "    "

        # Test if value for first column needed
        if row[descriptive_cols[0]] == prev_species:
            l += " & "
        else:
            l += row[descriptive_cols[0]] + " & "
        prev_species = row[descriptive_cols[0]]

        # Add values for other descriptive columns
        for col in descriptive_cols[1:]:
            l += row[col] + " & "

        # Add yearly values
        for y in df.columns[len(descriptive_cols) :]:
            l += "$ " + row[y] + " $"
            if y != df.columns[-1]:
                l += " & "

        # End line
        l += " \\\\ "

        # Add hline if needed
        for key in hline_place.keys():
            if row[key] == hline_place[key]:
                l += " \hline "

        # Add line to table
        table += l + "\n "

    # Close latex env
    end = str("  \\end{tabular}\n \\end{center}\n\\end{table}")

    table += end

    with open(output_path, "w") as text_file:
        text_file.write(table)
