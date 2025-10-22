import pandas as pd
import numpy as np
from pathlib import Path


def get_species_specific_settings(
    species: str, period: str, settings: list | dict
) -> list | dict:
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

def create_str_dataframe(
    res: dict,
    inventory_years: str | int,
    species: str | list[str],
    region: str | None = None,
    sector: str = "total",
    model: str = "PARIS mean",
    table_start_date: str | None = None,
) -> pd.DataFrame:
    
    if not table_start_date:
        table_start_date =  np.datetime64("1900-01-01")   
    elif isinstance(table_start_date, str):
        table_start_date = np.datetime64(table_start_date)         
    
    if not region:
        if res.country.unique().size!=1:
            raise ValueError(
                f"`region` parameter should be provided when there is more than one region in `res` (currently present: {res.country.unique()})."
            )
        region = res.country.unique()[0]

    if not sector:
        if res.sector.unique().size!=1:
            raise ValueError(
                f"`sector` parameter should be provided when there is more than one region in `res` (currently present: {res.sector.unique()})."
            )
        sector = res.sector.unique()[0]

    if not isinstance(species,list):
        species = [species,]

    res["time"] = pd.to_datetime(res["time"])
    data = res[(res.country==region)
                &(res.model.isin([model,f"inventory_{inventory_years}"]))
                &res.species.isin(species)
                &(res.type.isin(["posterior","inventory"]))
                &(res.time>=table_start_date)
                ].reset_index(drop=True)
    
    data["year"] = pd.to_datetime(data["time"]).dt.year.astype(str)

    values = data.mean_val.apply(lambda x: [(val[0],val[1][0],val[1][1:]) for val in f"{x:.2e}".split("e")]).values
    values = [(val, sign, exp) for val, sign, exp in values if ("").join([val,sign,exp])!="00+00"]
    
    if any([(sign=="+" or int(exp)==1) for _, sign, exp in values]):
        unit = "$\\rm{TgCO}_{2}\\rm{-eq} \\cdot \\rm{yr}^{-1}$"
        default_digit = 2
    else:
        for var in ["mean_val","min_unc","max_unc"]:
            data[var] *= 1e3
        unit = "$\\rm{GgCO}_{2}\\rm{-eq} \\cdot \\rm{yr}^{-1}$"
        default_digit = 1

    data["n_digits"] = data.apply(lambda x : 1 if x.species in ["ch4", "n2o", "all_hfc", "all_pfc", "sf6"] else default_digit, axis=1)

    data["val"] = data.apply(lambda x : f"{x.mean_val:.{x.n_digits}f}" if x.type=="inventory"
                             else f"{x.mean_val:.{x.n_digits}f} \\pm {(x.max_unc-x.min_unc)/2:.{x.n_digits}f}",
                             axis=1)

    output = data.pivot(index=["model","species"],columns="year",values = "val").reset_index()
    output.columns.name = None
    output.fillna(" ",inplace=True)

    output.rename(columns={"model":"source"},inplace=True)
    output["source"] = output["source"].apply(lambda x: x.replace("inventory_","NIR "))

    output.sort_values(by=["species","source"], inplace=True)

    species_name = {"ch4":"CH_4", "n2o":"N_2O", "sf6": "SF6", "cf4": "PFC-14", "all_pfc": "Total PFC", "all_hfc": "Total HFC"}
    for species in output.species.unique():
        if species not in species_name.keys():
            species_name[species] = species_name.replace("hfc","HFC-").replace("pfc","PFC-")
    output.replace(species_name, inplace=True)

    columns = np.concatenate(
        [
            ["species", "source"],
            np.sort(
                [
                    col
                    for col in output.columns
                    if col not in ["species", "source"]
                ]),
        ]
    )
    output = output[columns]
    
    return output, unit


def make_table(
    df: pd.DataFrame,
    output_path,
    inventory_years: str | int,
    unit: str = "$\\rm{TgCO}_{2}\\rm{-eq} \\cdot \\rm{yr}^{-1}$",
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
        + " in " + unit + f" according to the National Inventory Report (NIR) {inventory_years} and the inversions done in the PARIS project. For the PARIS estimation, the mean of the 3 inversion models is displayed, along with a range of uncertainty estimated via the half distance between the maximum and minimum uncertainties of the different models."
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
