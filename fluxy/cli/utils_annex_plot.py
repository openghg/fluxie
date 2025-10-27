import pandas as pd
import numpy as np

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
    inventory_year: str | int,
    species: str | list[str],
    region: str | None = None,
    sector: str = "total",
    model: str = "PARIS mean",
    table_start_date: str | None = None,
) -> pd.DataFrame:
    """
    Create a dataframe with results for a specific country, region and model. The columns are "species", "units", "source" (which values are "<model>" and "NID <inventory_year>") and the years present. 
    The values are string of the form "<mean> \\pm <uncertainty>", made to be usable directly to make the tex output for the annexes tables. Set the units and right number of digits.
    NOTE: Assume that the units in res is Tg CO2-eq yr-1
    Args:
        res: pandas dataframe containing the results. It should be the output (or concatenation of outputs) of `plot_country_flux` from fluxy/plots/flux_timeseries.py
            The columns of this dataframe are "type", "model", "sector", "country", "species", "time", "mean_val", "min_unc", "max_unc".
        inventory_year: inventory year to put in the outputted dataframe (has to b present in <res>).
        species: species to put include in the table
        region: region to restict the results to
        sector: sector to restrict the results to
        model: model to restrict the results to
        table_start_date: start_date for the outputted results
    Return:
        output: pandas Dataframe contianing the string values that will be put in the .tex files for the annexes tables.
    """
    
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
                &(res.sector==sector)
                &(res.model.isin([model,f"inventory_{inventory_year}"]))
                &res.species.isin(species)
                &(res.type.isin(["posterior","inventory"]))
                &(res.time>=table_start_date)
                ].reset_index(drop=True)
    
    data["year"] = pd.to_datetime(data["time"]).dt.year.astype(str)
    species_order = data[data.model==model].groupby("species").mean_val.mean().sort_values(ascending=False).index

    rescaled_data = list()
    for species in data.species.unique():
        data_per_species = data[data.species==species].copy()

        _, exp = np.stack(data_per_species.mean_val.apply(lambda x: np.array(f"{x:.2e}".split("e")).astype(float)
                                                            if f"{x:.2e}"!="0.00e+00" else [np.nan,np.nan]).values).T
        max_exp = np.nanmax(exp)
        if max_exp<-1 and max_exp>=-4:
            for var in ["mean_val","min_unc","max_unc"]:
                data_per_species[var] *= 1e3
            data_per_species["units"] = "\\footnotesize{$\\left (\\rm{GgCO}_{2}\\rm{\\text{-}eq} \\cdot \\rm{yr}^{-1} \\right )$}"
            max_exp += 3
        elif max_exp<-4:
            for var in ["mean_val","min_unc","max_unc"]:
                data_per_species[var] *= 1e6
            data_per_species["units"] = "\\footnotesize{$\\left (\\rm{MgCO}_{2}\\rm{\\text{-}eq} \\cdot \\rm{yr}^{-1} \\right )$}"
            max_exp += 6
        else:
            data_per_species["units"] = "\\footnotesize{$\\left (\\rm{TgCO}_{2}\\rm{\\text{-}eq} \\cdot \\rm{yr}^{-1} \\right )$}"
        
        n_figure = 3 if species in ["ch4", "n2o"] else 2
        n_digits = int(n_figure - max_exp - 1)
        data_per_species["mean_val"] = data_per_species.mean_val.apply(lambda x: f"{x:.{n_digits}f}")
        data_per_species["unc"] = data_per_species.apply(lambda x: f"{(x.max_unc-x.min_unc)/2:.{n_digits}f}" if x.type!="inventory" else "", axis=1)

        data_per_species = pd.concat([data_per_species])

        rescaled_data.append(data_per_species)
    data = pd.concat(rescaled_data)

    data["val"] = data.apply(lambda x : x.mean_val if x.type=="inventory"
                             else f"{x.mean_val} \\pm {x.unc}",
                             axis=1)

    output = data.pivot(index=["model","species","units"],columns="year",values = "val").reset_index()
    output.columns.name = None
    output.fillna(" ",inplace=True)

    output.rename(columns={"model":"source"},inplace=True)
    output["source"] = output["source"].apply(lambda x: x.replace("inventory_","NID "))

    output["sort_col1"] = output.species.apply(lambda x : species_order.get_loc(x))
    output["sort_col2"] = output.source.apply(lambda x : 1 if x==model else 0)
    output.sort_values(by=["sort_col1","sort_col2"], inplace=True, ignore_index=True)
    del output["sort_col1"], output["sort_col2"]

    species_name = {"ch4":"CH$_4$", "n2o":"N$_2$O", "sf6": "SF$_6$", "nf3": "NF$_3$", "cf4": "PFC-14", "all_pfc": "Total PFC", "all_hfc": "Total HFC"}
    for species in output.species.unique():
        if species not in species_name.keys():
            species_name[species] = species.replace("hfc","HFC-").replace("pfc","PFC-")
    output.replace(species_name, inplace=True)

    index_col = ["species", "units", "source"]
    columns = np.concatenate(
        [
            index_col,
            np.sort(
                [
                    col
                    for col in output.columns
                    if col not in index_col
                ]),
        ]
    )
    output = output[columns]
    
    return output


def make_table(
    df: pd.DataFrame,
    output_path,
    inventory_years: str | int,
    descriptive_cols: list[str] = ["species", "units", "source"],
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
        + f" according to the National Inventory Document (NID) {inventory_years} and the inversions done in the PARIS project. For the PARIS estimation, the mean of all models is displayed, along with a range of uncertainty estimated via the half distance between the maximum and minimum uncertainties of the different models."
    )
    caption = "\n \\caption{" + tmp + "}"
    begin = (
        "\\begin{table}[H]\n \\small"
        + label
        + caption
        + "\n \\begin{center}\n  \\begin{tabular}{ "
        + (len(descriptive_cols)-1) * "l "
        + (len(df.columns) - len(descriptive_cols)) * "l "
        + "}"
    )

    # Set first line with columns title
    header = "     " + (len(descriptive_cols)-1) * " & "
    for y in df.columns[len(descriptive_cols) :]:
        header += y
        if y != df.columns[-1]:
            header += " & "

    table = begin + "\n" + header + " \\\\ \\toprule" + "\n"

    # Iterate over lines of dataframe
    prev_species = ""
    nrows = df.shape[0]
    for k, row in df.iterrows():
        # Indentation
        l = "    "

        # Test if value for first column needed
        if row[descriptive_cols[0]] == prev_species:
            l += row[descriptive_cols[1]] + " & "
        else:
            l += row[descriptive_cols[0]] + " & "
        prev_species = row[descriptive_cols[0]]

        # Add values for other descriptive columns
        l += row[descriptive_cols[2]] + " & "

        # Add yearly values
        for y in df.columns[len(descriptive_cols) :]:
            l += "$ " + row[y] + " $"
            if y != df.columns[-1]:
                l += " & "

        # End line
        l += " \\\\ "

        # Add hline if needed
        for key in hline_place.keys():
            if k == nrows-1:
                l += " \\bottomrule "
            elif row[key] == hline_place[key]:
                l += " \\midrule "

        # Add line to table
        table += l + "\n "

    # Close latex env
    end = str("  \\end{tabular}\n \\end{center}\n\\end{table}")

    table += end

    with open(output_path, "w") as text_file:
        text_file.write(table)
