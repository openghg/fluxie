import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config_annex_plot import AnnexConfig

from pathlib import Path

from fluxy.io import read_config_files, read_model_output, read_flux_total_fgases
from fluxy.operators.select import slice_flux
from fluxy.config import set_model_colors, set_model_labels
from fluxy.plots.flux_timeseries import plot_country_flux
from fluxy.plots.flux_map import plot_flux_map_over_time

logger = logging.getLogger(__name__)


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


def get_species_specific_settings(settings: list | dict, species: str) -> list | dict:
    """
    Get species-specific setting from dictionary.
    Returns the input settings if it is already a list.

    Args:
        settings:
            List or dictionary with species as keys or "default" key if species not present.
        species:
            Gas species, e.g. 'ch4'.

    Returns:
        settings_species:
            List or dictionary with species-specific settings.
    """

    if isinstance(settings, list):
        return settings

    settings_species = settings.get(species, None)
    if settings_species is None:
        settings_species = settings.get("default", {})

    return settings_species


def produce_plots(
    region: str,
    output_path: str | Path,
    inventory_years: list | str,
) -> pd.DataFrame:
    """
    Create plots, table amd tex files to be used in annex reports.
    Info such as data directory, species to use,.. are read in adjacent config_annex_plot.py file.
    For monthly species, 5 plots are made :
        - country flux annual average covering the max possible time range,
        - country flux annual average covering the PARIS window (2018-202x),
        - country flux monthly results covering the PARIS window (2018-202x),
        - average flux map over the PARIS window (2018-202x),
        - seasonal flux map over the PARIS window (2018-202x).
    For annual and combined species, 2 plots are made:
        - country flux annual results covering the PARIS window (2018-202x),
        - average flux map over the PARIS window (2018-202x).

    Args:
        region :
            Region/Country of focus.
        output_path :
            Path where to store the figures/tables/tex files.
        inventory_years :
            Inventory year to use in the plots. If a list is given, only the first item will be used in the tables.

    Returns:
        annual_res :
            Aggregated annual results (outputs from dict_to_str_dataframe) containing all species.
    """

    ### Initialization
    logger.warning(
        "Make sure standard filenames are correctly set in models_info.json."
    )
    config_data = read_config_files()
    annex_config_data = AnnexConfig(region, inventory_years)
    annual_res_list = list()

    # Converting output_path into pathlib.Path object
    output_path = Path(output_path)

    #### CH4 and N2O
    print("\n--- PLOTTING MONTHLY SPECIES ---")
    for species in annex_config_data.monthly_species:
        print(f"-- {species.upper()}")

        ### Country fluxes
        ## Long time window
        start_date = annex_config_data.start_date_monthly_species
        end_date = annex_config_data.end_date

        # Models to plot
        models_std = get_species_specific_settings(
            annex_config_data.models_monthly_species, species
        )

        # Species-specific settings
        kwargs_species_specific = get_species_specific_settings(
            annex_config_data.kwargs_country_flux_monthly_species_per_species,
            species,
        )

        # Read and slice data
        ds_all_flux = read_model_output(
            annex_config_data.data_dir,
            "flux",
            species,
            models_std,
            config_data,
            period="monthly",
            add_sites_to_flux = True,
            read_standard_run=True,
        )
        ds_all_flux_scaled = slice_flux(
            ds_all_flux,
            config_data,
            start_date,
            end_date,
            species=species,
            country_flux_units_print=annex_config_data.country_flux_units_print,
        )

        # Define plotting colors and labels
        model_colors = set_model_colors(models_std)
        model_labels = set_model_labels(models_std,config_data,get_labels_from_file=True)

        # 1.1) Plot annual country fluxes over long time window
        print(f"- Annual country fluxes {start_date} - {end_date}")
        fig, res_dict = plot_country_flux(
            ds_all_flux_scaled,
            species=species,
            model_colors=model_colors,
            model_labels=model_labels,
            start_date=start_date,
            end_date=end_date,
            config_data=config_data,
            **annex_config_data.kwargs_country_flux_general,
            **annex_config_data.kwargs_country_flux_monthly_species,
            **kwargs_species_specific,
        )
        full_path = output_path / f"{species}_country_flux_annual_longrun_{region}.png"
        fig.savefig(full_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        plt.close()

        ## PARIS time window
        start_date = annex_config_data.start_date_paris_window
        end_date = annex_config_data.end_date

        # Re-slice the data
        ds_all_flux_scaled = slice_flux(
            ds_all_flux,
            config_data,
            start_date,
            end_date,
            species=species,
            country_flux_units_print=annex_config_data.country_flux_units_print,
            flux_units_print=annex_config_data.flux_units_print,
        )

        # 1.2) Plot annual country fluxes over PARIS time window
        print(f"- Annual country fluxes {start_date} - {end_date}")
        fig, res_dict = plot_country_flux(
            ds_all_flux_scaled,
            species=species,
            model_colors=model_colors,
            model_labels=model_labels,
            start_date=start_date,
            end_date=end_date,
            config_data=config_data,
            **annex_config_data.kwargs_country_flux_general,
            **annex_config_data.kwargs_country_flux_monthly_species,
            **kwargs_species_specific,
        )
        full_path = (
            output_path / f"{species}_country_flux_annual_parisonly_{region}.png"
        )
        fig.savefig(full_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        plt.close()

        # Store results for .csv and table
        annual_res = dict_to_str_dataframe(res_dict[region], inventory_years, species)
        annual_res_list.append(annual_res)

        # 2) Plot monthly country fluxes over PARIS time window
        print(f"- Monthly country fluxes")
        fig, res_dict = plot_country_flux(
            ds_all_flux_scaled,
            species=species,
            model_colors=model_colors,
            model_labels=model_labels,
            start_date=start_date,
            end_date=end_date,
            config_data=config_data,
            **annex_config_data.kwargs_country_flux_general,
            **annex_config_data.kwargs_country_flux_monthly_species_special,
            **kwargs_species_specific,
        )
        full_path = (
            output_path / f"{species}_country_flux_monthly_parisonly_{region}.png"
        )
        fig.savefig(full_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        plt.close()

        ### Spatial maps
        start_date = annex_config_data.start_date_spatial_maps
        end_date = annex_config_data.end_date

        # Reselect datasets to plot
        models_std = get_species_specific_settings(
            annex_config_data.models_spatial_maps, species
        )

        # Re-slice the data
        ds_all_flux_scaled = slice_flux(
            ds_all_flux,
            config_data,
            start_date,
            end_date,
            species=species,
            country_flux_units_print=annex_config_data.country_flux_units_print,
            flux_units_print=annex_config_data.flux_units_print,
        )

        # Define plotting labels
        model_labels = set_model_labels(models_std,config_data,get_labels_from_file=True)

        # 3) Plot spatial map of the posterior fluxes averaged over PARIS time window
        print(f"- Average map")
        dt = int(end_date[:4]) - int(start_date[:4])
        fig = plot_flux_map_over_time(
            ds_all_flux_scaled,
            species=species,
            model_labels=model_labels,
            config_data=config_data,
            dt=dt,
            set_fluxlim_percentile=annex_config_data.fluxlim_percentile.get(
                species, None
            ),
            **annex_config_data.kwargs_maps_general,
            **annex_config_data.kwargs_maps_mean,
        )
        full_path = output_path / f"{species}_posterior_map_{region}.png"
        fig.savefig(full_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        plt.close()

        # 4) Seasonnal maps
        print(f"- Seasonnal map")
        fig = plot_flux_map_over_time(
            ds_all_flux_scaled,
            species=species,
            model_labels=model_labels,
            config_data=config_data,
            set_fluxlim_percentile=annex_config_data.fluxlim_percentile.get(
                species, None
            ),
            **annex_config_data.kwargs_maps_general,
            **annex_config_data.kwargs_maps_seasonnal,
        )
        full_path = output_path / f"{species}_seasonal_map_{region}.png"
        fig.savefig(full_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        plt.close()

    #### F-gases
    print("\n--- PLOTTING ANNUAL SPECIES ---")
    for species in annex_config_data.annual_species:
        print(f"-- {species.upper()}")

        ### Country fluxes
        ## Long time window
        start_date = annex_config_data.start_date_fgases
        end_date = annex_config_data.end_date

        # Models to plot
        models_std = get_species_specific_settings(
            annex_config_data.models_yearly_species, species
        )

        # Species-specific settings
        kwargs_species_specific = get_species_specific_settings(
            annex_config_data.kwargs_country_flux_yearly_species_per_species,
            species,
        )

        # Read and slice data
        ds_all_flux = read_model_output(
            annex_config_data.data_dir,
            "flux",
            species,
            models_std,
            config_data,
            period="yearly",
            add_sites_to_flux = True,
            read_standard_run=True,
        )
        ds_all_flux_scaled = slice_flux(
            ds_all_flux,
            config_data,
            start_date,
            end_date,
            species=species,
            country_flux_units_print=annex_config_data.country_flux_units_print,
        )

        # Define plotting colors and labels
        model_colors = set_model_colors(models_std)
        model_labels = set_model_labels(models_std,config_data,get_labels_from_file=True)

        # 1) Plot annual country fluxes over long time window
        print(f"- Annual country fluxes {start_date} - {end_date}")
        fig, res_dict = plot_country_flux(
            ds_all_flux_scaled,
            species=species,
            model_colors=model_colors,
            model_labels=model_labels,
            start_date=start_date,
            end_date=end_date,
            config_data=config_data,
            **annex_config_data.kwargs_country_flux_general,
            **annex_config_data.kwargs_country_flux_yearly_species,
            **kwargs_species_specific,
        )
        full_path = output_path / f"{species}_country_flux_annual_longrun_{region}.png"
        fig.savefig(full_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        plt.close()

        # Store results for .csv and table
        annual_res = dict_to_str_dataframe(res_dict[region], inventory_years, species)
        annual_res_list.append(annual_res)

        ### Spatial maps
        start_date = annex_config_data.start_date_spatial_maps
        end_date = annex_config_data.end_date
        dt = int(end_date[:4]) - int(start_date[:4])

        # Select and reslice the data
        models_std = get_species_specific_settings(
            annex_config_data.models_spatial_maps, species
        )
        ds_all_flux = {m: ds_all_flux[m] for m in models_std}
        ds_all_flux_scaled = slice_flux(
            ds_all_flux,
            config_data,
            start_date,
            end_date,
            species=species,
            country_flux_units_print=annex_config_data.country_flux_units_print,
            flux_units_print=annex_config_data.flux_units_print,
        )

        # Define plotting labels
        model_labels = set_model_labels(models_std,config_data,get_labels_from_file=True)

        # 3) Plot spatial map of the posterior fluxes averaged over PARIS window
        print(f"- Average map")
        fig = plot_flux_map_over_time(
            ds_all_flux_scaled,
            species=species,
            model_labels=model_labels,
            config_data=config_data,
            dt=dt,
            set_fluxlim_percentile=annex_config_data.fluxlim_percentile.get(
                species, None
            ),
            **annex_config_data.kwargs_maps_general,
            **annex_config_data.kwargs_maps_mean,
        )
        full_path = output_path / f"{species}_posterior_map_{region}.png"
        fig.savefig(full_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        plt.close()

    #### Total HFCs/PFCs
    start_date = annex_config_data.start_date_fgases
    end_date = annex_config_data.end_date

    print("\n--- PLOTTING COMBINED SPECIES ---")
    for species in annex_config_data.combined_species:
        print(f"-- {species.upper()}")

        ### Country fluxes
        ## Long time window
        # Models to plot
        models_std = get_species_specific_settings(
            annex_config_data.models_yearly_species, species
        )

        # Read and scale fluxes
        ds_all_flux_scaled = read_flux_total_fgases(
            annex_config_data.data_dir,
            species,
            models_std,
            config_data,
            region,
            start_date,
            end_date,
            period="yearly",
        )

        # Define plotting colors and labels
        model_colors = set_model_colors(models_std)
        model_labels = set_model_labels(models_std,config_data,get_labels_from_file=True)

        # 3) Plot annual country fluxes over long time window
        print(f"- Annual country fluxes {start_date} - {end_date}")
        fig, res_dict = plot_country_flux(
            ds_all_flux_scaled,
            species=species,
            model_colors=model_colors,
            model_labels=model_labels,
            start_date=start_date,
            end_date=end_date,
            config_data=config_data,
            **annex_config_data.kwargs_country_flux_general,
            **annex_config_data.kwargs_country_flux_yearly_species,
            **kwargs_species_specific,
        )
        full_path = output_path / f"{species}_country_flux_annual_longrun_{region}.png"
        fig.savefig(full_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        plt.close()

        # Store results for .csv and table
        annual_res = dict_to_str_dataframe(res_dict[region], inventory_years, species)
        annual_res_list.append(annual_res)

    print("\n--- ALL PLOTS GENERATED SUCCESSFULLY! ---")

    print("\n\n--- GENERATING TABLES ---")
    annual_res = pd.concat(annual_res_list).reset_index(drop=True).fillna(value=" ")
    columns = np.concatenate(
        [
            ["source", "species"],
            np.sort(
                [
                    int(col)
                    for col in annual_res.columns
                    if col not in ["species", "source"]
                ]
            ).astype(str),
        ]
    )
    annual_res = annual_res[columns]

    print("\nTABLE HFC")
    hfc_res = annual_res[
        annual_res.species.apply(lambda x: x[:3].lower() == "hfc")
    ].copy()
    hfc_res["species"] = hfc_res.species.apply(lambda x: x.replace("hfc", "HFC-"))
    make_table(hfc_res, output_path / f"hfc_res_{region}.tex")
    hfc_res.to_csv(output_path / f"hfc_res_{region}.csv", index=False)

    print("\nTABLE PFC")
    pfc_res = annual_res[
        annual_res.species.apply(lambda x: x[:3].lower() == "pfc" or x.lower() == "cf4")
    ].copy()
    pfc_res["species"] = pfc_res.species.apply(lambda x: x.replace("pfc", "PFC-"))
    pfc_res["species"] = pfc_res.species.apply(lambda x: x.replace("cf4", "PFC-14"))
    make_table(pfc_res, output_path / f"pfc_res_{region}.tex")
    pfc_res.to_csv(output_path / f"pfc_res_{region}.csv", index=False)

    print("\nTABLE main gases")

    main_gases_res = annual_res[
        annual_res.species.isin(["ch4", "n2o", "sf6", "all_pfc", "all_hfc"])
    ].copy()
    main_gases_res["species"] = main_gases_res.species.apply(
        lambda x: x.upper().replace("ALL_", "Total ")
    )
    make_table(main_gases_res, output_path / f"main_gases_res_{region}.tex")
    main_gases_res.to_csv(output_path / f"main_gases_res_{region}.csv", index=False)

    print("\n--- TABLES GENERATED SUCCESSFULLY! ---")
    return annual_res


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
