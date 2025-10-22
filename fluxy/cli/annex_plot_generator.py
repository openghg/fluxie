import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config_annex_plot import AnnexConfig

from fluxy.cli.utils_annex_plot import (
    get_species_specific_settings,
    dict_to_str_dataframe,
    make_table,
)
from fluxy.io import read_config_files, read_model_output, read_flux_total_fgases
from fluxy.operators.select import slice_flux
from fluxy.config import set_model_colors, set_model_labels
from fluxy.plots.flux_timeseries import plot_country_flux
from fluxy.plots.flux_map import plot_flux_map_over_time

logger = logging.getLogger(__name__)


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
            Inventory year to use in the plots. If a list is given, only the last item will be used in the tables.

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

    # Get end date and list of species to plot
    end_date = annex_config_data.end_date

    all_species = (
        annex_config_data.monthly_species
        + annex_config_data.yearly_species
        + annex_config_data.combined_species
    )

    # Get last inventory year (most recent)
    if isinstance(inventory_years, list):
        inventory_years = inventory_years[-1]

    print("\n\n--- GENERATING PLOTS ---")
    for species in all_species:
        print(f"-- {species.upper()}")

        ### Get settings
        period = "monthly" if species in annex_config_data.monthly_species else "yearly"
        start_date = annex_config_data.start_date[period]
        models_country_flux = get_species_specific_settings(
            species, period, annex_config_data.models_country_flux
        )
        models_spatial_maps = get_species_specific_settings(
            species, period, annex_config_data.models_spatial_maps
        )
        kwargs_country_flux_species_specific = get_species_specific_settings(
            species, period, annex_config_data.kwargs_country_flux_species_specific
        )

        ### Country fluxes
        # Read and slice data
        if "all" in species:
            ds_all_flux_scaled = read_flux_total_fgases(
                annex_config_data.data_dir,
                species,
                models_country_flux,
                config_data,
                region,
                start_date,
                end_date,
                period=period,
            )
        else:
            ds_all_flux = read_model_output(
                annex_config_data.data_dir,
                "flux",
                species,
                models_country_flux,
                config_data,
                period=period,
                add_sites_to_flux=True,
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
        model_colors = set_model_colors(models_country_flux)
        model_labels = set_model_labels(
            models_country_flux, config_data, get_labels_from_file=True
        )

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
            **kwargs_country_flux_species_specific,
        )
        full_path = output_path / f"{species}_country_flux_annual_longrun_{region}.png"
        fig.savefig(full_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        plt.close()

        # Zoomed-in plots for monthly species only
        if species in annex_config_data.monthly_species:

            # Get settings
            kwargs_country_flux_monthly_species_special = get_species_specific_settings(
                species,
                period,
                annex_config_data.kwargs_country_flux_monthly_species_special,
            )

            # Re-slice the data
            start_date = annex_config_data.start_date_paris_window
            ds_all_flux_scaled = slice_flux(
                ds_all_flux,
                config_data,
                start_date,
                end_date,
                species=species,
                country_flux_units_print=annex_config_data.country_flux_units_print,
                flux_units_print=annex_config_data.flux_units_print,
            )

            # 1.1) Plot annual country fluxes over PARIS time window
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
                **kwargs_country_flux_species_specific,
            )
            full_path = (
                output_path / f"{species}_country_flux_annual_parisonly_{region}.png"
            )
            fig.savefig(full_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
            plt.close()

            # Store results for .csv and table
            annual_res = dict_to_str_dataframe(
                res_dict,
                inventory_years,
                species,
                table_start_date=annex_config_data.start_date_table,
            )
            annual_res_list.append(annual_res)

            # 1.2) Plot monthly country fluxes over PARIS time window
            print(f"- Monthly country fluxes {start_date} - {end_date}")
            fig, res_dict = plot_country_flux(
                ds_all_flux_scaled,
                species=species,
                model_colors=model_colors,
                model_labels=model_labels,
                start_date=start_date,
                end_date=end_date,
                config_data=config_data,
                **annex_config_data.kwargs_country_flux_general,
                **kwargs_country_flux_monthly_species_special,
            )
            full_path = (
                output_path / f"{species}_country_flux_monthly_parisonly_{region}.png"
            )
            fig.savefig(full_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
            plt.close()

        else:
            # Store results for .csv and table
            annual_res = dict_to_str_dataframe(
                res_dict,
                inventory_years,
                species,
                table_start_date=annex_config_data.start_date_table,
            )
            annual_res_list.append(annual_res)

        ### Spatial maps
        if "all" not in species:
            # Select datasets to plot and re-slice the data
            start_date = annex_config_data.start_date_spatial_maps
            ds_all_flux = {m: ds_all_flux[m] for m in models_spatial_maps}
            ds_all_flux_scaled = slice_flux(
                ds_all_flux,
                config_data,
                start_date,
                end_date,
                species=species,
                country_flux_units_print=annex_config_data.country_flux_units_print,
                flux_units_print=annex_config_data.flux_units_print,
            )

            # 2) Plot spatial map of the posterior fluxes averaged over PARIS time window
            print(f"- Average map {start_date} - {end_date}")
            dt = int(end_date[:4]) - int(start_date[:4])
            fig = plot_flux_map_over_time(
                ds_all_flux_scaled,
                species=species,
                model_labels=model_labels,
                config_data=config_data,
                dt=dt,
                set_fluxlim=annex_config_data.fluxlim.get(species, "auto"),
                set_fluxlim_percentile=annex_config_data.fluxlim_percentile.get(
                    species, None
                ),
                **annex_config_data.kwargs_maps_general,
                **annex_config_data.kwargs_maps_mean,
            )
            full_path = output_path / f"{species}_posterior_map_{region}.png"
            fig.savefig(full_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
            plt.close()

            if species in annex_config_data.monthly_species:
                # 3) Seasonnal maps
                print(f"- Seasonnal map {start_date} - {end_date}")
                fig = plot_flux_map_over_time(
                    ds_all_flux_scaled,
                    species=species,
                    model_labels=model_labels,
                    config_data=config_data,
                    set_fluxlim_percentile=annex_config_data.difflim_percentile.get(
                        species, None
                    ),
                    **annex_config_data.kwargs_maps_general,
                    **annex_config_data.kwargs_maps_seasonnal,
                )
                full_path = output_path / f"{species}_seasonal_map_{region}.png"
                fig.savefig(full_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
                plt.close()

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
    make_table(hfc_res, output_path / f"hfc_res_{region}.tex", inventory_years)
    hfc_res.to_csv(output_path / f"hfc_res_{region}.csv", index=False)

    print("\nTABLE PFC")
    pfc_res = annual_res[
        annual_res.species.apply(lambda x: x[:3].lower() == "pfc" or x.lower() == "cf4")
    ].copy()
    pfc_res["species"] = pfc_res.species.apply(lambda x: x.replace("pfc", "PFC-"))
    pfc_res["species"] = pfc_res.species.apply(lambda x: x.replace("cf4", "PFC-14"))
    make_table(pfc_res, output_path / f"pfc_res_{region}.tex", inventory_years)
    pfc_res.to_csv(output_path / f"pfc_res_{region}.csv", index=False)

    print("\nTABLE main gases")
    main_gases_res = annual_res[
        annual_res.species.isin(["ch4", "n2o", "sf6", "all_pfc", "all_hfc"])
    ].copy()
    main_gases_res["species"] = main_gases_res.species.apply(
        lambda x: x.upper()
        .replace("ALL_", "Total ")
        .replace("CH4", "CH$_4$")
        .replace("N2O", "N$_2$O")
        .replace("SF6", "SF$_6$")
    )
    make_table(
        main_gases_res, output_path / f"main_gases_res_{region}.tex", inventory_years
    )
    main_gases_res.to_csv(output_path / f"main_gases_res_{region}.csv", index=False)

    print("\n--- TABLES GENERATED SUCCESSFULLY! ---")
    return annual_res
