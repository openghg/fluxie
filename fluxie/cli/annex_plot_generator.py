import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config_annex_plot import AnnexConfig

from fluxie.cli.utils_annex_plot import (
    get_species_specific_settings,
    create_str_dataframe,
    make_table,
)
from fluxie.io import read_config_files, read_model_output, read_flux_total_fgases
from fluxie.operators.select import slice_flux
from fluxie.config import set_model_colors, set_model_labels
from fluxie.plots.flux_timeseries import plot_country_flux
from fluxie.plots.flux_map import (
    plot_flux_map_over_time,
    plot_flux_map_model_comparison,
)

logger = logging.getLogger(__name__)


def produce_plots(
    region: str,
    output_path: str | Path,
    inventory_years: list | str,
) -> pd.DataFrame:
    """
    Create plots, table amd tex files to be used in annex reports.
    Info such as data directory, species to use,.. are read in adjacent config_annex_plot.py file.
    For monthly species, 6 plots are made :
        - country flux annual average covering the max possible time range,
        - country flux annual average covering the PARIS window (2018-202x),
        - country flux monthly results covering the PARIS window (2018-202x),
        - country flux mean seasonal cycle covering the PARIS window (2018-202x),
        - average flux map over the PARIS window (2018-202x),
        - seasonal flux map over the PARIS window (2018-202x).
    For annual and combined species, 2 plots are made:
        - country flux annual results covering the PARIS window (2018-202x),
        - average flux map over the PARIS window (2018-202x).
    For every entry in `network_comparisons` (config), maps of two inversion runs and their
    difference are made (e.g. expanded network vs. base network).

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
            annual_res_list.append(res_dict)

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

            # 1.3) Plot mean seasonal cycle of country fluxes over PARIS time window
            settings_seasonal_cycle = getattr(
                annex_config_data, "kwargs_country_flux_seasonal_cycle", None
            )
            if settings_seasonal_cycle:
                print(f"- Seasonal cycle of country fluxes {start_date} - {end_date}")
                kwargs_country_flux_seasonal_cycle = get_species_specific_settings(
                    species, period, settings_seasonal_cycle
                )
                # Species-specific settings override the general ones (e.g. plot_inventory)
                fig, _ = plot_country_flux(
                    ds_all_flux_scaled,
                    species=species,
                    model_colors=model_colors,
                    model_labels=model_labels,
                    start_date=start_date,
                    end_date=end_date,
                    config_data=config_data,
                    **{
                        **annex_config_data.kwargs_country_flux_general,
                        **kwargs_country_flux_seasonal_cycle,
                    },
                )
                full_path = (
                    output_path
                    / f"{species}_country_flux_seasonal_cycle_parisonly_{region}.png"
                )
                fig.savefig(full_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
                plt.close()

        else:
            # Store results for .csv and table
            annual_res_list.append(res_dict)

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

    ### Network comparison maps
    for comparison in getattr(annex_config_data, "network_comparisons", []):
        print(f"-- NETWORK COMPARISON {comparison['name']}")
        plot_network_comparison_maps(
            annex_config_data, config_data, comparison, region, output_path
        )

    print("\n--- ALL PLOTS GENERATED SUCCESSFULLY! ---")

    print("\n\n--- GENERATING TABLES ---")
    annual_res = pd.concat(annual_res_list, ignore_index=True)
    annual_res["model"] = annual_res["model"].replace({"Mean": "PARIS mean"})

    hfcs_list = [s for s in annual_res.species.unique() if s[:3].lower() == "hfc"]
    pfcs_list = [
        s for s in annual_res.species.unique() if s[:3].lower() in ["pfc", "cf4"]
    ]
    main_gases_list = ["ch4", "n2o", "sf6", "nf3", "all_pfc", "all_hfc"]

    for name, species_list in zip(
        ["hfc", "pfc", "main_gases"], [hfcs_list, pfcs_list, main_gases_list]
    ):
        print(f"\nTABLE {name.upper().replace('_',' ')}")
        sp_res = create_str_dataframe(
            annual_res,
            inventory_years,
            species_list,
            table_start_date=annex_config_data.start_date_table,
        )
        make_table(sp_res, output_path / f"{name}_res_{region}.tex", inventory_years)
        sp_res.to_csv(output_path / f"{name}_res_{region}.csv", index=False)

    print("\n--- TABLES GENERATED SUCCESSFULLY! ---")
    return annual_res


def plot_network_comparison_maps(
    annex_config_data: AnnexConfig,
    config_data: dict,
    comparison: dict,
    region: str,
    output_path: Path,
) -> None:
    """
    Plot flux maps of two inversion runs (e.g. different site networks) and their difference,
    averaged over the spatial maps time window, and save one figure per species.

    Args:
        annex_config_data: annex settings (see config_annex_plot.py).
        config_data: settings read from the json config files.
        comparison: dictionary with keys "name", "species" and "models" (see `network_comparisons` in config_annex_plot.py).
        region: region/country of focus.
        output_path: directory where to store the figures.
    """
    models = list(comparison["models"].keys())
    if len(models) != 2:
        raise ValueError(
            f"Network comparison '{comparison['name']}' must contain exactly two models, got {models}."
        )

    start_date = annex_config_data.start_date_spatial_maps
    end_date = annex_config_data.end_date
    standard_run = config_data.get("models_info", {}).get("standard_run", {})

    for species in comparison["species"]:
        print(f"- {species.upper()} network comparison map {start_date} - {end_date}")
        period = "monthly" if species in annex_config_data.monthly_species else "yearly"

        # Each model needs its own standard run, otherwise both models could fall back
        # to the same "default" run and the comparison would be meaningless
        models_without_run = [
            m for m in models if species not in standard_run.get(m, {})
        ]
        if models_without_run:
            logger.warning(
                f"Network comparison '{comparison['name']}': no standard_run entry for {models_without_run} ({species}) "
                "in models_info.json. Skipping this species."
            )
            continue

        ds_all_flux = read_model_output(
            annex_config_data.data_dir,
            "flux",
            species,
            models,
            config_data,
            period=period,
            add_sites_to_flux=True,
            read_standard_run=True,
        )
        missing_models = [m for m in models if m not in ds_all_flux]
        if missing_models:
            logger.warning(
                f"Network comparison '{comparison['name']}': no {period} flux file found for {missing_models} ({species}). "
                "Skipping this species."
            )
            continue

        ds_all_flux_scaled = slice_flux(
            ds_all_flux,
            config_data,
            start_date,
            end_date,
            species=species,
            country_flux_units_print=annex_config_data.country_flux_units_print,
            flux_units_print=annex_config_data.flux_units_print,
        )

        fig = plot_flux_map_model_comparison(
            ds_all_flux_scaled,
            models=models,
            species=species,
            region=annex_config_data.kwargs_maps_general["region"],
            config_data=config_data,
            model_labels=comparison["models"],
            add_markers=annex_config_data.kwargs_maps_general["add_markers"],
            set_fluxlim=annex_config_data.fluxlim.get(species, "auto"),
            set_fluxlim_percentile=annex_config_data.fluxlim_percentile.get(
                species, None
            ),
            **annex_config_data.kwargs_maps_network,
        )
        full_path = (
            output_path
            / f"{species}_network_comparison_{comparison['name']}_{region}.png"
        )
        fig.savefig(full_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        plt.close()
