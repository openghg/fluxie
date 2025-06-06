import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import config_annex_plot as annex_config

from pathlib import Path

from fluxy.io import read_config_files, read_model_output, read_flux_total_fgases
from fluxy.operators.select import slice_flux
from fluxy.config import set_model_colors, set_model_labels
from fluxy.plots.flux_timeseries import plot_country_flux

###########################################
### GENERAL SETTINGS
###########################################
# Species to plot
###########################################

def define_model_list(models, species, config_data):
    print("WARNING : each team should check that the right files are used. Still to be checked by RHIME, ELRIS, InTEM.")
    filepath = Path("filenames.json")
    models_std = []
    
    for m, model in enumerate(models):
        
        if filepath:
            with open(filepath, "r") as f:
                json_data = json.load(f)
            model_read = f"{model.split('_')[0]}_{json_data[model][species]}"
            
        else:
            model_read = f"{model.split('_')[0]}_{config_data['models_info']['standard_run']['default'][species]}"
            if "longrun" in model:
                model_read = f"{model_read}_longrun"
                
        models_std.append(model_read)
        
    return models_std

def produce_plots(region, output_path, inventory_years):

    ### Initialization
    config_data = read_config_files()    
    annual_res_list = list()
    
    ### Settings for country fluxes
    models_country_fluxes = [
        "InTEM_longrun",
        "InTEM",
        "ELRIS",
        "RHIME",
    ]  # NOTE: only options are basic model names w/ and w/o longrun
    scale_co2eq = True
    country_flux_units_print = "Tg CO2-eq yr-1"
        
    kwargs_country_flux_general = dict(
        plot_regions = region,
        inventory_years = inventory_years,
        data_dir = annex_config.data_dir,
        config_data = config_data,
        annex_mode = True,
        plot_inventory = True,
        fix_y_axes = False,
        add_prior = True,
        add_prior_unc = False,
        set_global_leg = False,
        country_codes_as_titles = None,
        plot_resample_and_original = False,
        plot_separate = [True, False, False, False],
        plot_combined = [False, True, True, True],
        return_res=True)
        
    kwargs_country_flux_monthly_species = dict(
        rolling_mean = False)
        
    kwargs_country_flux_annual_species = dict(
        rolling_mean = True)
    
    ### Settings for spatial maps
    models_spatial_maps = ["InTEM", "ELRIS", "RHIME"]
    plot_area = region
    plot_site_locations = True
    plot_point_markers = annex_config.point_markers[region]
    convert_flux_units = True
    set_fluxlim = "auto"
    plot_inversion_grid_flux = True

    ### CH4 and N2O
    print("\n--- PLOTTING COUNTRY FLUXES FOR CH4/N2O ---")
    for species in annex_config.monthly_species:

        # Long time window
        start_date = ["2008-01-01","2018-01-01","2018-01-01","2018-01-01"]
        end_date = "2024-01-01"
        
        models_std = define_model_list(models_country_fluxes, species, config_data)
        period = ["monthly" if "longrun" not in m else "yearly" for m in models_country_fluxes]
        
        ds_all_flux = read_model_output(annex_config.data_dir,"flux",species,models_std,config_data,period=period)
        ds_all_flux_scaled = slice_flux(ds_all_flux,config_data,start_date,end_date,species=species,country_flux_units_print=country_flux_units_print)

        ### Define plotting colors
        model_colors = set_model_colors(models_std)
        model_labels = set_model_labels(models_std,config_data,True)

        # 1.1) Plot annual country fluxes from 2008 to 2023 from intem_longrun and combined from 3 std_run    
        fig, res_dict = plot_country_flux(ds_all_flux_scaled,
                        species = species,
                        model_colors = model_colors,
                        model_labels = model_labels,
                        start_date = start_date[0],
                        end_date = end_date,
                        resample = "year",
                        resample_uncert_correlation = False,
                        **kwargs_country_flux_general,
                        **kwargs_country_flux_monthly_species)

        plot_name = f"{species}_country_flux_annual_longrun_{region}.png"
        full_path = os.path.join(output_path, plot_name)
        fig.savefig(full_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        plt.close()

        # PARIS time window
        start_date = "2018-01-01"
        end_date = "2024-01-01"

        ### Re-slice the data
        print(start_date,end_date,species,country_flux_units_print)
        ds_all_flux_scaled = slice_flux(ds_all_flux,config_data,start_date,end_date,species=species,country_flux_units_print=country_flux_units_print)

        # 1.2) Plot annual country fluxes from 2018 to 2023 from intem_longrun and combined from 3 std_run  
        fig, res_dict = plot_country_flux(ds_all_flux_scaled,
                        species = species,
                        model_colors = model_colors,
                        model_labels = model_labels,
                        start_date = start_date,
                        end_date = end_date,
                        resample = "year",
                        resample_uncert_correlation = False,
                        **kwargs_country_flux_general,
                        **kwargs_country_flux_monthly_species)

        plot_name = f"{species}_country_flux_annual_parisonly_{region}.png"
        full_path = os.path.join(output_path, plot_name)
        fig.savefig(full_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        plt.close()

        # Store results for .csv and table
        comb = res_dict[region]["combined"]
        try:
            inv = res_dict[region]["inventory"]
        except:
            print("NO INVENTORY FOUND")
            inv = {
                "time": comb["time"],
                "value": np.array(
                    [
                        np.NaN,
                    ]
                    * len(comb["time"])
                ),
            }

        tmp = {
            "species": [
                species,
            ]
            * 2,
            "source": ["NIR " + inventory_years[0], "PARIS mean"],
        }
        for it, time in enumerate(comb["time"].astype("datetime64[Y]")):
            paris_val = (
                f"{comb['mean'][it]:.0f} \\pm {(comb['max'][it]-comb['min'][it])/2:.0f}"
            )
            inv_val = inv["value"][inv["time"].astype("datetime64[Y]") == time]
            if len(inv_val) == 1:
                tmp[str(time)] = [f"{inv_val[0]:.0f}", paris_val]
            else:
                tmp[str(time)] = [None, paris_val]
        annual_res_list.append(pd.DataFrame(tmp))

        # Monthly country fluxes

        # 2) Plot monthly country fluxes from 2018 to 2023 from intem_longrun and combined from 3 std_run
        fig, res_dict = plot_country_flux(ds_all_flux_scaled,
                        species = species,
                        model_colors = model_colors,
                        model_labels = model_labels,
                        start_date = start_date,
                        end_date = end_date,
                        resample = None,
                        **kwargs_country_flux_general,
                        **kwargs_country_flux_monthly_species)

        plot_name = f"{species}_country_flux_monthly_parisonly_{region}.png"
        full_path = os.path.join(output_path, plot_name)
        fig.savefig(full_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        plt.close()

    ### Total HFCs/PFCs (w/o HFC-4310mee)
    start_date = [
        annex_config.start_date_fgases[plot_area],
        "2018-01-01",
        "2018-01-01",
        "2018-01-01",
    ]
    end_date = ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01"]

    print("\n--- PLOTTING COUNTRY FLUXES FOR TOTAL HFC/PFC ---")
    for species in annex_config.combined_species:
        
        ### Read and scale fluxes
        ds_all_flux_scaled = read_flux_total_fgases(annex_config.data_dir,species,models_country_fluxes,
                                                    config_data,region,start_date,end_date,
                                                    period="yearly")
        model_std = list(ds_all_flux_scaled.keys())

        ### Define plotting colors
        model_colors = set_model_colors(models_std)
        model_labels = set_model_labels(models_std,config_data,True)

        # 3) Plot annual country fluxes from 2008 to 2023 from intem_longrun and combined from 3 std_run
        fig, res_dict = plot_country_flux(ds_all_flux_scaled,
                        species = species,
                        model_colors = model_colors,
                        model_labels = model_labels,
                        start_date = start_date,
                        end_date = end_date,
                        resample = None,
                        **kwargs_country_flux_general,
                        **kwargs_country_flux_annual_species)
        
        plot_name = f"{species}_country_flux_annual_longrun_{region}.png"
        full_path = os.path.join(output_path, plot_name)
        fig.savefig(full_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        plt.close()

        # Store results for .csv and table
        comb = res_dict[region]["combined"]
        try:
            inv = res_dict[region]["inventory"]
        except:
            print("NO INVENTORY FOUND")
            inv = {
                "time": comb["time"],
                "value": np.array(
                    [
                        np.NaN,
                    ]
                    * len(comb["time"])
                ),
            }

        if "pfc" in species:
            n_digits = 2
        else:
            n_digits = 1

        tmp = {
            "species": [
                species,
            ]
            * 2,
            "source": ["NIR " + inventory_years[0], "PARIS mean"],
        }
        for it, time in enumerate(comb["time"].astype("datetime64[Y]")):
            paris_val = f"{comb['mean'][it]:.{n_digits}f} \\pm {(comb['max'][it]-comb['min'][it])/2:.{n_digits}f}"
            inv_val = inv["value"][inv["time"].astype("datetime64[Y]") == time]
            if len(inv_val) == 1:
                tmp[str(time)] = [f"{inv_val[0]:.{n_digits}f}", paris_val]
            else:
                tmp[str(time)] = [None, paris_val]
        annual_res_list.append(pd.DataFrame(tmp))

    ### F-gases
    end_date = "2024-01-01"

    print("\n--- PLOTTING COUNTRY FLUXES FOR ALL F-GASES ---")
    for species in annex_config.annual_species:

        start_date = annex_config.start_date_fgases[plot_area]
        start_year = start_date.split("-")[0]
        if species == "hfc4310mee" and int(start_year) < 2011:
            start_date = "2011-01-01"  # Fix for InTEM longrun which is zero in 2010
        
        models_std = define_model_list(models_country_fluxes, species, config_data)

        ds_all_flux = read_model_output(annex_config.data_dir,"flux",species,models_std,config_data,period="monthly")
        ds_all_flux_scaled = slice_flux(ds_all_flux,config_data,start_date,end_date,species=species,country_flux_units_print=country_flux_units_print)
        
        ### Define plotting colors
        model_colors = set_model_colors(models_std)
        model_labels = set_model_labels(models_std,config_data,True)

        # 4) Plot annual country fluxes from 2008 to 2023 from intem_longrun and combined from 3 std_run
        fig, res_dict = plot_country_flux(ds_all_flux_scaled,
                        species = species,
                        model_colors = model_colors,
                        model_labels = model_labels,
                        start_date = start_date,
                        end_date = end_date,
                        resample = None,
                        **kwargs_country_flux_general,
                        **kwargs_country_flux_annual_species)

        plot_name = f"{species}_country_flux_annual_longrun_{region}.png"
        full_path = os.path.join(output_path, plot_name)
        fig.savefig(full_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        plt.close()

        # Store results for .csv and table
        comb = res_dict[region]["combined"]
        try:
            inv = res_dict[region]["inventory"]
        except:
            print("NO INVENTORY FOUND")
            inv = {
                "time": comb["time"],
                "value": np.array(
                    [
                        np.NaN,
                    ]
                    * len(comb["time"])
                ),
            }

        tmp = {
            "species": [
                species,
            ]
            * 2,
            "source": ["NIR " + inventory_years[0], "PARIS mean"],
        }

        if species in ["hfc23", "hfc32", "hfc125", "hfc134a", "hfc143a", "sf6"]:
            n_digits = 2
        else:
            n_digits = 3

        for it, time in enumerate(comb["time"].astype("datetime64[Y]")):
            paris_val = f"{comb['mean'][it]:.{n_digits}f} \\pm {(comb['max'][it]-comb['min'][it])/2:.{n_digits}f}"
            inv_val = inv["value"][inv["time"].astype("datetime64[Y]") == time]
            if len(inv_val) == 1:
                tmp[str(time)] = [f"{inv_val[0]:.{n_digits}f}", paris_val]
            else:
                tmp[str(time)] = [None, paris_val]
        annual_res_list.append(pd.DataFrame(tmp))

#     ### Models for spatial maps
#     models = models_spatial_maps

#     # Settings for average posterior
#     cmap = "viridis"
#     c_border = "floralwhite"
#     var = "flux_total_posterior"
#     plot_combined = True
#     chop_by = "year"

#     start_date = "2018-01-01"
#     all_species = annex_config.monthly_species + annex_config.annual_species

#     # All species
#     print("\n--- PLOTTING MEAN POSTERIOR MAP FOR ALL SPECIES ---")
#     for species in all_species:

#         ds_all_flux = {}
#         ds_all_flux_scaled = {}
#         models_std = []

#         if species in annex_config.fluxlim_percentiles[plot_area].keys():
#             set_fluxlim_percentile = annex_config.fluxlim_percentiles[plot_area][species]
#         else:
#             set_fluxlim_percentile = None

#         if species == "hfc4310mee":
#             end_date = "2023-01-01"  # NOTE: no 2023 results for HFC-4310mee
#             dt = 5
#         else:
#             end_date = "2024-01-01"
#             dt = 6

#         # NOTE: easy fix while there are no Rhime results for N2O
#         if species == "n2o":
#             models = ["intem", "elris"]
#         else:
#             models = models_spatial_maps

#         ### Read and scale fluxes
#         for m, model in enumerate(models):

#             m0 = model.split("_")[0]

#             model_read = f'{m0}_{s_data[species]["std_run"][m0]}'
#             models_std.append(model_read)

#             # use model_read instead of model
#             ds_all_flux[model_read] = func.read_flux(
#                 annex_config.data_dir,
#                 species,
#                 [model_read],
#                 s_data,
#                 m_data,
#                 period_override=period_override,
#             )[model_read]
#             ds_all_flux_scaled[model_read] = func.slice_flux(
#                 {model_read: ds_all_flux[model_read]},
#                 start_date,
#                 end_date,
#                 s_data,
#                 scale_units=True,
#                 convert_flux_units=convert_flux_units,
#                 species=species,
#             )[model_read]

#         # 5) Plot spatial map of the posterior fluxes averaged between 2018 and 2023 (combined from 3 std_run)
#         fig = func.plot_spatial_flux_per_timestamp(
#             ds_all_flux_scaled,
#             species,
#             plot_area,
#             end_date,
#             s_data,
#             m_data,
#             cmap=cmap,
#             c_border=c_border,
#             var=var,
#             plot_combined=plot_combined,
#             annex_mode=annex_config.annex_mode,
#             chop_by=chop_by,
#             dt=dt,
#             period_override=period_override,
#             plot_site_locations=plot_site_locations,
#             plot_point_markers=plot_point_markers,
#             set_fluxlim=set_fluxlim,
#             set_fluxlim_percentile=set_fluxlim_percentile,
#             plot_inversion_grid_flux=plot_inversion_grid_flux,
#         )

#         plot_name = f"{species}_posterior_map_{region}.png"
#         full_path = os.path.join(output_path, plot_name)
#         fig.savefig(full_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
#         plt.close()

#     # Settings for seasonal difference to the mean
#     cmap = "coolwarm"
#     c_border = "dimgrey"
#     chop_by = "season"
#     dt = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
#     var = "posterior_mean_diff"
#     plot_combined = True

#     # CH4 and N2O
#     start_date = "2018-01-01"
#     end_date = "2024-01-01"

#     print("\n--- PLOTTING SEASONAL POSTERIOR MAP FOR CH4/N2O ---")
#     for species in annex_config.monthly_species:

#         ds_all_flux = {}
#         ds_all_flux_scaled = {}
#         models_std = []

#         if species in annex_config.fluxlim_percentiles[plot_area].keys():
#             set_fluxlim_percentile = annex_config.fluxlim_percentiles[plot_area][species]
#         else:
#             set_fluxlim_percentile = None

#         # NOTE: easy fix while there are no Rhime results for N2O
#         if species == "n2o":
#             models = ["intem", "elris"]
#         else:
#             models = models_spatial_maps

#         ### Read and scale fluxes
#         for m, model in enumerate(models):

#             m0 = model.split("_")[0]

#             model_read = f'{m0}_{s_data[species]["std_run"][m0]}'
#             models_std.append(model_read)

#             # use model_read instead of model
#             ds_all_flux[model_read] = func.read_flux(
#                 annex_config.data_dir,
#                 species,
#                 [model_read],
#                 s_data,
#                 m_data,
#                 period_override=period_override,
#             )[model_read]
#             ds_all_flux_scaled[model_read] = func.slice_flux(
#                 {model_read: ds_all_flux[model_read]},
#                 start_date,
#                 end_date,
#                 s_data,
#                 scale_units=True,
#                 convert_flux_units=convert_flux_units,
#                 species=species,
#             )[model_read]

#         # 6) Plot spatial maps of the seasonal posterior fluxes (averaged between 2018 and 2023) subtracted by the mean (combined from 3 std_run)
#         fig = func.plot_spatial_flux_per_timestamp(
#             ds_all_flux_scaled,
#             species,
#             plot_area,
#             end_date,
#             s_data,
#             m_data,
#             cmap=cmap,
#             c_border=c_border,
#             var=var,
#             plot_combined=plot_combined,
#             annex_mode=annex_config.annex_mode,
#             chop_by=chop_by,
#             dt=dt,
#             period_override=period_override,
#             plot_site_locations=plot_site_locations,
#             plot_point_markers=plot_point_markers,
#             set_fluxlim=set_fluxlim,
#             set_fluxlim_percentile=set_fluxlim_percentile,
#             plot_inversion_grid_flux=plot_inversion_grid_flux,
#         )

#         plot_name = f"{species}_seasonal_map_{region}.png"
#         full_path = os.path.join(output_path, plot_name)
#         fig.savefig(full_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
#         plt.close()

#     print("\n--- ALL PLOTS GENERATED SUCCESSFULLY! ---")

#     print("\n\n\n--- GENERATING TABLES ---")
#     annual_res = pd.concat(annual_res_list).reset_index(drop=True).fillna(value=" ")

#     print("\n\nTABLE HFC\n\n")
#     hfc_res = annual_res[
#         annual_res.species.apply(lambda x: x[:3].lower() == "hfc")
#     ].copy()
#     hfc_res["species"] = hfc_res.species.apply(lambda x: x.replace("hfc", "HFC-"))
#     make_table(hfc_res, f"{output_path}/hfc_res_{region}.tex")
#     hfc_res.to_csv(f"{output_path}/hfc_res_{region}.csv", index=False)

#     print("\n\nTABLE PFC\n\n")
#     pfc_res = annual_res[
#         annual_res.species.apply(lambda x: x[:3].lower() == "pfc" or x.lower() == "cf4")
#     ].copy()
#     pfc_res["species"] = pfc_res.species.apply(lambda x: x.replace("pfc", "PFC-"))
#     pfc_res["species"] = pfc_res.species.apply(lambda x: x.replace("cf4", "PFC-14"))
#     make_table(pfc_res, f"{output_path}/pfc_res_{region}.tex")
#     pfc_res.to_csv(f"{output_path}/pfc_res_{region}.csv", index=False)

#     print("\n\nTABLE main gases\n\n")

#     main_gases_res = annual_res[
#         annual_res.species.isin(["ch4", "n2o", "sf6", "all_pfc", "all_hfc"])
#     ].copy()
#     main_gases_res["species"] = main_gases_res.species.apply(
#         lambda x: x.upper().replace("ALL_", "Total ")
#     )
#     make_table(main_gases_res, f"{output_path}/main_gases_res_{region}.tex")
#     main_gases_res.to_csv(f"{output_path}/main_gases_res_{region}.csv", index=False)

#     print("\n--- TABLES GENERATED SUCCESSFULLY! ---")
#     return annual_res


def make_table(
    df,
    output_path,
    descriptive_cols=["species", "source"],
    hline_place={"source": "PARIS mean"},
):
    if "hfc" in output_path:
        species = "HFCs"
    elif "pfc" in output_path:
        species = "PFCs"
    if "main_gases" in output_path:
        species = "the main greenhouse gases of focus"
    # Set latex Table env and number of cols
    tmp = output_path.split("/")[-1].split(".")[0]
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
