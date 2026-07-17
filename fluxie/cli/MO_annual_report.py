import pandas as pd
import numpy as np
import os
import logging
from fluxie.cli.utils_annex_plot import create_str_dataframe

logger = logging.getLogger(__name__)

def make_AR_table(df: pd.DataFrame,
                  output_dir,
                  output_name,
                  s_data: dict[str,dict],
                  regions: list[str],
                  start_date: str | list[str],
                  country_flux_units_print: str,
                  inventory_years: str | int,
                  species: str | None =  None,
                  include_inventory_uncert: bool = True,
                  include_inventory: bool = True,
                  n_digits: int = 2,
                  save_latex: bool = True,
                  save_csv: bool = True,
                  sectors: list[str] | str = 'total',
                  inv_model: str = 'Inversion'):
    """
    Function to create a table of inventory and InTEM emission estimates, in the format
    required for the Met Office annual report.
    
    Args:
        df: dataframe containing inventory and InTEM results (produced by plot_country_flux)
        output_dir: directory where the output table will be saved
        output_name: name of the output table file (the full filename will be: '{species}_{output_name}_table.tex')
        inventory_years: years of inventory data to include
        species: specific species to include (optional)
        s_data: dictionary containing species information

    Returns:
    A text file containing the table, saved to the specified output directory.
    """
    
    if type(sectors) == str:
        sectors = [sectors]
    
    all_model_names = []
    for m in np.unique(df[sectors[0]]['model'].values):
        if 'inventory' not in m:
            all_model_names.append(m)

    res_combined = {}
    for s in sectors:
        res_combined[s] = df[s].replace({'model':all_model_names},inv_model)
    
    all_read_in = [inv_model]
    if include_inventory:
        all_read_in.append(f'inventory_{max(inventory_years)}')

    species_list = [species]
    if type(start_date) != list: start_date = [start_date]

    annual_res = {}
    for s in sectors:
        annual_res[s] = pd.concat([res_combined[s]], ignore_index=True)
    
    logger.warning("If you get an 'Index contains duplicate entries' error, "
                        "this may be because the two models overlap in time. To fix this "
                        "set start_date and end_date and lists of dates with no overlap.")
    
    all_sp_res = {}
    
    for r in regions:
        
        all_sp_res[r] = {}
        
        for s in sectors:
        
            all_sp_res[r][s] = create_str_dataframe(
                annual_res[s],
                inventory_years,
                species_list,
                model=all_read_in,
                table_start_date=min(start_date),
                region=r,
                include_inventory_uncert=include_inventory_uncert,
                n_digits=2,
                sector=s)
    
    all_years = [t for t in all_sp_res[r][sectors[0]] if t not in ['species','units','source']]

    species_print = s_data[species]['species_print']
    units_print = country_flux_units_print.replace('-1',"$^{-1}$")

    caption = "\n \\caption{" + f"{species_print} emission {units_print} estimates with 1$\sigma$ uncertainty" + "}"
    caption_csv = f"CH4 emission {units_print.replace('$^{-1}$','-1')} estimates with 1-sigma uncertainty\n"
    label = "\n \\label{" + f"{species}_emit" + "}"

    region_title = ""
    type_title = "Years"
    type_title_csv = "Years"
    fill_line = ""
    cols_line = "\n {\\begin{tabular}{|l|"

    for r in regions:
        if r == 'NW_EU2':
            region_name = 'NWEU'
        else:
            region_name = r
            
        for s in sectors:
            
            if sectors == ['total']:
                sector_name = ''
                sector_name_csv = ''
                
            else:
                sector_name = f' {s.capitalize()}'
                sector_name_csv = f'_{s.capitalize()}'
                
            if include_inventory: 
                type_title += f" & Inventory{sector_name} & {inv_model}{sector_name}"
                if include_inventory_uncert:
                    type_title_csv += f",Inventory{sector_name_csv},Inventory{sector_name_csv}_uncert,InTEM{sector_name_csv},InTEM{sector_name_csv}_uncert"
                else:
                    type_title_csv += f",Inventory{sector_name_csv},{inv_model}{sector_name_csv}"
                region_title += f" & {region_name} & {region_name}"
                fill_line += " & &"
                
            else:
                type_title += f" & {inv_model}{sector_name}"
                if include_inventory_uncert:
                    type_title_csv += f",{inv_model}{sector_name_csv},{inv_model}{sector_name_csv}_uncert"
                else:
                    type_title_csv += f",{inv_model}{sector_name_csv}"
                region_title += f" & {region_name}"
                fill_line += " &"

    n_cols = len(regions)*len(sectors)

    if include_inventory:
        cols_line += f"{'c'*n_cols}|{'c'*n_cols}|}}"
    else:
        cols_line += f"{'c'*n_cols}|}}"
    
    region_title = "\n " + region_title.strip() + " \\\\"
    type_title = "\n " + type_title.removesuffix('& ') + "\\\\"
    type_title_csv = "\n" + type_title_csv + "\n"
    fill_line = "\n " + fill_line.strip() + " \\\\"

    header = ("\\begin{table}[H]" 
            + "\n \\captionsetup{width=0.9\linewidth}"
            + "\n \\centering"
            + caption
            + label
            + cols_line
            + "\n \\hline"
            + region_title
            + type_title
            + "\n \\hline"
            + fill_line
            )

    footer = ("\n \\hline"
            + "\n \\end{tabular}"
            + "\n }"
            + "\n \\end{table}")

    all_lines = ""
    all_lines_csv = ""
    
    for i,t in enumerate(all_years):
        new_line = f"{t}"
        new_line_csv = f"{t}"
        for r in regions:
            for s in sectors:
                if r != 'UK' and int(t) > 2023:
                    if include_inventory:
                        new_line += f"&  &  {all_sp_res[r][s][t].values[0]}".replace("\\pm","${\\pm}$")
                        new_line_csv += f",,{all_sp_res[r][s][t].values[0]}".replace(" \\pm ",",")
                        
                    else:
                        new_line += f"&  {all_sp_res[r][s][t].values[0]}".replace("\\pm","${\\pm}$")
                        new_line_csv += f",{all_sp_res[r][s][t].values[0]}".replace(" \\pm ",",")
                        
                else:
                    if include_inventory:
                        if include_inventory_uncert == True:
                            extra_commas = ',,'
                        else:
                            extra_commas = ','
                        new_line += f"&  {all_sp_res[r][s][t].values[1]}&  {all_sp_res[r][s][t].values[0]}".replace("\\pm","${\\pm}$")
                        if all_sp_res[r][s][t].values[1] == ' ':
                            new_line_csv += f",{extra_commas}{all_sp_res[r][s][t].values[0]}".replace(" \\pm ",",")
                        else:
                            new_line_csv += f",{all_sp_res[r][s][t].values[1]},{all_sp_res[r][s][t].values[0]}".replace(" \\pm ",",")
                        
                    else:
                        new_line += f"&  {all_sp_res[r][s][t].values[0]}".replace("\\pm","${\\pm}$")
                        new_line_csv += f",{all_sp_res[r][s][t].values[0]}".replace(" \\pm ",",")
                                    
        new_line = "\n" + new_line + " \\\\"
        new_line_csv = "\n" + new_line_csv

        all_lines += new_line
        all_lines_csv += new_line_csv
        
    outlines = header + all_lines + footer
    outlines_csv = caption_csv + type_title_csv + all_lines_csv

    output_path = os.path.join(output_dir,f'{species}_{output_name}_table.tex')
    output_path_csv = os.path.join(output_dir,f'{species}_{output_name}_table.txt')
    
    if save_latex == True:
        with open(output_path,'w') as f:
            f.writelines(outlines)

        print(f'Table saved to : {output_path}')

    if save_csv == True:
        with open(output_path_csv,'w') as f:
            f.writelines(outlines_csv)
        
        print(f'Table saved to : {output_path_csv}')
        
    return outlines_csv