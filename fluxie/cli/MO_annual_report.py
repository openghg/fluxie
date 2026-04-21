import pandas as pd
import numpy as np
import os
from fluxie.cli.utils_annex_plot import create_str_dataframe

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
                  n_digits: int = 2):
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
    
    all_model_names = []
    for m in np.unique(df['model'].values):
        if 'inventory' not in m:
            all_model_names.append(m)
    
    #res_combined = df.replace({'model':['InTEM yearly','InTEM monthly']},'InTEM')
    res_combined = df.replace({'model':[all_model_names]},'InTEM')

    species_list = [species]
    if type(start_date) != list: start_date = [start_date]

    annual_res = pd.concat([res_combined], ignore_index=True)
    #annual_res = pd.concat([df], ignore_index=True)
    

    all_sp_res = {}

    for r in regions:
        all_sp_res[r] = create_str_dataframe(
            annual_res,
            inventory_years,
            species_list,
            #model=['InTEM',f'inventory_{max(inventory_years)}'],
            model=['InTEM',f'inventory_{max(inventory_years)}'],
            table_start_date=min(start_date),
            region=r,
            include_inventory_uncert=True,
            n_digits=2)
    
    all_years = [t for t in all_sp_res[r] if t not in ['species','units','source']]

    species_print = s_data[species]['species_print']
    units_print = country_flux_units_print.replace('-1',"$^{-1}$")
    
    print(units_print)

    caption = "\n \\caption{" + f"{species_print} emission {units_print} estimates with 1$\sigma$ uncertainty" + "}"
    label = "\n \\label{" + f"{species}_emit" + "}"

    region_title = ""
    type_title = "Years"
    fill_line = ""

    for r in regions:
        if r == 'NW_EU2':
            region_name = 'NWEU'
        else:
            region_name = r
            
        region_title += f" & {region_name} & {region_name}"
        type_title += f" & Inventory & InTEM"
        fill_line += " & &"
    
    region_title = "\n " + region_title.strip() + " \\\\"
    type_title = "\n " + type_title.removesuffix('& ') + "\\\\"
    fill_line = "\n " + fill_line.strip() + " \\\\"

    header = ("\\begin{table}[H]" 
            + "\n \\captionsetup{width=0.9\linewidth}"
            + "\n \\centering"
            + caption
            + label
            + "\n {\\begin{tabular}{|l|cc|cc|}"
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
    
    #return all_sp_res
    
    for i,t in enumerate(all_years):
        print(t)
        new_line = f"{t}"
        for r in regions:
            if r != 'UK' and int(t) > 2023:
                new_line += f"&  &  {all_sp_res[r][t].values[0]}".replace("\\pm","${\\pm}$")

            else:
                new_line += f"&  {all_sp_res[r][t].values[1]}&  {all_sp_res[r][t].values[0]}".replace("\\pm","${\\pm}$")
                
        new_line = "\n" + new_line + " \\\\"
        all_lines += new_line
        
    outlines = header + all_lines + footer

    output_path = os.path.join(output_dir,f'{species}_{output_name}_table.tex')

    with open(output_path,'w') as f:
        f.writelines(outlines)
        
    print(f'Table saved to : {output_path}')
    