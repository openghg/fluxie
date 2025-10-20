import pandas as pd
import numpy as np

def dict_to_str_dataframe(
    res: dict, inventory_years: list | str | int, species: str
) -> pd.DataFrame:
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
        paris_val = f"{comb['mean'][it]:.{n_digits}f} $\\pm$ {(comb['max'][it]-comb['min'][it])/2:.{n_digits}f}"
        inv_val = inv["value"][inv["time"].astype("datetime64[Y]") == time]
        if len(inv_val) == 1:
            output[str(time)] = [f"{inv_val[0]:.{n_digits}f}", paris_val]
        else:
            output[str(time)] = [None, paris_val]

    return pd.DataFrame(output)

def region_dataframe(
    res: dict, regions:dict, inventory_years: list | str | int, species: str
) -> pd.DataFrame:
    dfs = []
    for r in regions:
        df = dict_to_str_dataframe(res[r], inventory_years, species)
        df["region"] = r
        df = df[["region"] + [c for c in df.columns if c != "region"]]
        dfs.append(df)

    return(pd.concat(dfs, ignore_index=True))

def summarize_models_yearly_dict(models_dict, model_list):
    """
    Summarize selected models per year and return in dict-of-arrays format.
    
    Parameters
    ----------
    models_dict : dict
        Dictionary of models like res['NW_EU2'].
    model_list : list of str
        Names of models to include.
    
    Returns
    -------
    summary_dict : dict
        Dictionary with keys 'time', 'mean', 'min', 'max' and NumPy arrays.
    """
    all_records = []

    for model_name, data in models_dict.items():
        if model_name not in model_list:
            continue

        df = pd.DataFrame({
            "year": pd.to_datetime(data["time"]).year,
            "mean": data["mean"],
            "min": data["min"],
            "max": data["max"]
        })
        all_records.append(df)

    if not all_records:
        raise ValueError("No models selected!")

    combined = pd.concat(all_records, ignore_index=True)

    # Aggregate per year across all models
    summary_yearly = combined.groupby("year").agg({
        "mean": "mean",
        "min": "min",
        "max": "max"
    })

    # Convert the year to a datetime (e.g., July 2nd) to match original format
    times = pd.to_datetime(summary_yearly.index.astype(str) + "-07-02")  # adjust month/day as needed

    summary_dict = {
        "time": times.to_numpy(dtype="datetime64[ns]"),
        "mean": summary_yearly["mean"].to_numpy(),
        "min": summary_yearly["min"].to_numpy(),
        "max": summary_yearly["max"].to_numpy()
    }

    return summary_dict

def summarize_models_region(
    res: dict, inventory_years: list | str | int, regions: list, models: list
) -> dict:

    inv = f"inventory_{inventory_years}"
    region_dict = {}
    for r in regions:
        region_dict[r] = {}
        region_dict[r][inv] = res[r][inv]
        region_dict[r]['combined'] = summarize_models_yearly_dict(res[r], models)

    return region_dict

def extract_number(val):
    if pd.isna(val):
        return np.nan
    return float(str(val).split()[0])

def subset_with_mean(df, region=None, species=None, source=None, start_year=2013, end_year=2017):
    """
    Filter dataframe by region, species, source, and compute mean emissions 
    over a given year range (inclusive).
    
    Parameters:
        df : pd.DataFrame
            Input dataframe
        region : str or None
            Region to filter (if None, keep all)
        species : str or None
            Species to filter (if None, keep all)
        source : str or None
            Source to filter (if None, keep all)
        start_year : int
            First year in averaging
        end_year : int
            Last year in averaging (inclusive)
            
    Returns:
        pd.DataFrame
            Filtered dataframe with year columns and a new column "MEAN"
    """
    
    # --- Clean numeric year columns ---
    year_cols = [col for col in df.columns if col.isdigit()]
    df_clean = df.copy()
    df_clean[year_cols] = df_clean[year_cols].map(extract_number)
    
    # --- Apply filters ---
    mask = pd.Series(True, index=df_clean.index)
    if region is not None:
        mask &= df_clean["region"] == region
    if species is not None:
        mask &= df_clean["species"] == species
    if source is not None:
        mask &= df_clean["source"] == source
        
    subset = df_clean[mask].copy()
    
    # --- Select years of interest ---
    years_to_average = [str(y) for y in range(start_year, end_year + 1)]
    id_cols = ["region", "species", "source"]
    subset = subset[id_cols + years_to_average]
    
    # --- Compute mean ---
    subset["MEAN"] = subset[years_to_average].mean(axis=1).round(1)
    
    return subset
