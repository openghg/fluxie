import numpy as np
import pandas as pd

from fluxy.cli.utils_annex_plot import create_str_dataframe


def make_test_data():
    time = np.array([np.datetime64("1900-01-01"), np.datetime64("2000-01-01")])
    sector = "flower"
    country = "WONDERLAND"
    species = "ch4"
    inv_year = 2000
    model = "CAT"

    inv_data = pd.DataFrame({"type":["inventory",]*time.size,
                            "model":[f"inventory_{inv_year}",]*time.size,
                            "sector":[sector,]*time.size,
                            "country":[country,]*time.size,
                            "species":[species,]*time.size,
                            "time": time.astype("datetime64[ns]"),
                            "mean_val": np.array([0.0, 0.1]),})

    prior_data = pd.DataFrame({"type":["prior",]*time.size,
                            "model":[model,]*time.size,
                            "sector":[sector,]*time.size,
                            "country":[country,]*time.size,
                            "species":[species,]*time.size,
                            "time": time.astype("datetime64[ns]"),
                            "mean_val": np.array([0.0, 0.5]),})
    
    post_data = pd.DataFrame({"type":["posterior",]*time.size,
                            "model":[model,]*time.size,
                            "sector":[sector,]*time.size,
                            "country":[country,]*time.size,
                            "species":[species,]*time.size,
                            "time": time.astype("datetime64[ns]"),
                            "mean_val": np.array([1.0, 1.0]),
                            "min_unc": np.array([0.0, 0.0]),
                            "max_unc": np.array([2.0, 2.0]),})
    
    res = pd.concat([inv_data, prior_data, post_data], ignore_index=True)

    return res

def test_dict_to_str_dataframe():
    data = make_test_data()

    expected = pd.DataFrame(
        {
            "species": ["ch4", "ch4"],
            "source": ["CAT", "NIR 2000"],
            "1900": ["1.0 \\pm 1.0", "0.0"],
            "2000": ["1.0 \\pm 1.0", "0.1"],
        }
    )

    output, unit = create_str_dataframe(data,"2000","ch4",model="CAT")
    
    assert (output==expected).values.all()
    assert unit == '$\\rm{TgCO}_{2}\\rm{-eq} \\cdot \\rm{yr}^{-1}$'

    for var in ["mean_val", "min_unc", "max_unc"]:
        data[var] *= 1e-2
    
    expected = pd.DataFrame(
        {
            "species": ["ch4", "ch4"],
            "source": ["CAT", "NIR 2000"],
            "1900": ["10.0 \\pm 10.0", "0.0"],
            "2000": ["10.0 \\pm 10.0", "1.0"],
        }
    )
    
    output, unit = create_str_dataframe(data,"2000","ch4",model="CAT")
    
    assert (output==expected).values.all()
    assert unit == '$\\rm{GgCO}_{2}\\rm{-eq} \\cdot \\rm{yr}^{-1}$'