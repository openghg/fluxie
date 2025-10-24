import numpy as np
import pandas as pd

from fluxy.cli.utils_annex_plot import dict_to_str_dataframe


def make_test_data():
    time = np.array([np.datetime64("1900-01-01"), np.datetime64("2000-01-01")])

    posterior_data = {
        "WONDERLAND": {
            "combined": {
                "time": time,
                "mean": np.array([1.0, 1.0]),
                "min": np.array([0.0, 0.0]),
                "max": np.array([2.0, 2.0]),
            }
        }
    }
    prior_data = {
        "WONDERLAND": {
            "combined": {
                "time": time,
                "mean": np.array([0.0, 0.5]),
            }
        }
    }
    inventory_data = {
        "WONDERLAND": {
            "inventory_2000": {
                "time": time,
                "value": np.array([0.0, 0.1]),
            }
        }
    }

    res = {
        "posterior": posterior_data,
        "prior": prior_data,
        "inventory": inventory_data,
    }

    return res


def test_dict_to_str_dataframe():
    res = make_test_data()

    expected = pd.DataFrame(
        {
            "species": ["ch4", "ch4"],
            "source": ["NIR 2000", "PARIS mean"],
            "1900": ["0", "1 \\pm 1"],
            "2000": ["0", "1 \\pm 1"],
        }
    )

    output = dict_to_str_dataframe(res, "2000", "ch4")
    assert all(output == expected)
