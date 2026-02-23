from fluxy.io import read_model_output
from fluxy.operators.select import slice_flux
from fluxy.config import set_model_colors, set_model_labels
from fluxy.plots.flux_timeseries import add_xlims_and_ticks, get_unit, add_title, create_fig_and_axes, prepare_data_to_plot, add_posterior_plot, add_prior_plot, add_inventory_barplot, add_ylabel, add_legend
from fluxy.plots.utils import get_units
from fluxy.operators.regions import extract_region_flux

import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.dates import YearLocator, MonthLocator
from functools import partial

def prepare_flux_data(
    species, 
    models, 
    config_data,
    data_dir, 
    start_date, 
    end_date, 
    read_standard_run, 
    country_flux_units_print
) -> dict[str, xr.Dataset]:

    # Read model output data
    ds_all_flux = read_model_output(
        species = species, 
        models = models, 
        data_dir = data_dir, 
        file_type = 'flux', 
        config_data = config_data,
        read_standard_run = read_standard_run)

    # Slice the data for the specified time range
    ds_all_flux_scaled = slice_flux(
        ds_all = ds_all_flux,
        config_data = config_data,
        start_date = start_date,
        end_date = end_date,
        species = species,
        country_flux_units_print = country_flux_units_print)

    return ds_all_flux_scaled

def plot_multi_species_flux(
    gases, 
    region, 
    models, 
    data_dir, 
    start_date, 
    end_date, 
    config_data,
    read_standard_run, 
    country_flux_units_print,
    annex_mode,
    plot_inventory,
    add_prior,
    add_prior_unc,
    set_global_leg,
    plot_separate,
    plot_separate_unc,
    plot_combined,
    plot_combined_unc,
    combined_models_dict,
    rolling_mean,
    add_vline,
    add_Gg_per_year_axis,
) -> Figure:

    model_colors = set_model_colors(models)
    model_labels = set_model_labels(models, config_data, get_labels_from_file=False)

    s_data = config_data.get("species_info", {})
    r_data = config_data.get("regions_info", {})

    sector='total'

    plotted_data_df = pd.DataFrame()

    plot_separate_unc = np.any(plot_separate) if plot_separate_unc is None else plot_separate_unc
    plot_combined_unc = np.any(plot_combined) if plot_combined_unc is None else plot_combined_unc

    fig, axes = create_fig_and_axes(len(gases))

    # Iterate over each axe and gas
    for i, (ax, gas) in enumerate(zip(axes, gases)):

        # Prepare data for the specific gas
        ds_all = prepare_flux_data(
            species = gas, 
            models = models, 
            config_data = config_data,
            data_dir = data_dir, 
            start_date = start_date, 
            end_date = end_date, 
            read_standard_run = read_standard_run, 
            country_flux_units_print = country_flux_units_print)

        unit = get_unit(ds_all)
        ds_all = {k: ds.sel(time=slice(start_date, end_date)) for k, ds in ds_all.items()}

        # Prepare model results
        ds_all_region = extract_region_flux(ds_all, region, r_data, sector=sector)
        ds_to_plot = prepare_data_to_plot(
            ds_all_region=ds_all_region,
            model_labels=model_labels,
            model_colors=model_colors,
            plot_separate=plot_separate,
            plot_combined=plot_combined,
            combined_models_dict=combined_models_dict,
            rolling_mean=rolling_mean,
        )

        # Plot posterior and prior (if requested)
        for m, ds_region in ds_to_plot.items():
            highlighted_post = annex_mode
            add_post_unc = (("combined" in m) & plot_combined_unc) |  (("combined" not in m) & plot_separate_unc)
            posterior_df = add_posterior_plot(
                ax, ds_region, highlighted_post, add_post_unc
            )
            plotted_data_df = pd.concat([plotted_data_df, posterior_df], ignore_index=True)
            
            if add_prior and "NAME" in m:
                prior_df = add_prior_plot(
                    ax, ds_region, annex_mode, add_prior_unc
                )
                plotted_data_df = pd.concat([plotted_data_df, prior_df], ignore_index=True)

        # Plot inventory
        if plot_inventory:
            inventory_df = add_inventory_barplot(
                ax,
                data_dir,
                region,
                gas,
                start_date,
                end_date,
                unit,
                s_data,
                r_data,
                None,
                "UNFCCC_inventory",
                sector,
                annex_mode,
            )
            plotted_data_df = pd.concat([plotted_data_df, inventory_df], ignore_index=True)

        # add vertical lines if any
        if add_vline is not None:
            for vline_date in add_vline:
                ax.axvline(
                    np.datetime64(vline_date),
                    color="grey",
                    linestyle="dotted",
                    linewidth=2.5,
                )

        # Set y label
        add_ylabel(ax, s_data, gas, sector, unit)

        # Set x ticks
        if i in [4, 5]:
            ax.tick_params(labelbottom=True)   # re-enable x tick labels
            ax.xaxis.get_label().set_visible(True)

        # Set grid
        ax.grid(visible=True, which="major", alpha=0.4)

        # add second axis
        if add_Gg_per_year_axis:
            GWP = s_data.get(gas, {}).get("gwp", None)
            # Conversion functions
            def Tg_to_Gg(y, GWP):
                return y * 1000 / GWP

            def Gg_to_Tg(y, GWP):
                return y * GWP / 1000

            # Secondary y-axis
            sec_color = 'darkred' #(196/255, 112/255, 138/255)
            secax = ax.secondary_yaxis(
                'right',
                functions=(partial(Tg_to_Gg, GWP), partial(Gg_to_Tg, GWP))
            )

            secax.set_ylabel(
                f"{s_data.get(gas, {}).get('species_print', gas)} (Gg yr$^{{-1}}$)",
                rotation=270, labelpad=20, color=sec_color
            )

            secax.tick_params(axis='y', colors=sec_color)
            secax.spines['right'].set_color(sec_color)
            secax.spines['right'].set_linewidth(1.5)

        # Set ax title
        ax.set_title(f"{s_data.get(gas, {}).get('species_print', gas)}")

    yearly_freq = (
        "yearly" in [ds.attrs["frequency"] for ds in ds_to_plot.values()]
        or resample == "year"
    )
    add_xlims_and_ticks(axes[-1], yearly_freq, plotted_data_df, aggreg_month=False)

    # add_legend(fig, set_global_leg, annex_mode, plot_inventory)
    # Hide remaining axes (but keep layout)
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    # Add legend in the first hidden subplot
    handles, labels = fig.axes[0].get_legend_handles_labels()
    legend_ax = axes[i+1]  # e.g. axes[7]
    legend_ax.legend(
        handles,
        labels,
        loc="upper left",
        # ncol=(
        #     len(labels) if len(labels) <= 6 else len(labels) // 2 + len(labels) % 2
        # ),
        borderpad=0.4,
        columnspacing=1.0,
    )

    return fig

#################################################################################################

def sig_figs(x, n=3):
    return float(f"{x:.{n}g}")

def reshape_emissions(initial_df):
    """
    Restructure emissions dataframe and compute:
      - yearly formatted values (mean ± unc)
    #   - Period mean values (mean ± mean_unc)
    #     * Mean_2013_2016
    #     * Mean_2017_2023
    #   - Pct_change_%
      - Total mean values (2013 to 2024)

    Parameters
    ----------
    initial_df : pandas.DataFrame
        Must contain columns:
        ['country','species','model','time','mean_val','min_unc','max_unc']

    Returns
    -------
    pandas.DataFrame
        Restructured dataframe in wide format.
    """

    import pandas as pd
    import numpy as np

    df = initial_df.copy()

    # Extract year
    df["year"] = df["time"].dt.year

    # Compute ± uncertainty
    df["pm"] = (df["max_unc"] - df["min_unc"]) / 2

    # Format "mean ± unc"
    def format_val(row):
        if pd.isna(row["mean_val"]):
            return np.nan
        if pd.isna(row["pm"]):
            return f"{row['mean_val']:.2g}"
        return f"{row['mean_val']:.2g} ± {row['pm']:.2g}"

    df["formatted"] = df.apply(format_val, axis=1)

    # Minimal table
    df_small = df[["country", "species", "model", "year", "mean_val", "pm", "formatted"]]

    # Wide formatted table (yearly strings)
    wide_fmt = df_small.pivot_table(
        index=["country", "species", "model"],
        columns="year",
        values="formatted",
        aggfunc="first"
    )
    wide_fmt.columns.name = None

    # Wide numerical (means only)
    wide_vals = df_small.pivot_table(
        index=["country", "species", "model"],
        columns="year",
        values="mean_val",
        aggfunc="first"
    )

    # Wide uncertainty table
    wide_unc = df_small.pivot_table(
        index=["country", "species", "model"],
        columns="year",
        values="pm",
        aggfunc="first"
    )

    # Compute whole period means
    # mean_2013_2024 = wide_vals.loc[:, 2013:2024].mean(axis=1).round(2)
    # mean_unc_2013_2024 = wide_unc.loc[:, 2013:2024].mean(axis=1).round(2)
    avr = wide_vals.mean(axis=1)
    avr_unc = wide_unc.mean(axis=1)

    # # Format "mean ± mean_unc" for period means
    # def format_period(m, u):
    #     if pd.isna(m):
    #         return np.nan
    #     if pd.isna(u):
    #         return f"{m:.3g}"
    #     return f"{m:.3g} ± {u:.3g}"

    # wide_fmt["Mean_2013_2016"] = (
    #     mean_2013_2016.combine(mean_unc_2013_2016, lambda m, u: f"{m:.2f} ± {u:.2f}")
    # )

    # wide_fmt["Mean_2017_2023"] = (
    #     mean_2017_2023.combine(mean_unc_2017_2023, lambda m, u: f"{m:.2f} ± {u:.2f}")
    # )

    wide_fmt["Mean"] = (
        avr.combine(avr_unc, lambda m, u: f"{m:.2g} ± {u:.2g}")
    )

    # # Percent change (numerical means only!)
    # wide_fmt["Pct_change_%"] = (
    #     (mean_2017_2023 - mean_2013_2016) / mean_2013_2016 * 100
    # ).round(2)

    # Build final column list
    year_columns = sorted([c for c in wide_fmt.columns if isinstance(c, int)])
    final_cols = (
        ["country", "species", "model"] +
        year_columns +
        ["Mean"]
    )

    # Final dataframe
    final = wide_fmt.reset_index()[final_cols]

    # Rename columns
    final = final.rename(columns={
        "country": "region",
        "model": "source"
    })

    return final

def summarize_emissions(df, region, period1, period2):
    """
    Filter reshape_emissions() output to a specific region
    and return only summary columns:
      - Period 1
      - Period 2 
      - Pct_change_%

    Parameters
    ----------
    df : pandas.DataFrame
        Output from reshape_emissions()
    region : str or list of str
        Region(s) to include.

    Returns
    -------
    pandas.DataFrame
        Filtered summary table.
    """

    import pandas as pd

    # Allow single string or list
    if isinstance(region, str):
        region = [region]

    # Filter
    df_sel = df[df["region"].isin(region)].copy()

    # Keep only desired columns
    cols = [
        "region", "species", "source",
        period1,
        period2,
        "Pct_change_%",
    ]

    # Some safety: keep only columns that exist
    cols = [c for c in cols if c in df_sel.columns]

    df = df_sel[cols].reset_index(drop=True)

    # Percent change
    period1_val = (
        df[period1]
        .astype(str)
        .str.extract(r"([-+]?\d*\.?\d+)")[0]
        .astype(float)
    )
    period2_val = (
        df[period2]
        .astype(str)
        .str.extract(r"([-+]?\d*\.?\d+)")[0]
        .astype(float)
    )
    pct_change = (
        (period2_val - period1_val) / period1_val * 100
    )
    df["Pct_change_%"] = pct_change.apply(lambda x: f"{x:.2g}" if pd.notnull(x) else x)
    return df
