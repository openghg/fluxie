import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from fluxie.plots.utils import add_colorbar


def plot_flux_country_covariance(
    ds_all: dict[xr.Dataset],
    selected_countries: list[str] = None,
    model_labels: dict[str, str] = {},
    cov_max: float = 0.5,
    cmap_diff: str = "coolwarm",
) -> plt.Figure: 
    """
    Plot posterior country covariance for each model that provides variable 'covariance_flux_total_posterior_country'.
    Only lower triangle of matrix plotted. Covariance is normalised before plotting. Values will range between -1 and 1. 

    Args:
        ds_all (dictionary of datasets):
            Dictionary of fluxes xarray datasets.
        selected_countries (list of str):
            Countries to be evaluated.
        model_labels (list):
            List of model_labels from fluxie.config.
        cov_max (float, optional): 
            Maximum value of covariance for colorbar. The colorbar will run from -cov_max to cov_max.
        cmap_diff (str, optional):
            Colour map for flux difference plots.        

    Returns:
        fig (figure):
            Plot of covariance matrix (lower triangle) for each model that provides covariance_flux_total_posterior_country.
            
    """
    
    # function for selecting only datasets with covariance variable
    def filter_cov(pair):
        check_var='covariance_flux_total_posterior_country'
        key, value = pair
        return check_var in value.data_vars
    

    # filter for datasets that contain covariance 
    ds = dict(filter(filter_cov, ds_all.items()))
    
    # number of models required to decide on plot layout
    models = list(ds.keys())
    nn_models = len(ds)
   
    # start plot layout
    fig, ax = plt.subplots(ncols=nn_models)
    for ii, mod in enumerate(ds): 

        #  extract the data from each model 
        msk = range(0, len(ds[mod].country))
        if selected_countries is not None:
            msk = [i for i, val in enumerate(ds[mod].country.values) if val in selected_countries]
        cov = ds[mod].covariance_flux_total_posterior_country.mean(dim='time')[msk, msk]
        countries = ds[mod].country.values[msk]
        nn_country = len(countries)

        # normalise covariance (using variance on diagonal)
        dd = np.sqrt(np.diag(cov))
        tmp = np.repeat(dd, nn_country)
        cov = cov / np.reshape(tmp, [nn_country, nn_country]) / np.reshape(tmp, [nn_country, nn_country], order='F')
        # construct matrix to hide upper triangle of matrix
        upper_mask = np.tri(cov.shape[0], cov.shape[1], k=-1)
        upper_mask[upper_mask==0] = np.nan

        cov = cov*upper_mask
        # remove top row and rightmost column (no information contained)
        cov = cov[1:nn_country,0:(nn_country-1)]
        
        #  plot
        im = ax[ii].imshow(cov, origin='upper', cmap=cmap_diff, vmin=-cov_max, vmax=cov_max)
        ax[ii].xaxis.set_ticks(range(0,nn_country-1), countries[0:(nn_country-1)], rotation=45)
        ax[ii].yaxis.set_ticks(range(0,nn_country-1), countries[1:nn_country], rotation=45)
        ax[ii].spines['top'].set_visible(False)
        ax[ii].spines['right'].set_visible(False)
        ax[ii].set_title(model_labels[mod], fontsize='small')

    # add colorbar
    period_str = ds[models[0]].time[0].dt.strftime("%Y").values+'-'+ds[models[0]].time[-1].dt.strftime("%Y").values  
    add_colorbar(fig, ax[ii], im, 'both', 'Correlation (-)\n'+period_str, n_cbar=nn_models, idx_cbar=1, 
            colorbar_type="row")

    return(fig)
