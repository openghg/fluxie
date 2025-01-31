import numpy as np
import xarray as xr
import pandas as pd

def calculate_resampled_flux(ds_all: dict[str,xr.Dataset],
                            rtime: list[str]
                            )-> dict[str,xr.Dataset] :
    """
    Resample the datasets.
    
    Args:
        ds_all: xarray datasets of fluxes, scaled and sliced between 
            chosen dates.dict[str, xr.Dataset]
        resample: Option to be passed to resample built-in function of xarray Dataset. For yearly average, 'YS' option should be used; 'QS-DEC' for seasonaly average.
            See http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
        resample_uncert_correlation: If True, calculates the resampled uncertainty as the mean from all averaged periods. 
            If False, recalculates uncertainty assuming no correlation between all averaged periods, by taking the square root of the summed variances, divided by the number of averaging periods.
    Returns:
        ds_all_p: resampled datasets
    """
    ds_all_output = dict()
    for i, (m,ds) in enumerate(ds_all.items()) : 
        if rtime[i] is not None :
            ds_all_output[m] = ds.resample(time=rtime[i]).mean(dim="time")
        else :
            ds_all_output[m] = ds.copy()

    return ds_all_output


def calculate_resampled_uncertainty(ds_all_original: dict[str, xr.Dataset],
                                   ds_all_p: dict[str, xr.Dataset],
                                   rtime: list[str]
                                   )-> dict[str, xr.Dataset]:
    """
    Recalculates resampled flux uncertainty, using the assumption
    that all periods in the resampled flux average are uncorrelated.
    Args:
        ds_all_original: Extracted flux datasets from flux-format netcdfs.
        ds_all_p: Same as above, but resampled down to lower time resolution, using the .mean() method.
        rtime: Time used for resampling, e.g. 'YS' or 'QS-DEC'.
    Returns:
        ds_all_p: Same dataset as above, but with updated 'percentile_...' terms.
    """
    
    for i,m in enumerate(ds_all_original.keys()):

        if rtime[i] is None:
            continue

        for v in ds_all_original[m].keys():

            if 'percentile_country' in v:
                # reecrire cette partie
                n_periods = ds_all_original[m][v].resample(time=rtime[i]).count()[:,0,:]   #number of periods in each average
                lower = (ds_all_original[m][v.replace('percentile_','')] - ds_all_original[m][v][:,0,:])    #recalculate upper and low standard deviations
                upper = (ds_all_original[m][v][:,1,:] - ds_all_original[m][v.replace('percentile_','')])
                lower_resampled = np.sqrt(((lower**2).resample(time=rtime[i]).sum(dim="time")))/n_periods  #resample using sqrt of variances,divided by number of periods
                upper_resampled = np.sqrt(((upper**2).resample(time=rtime[i]).sum(dim="time")))/n_periods
                lower_out = ds_all_p[m][v.replace('percentile_','')] - lower_resampled  #recalculated percentile upper and lower bounds
                upper_out = ds_all_p[m][v.replace('percentile_','')] + upper_resampled
                
                ds_all_p[m][v] = xr.DataArray(np.concatenate((np.expand_dims(lower_out,axis=1),
                                                                np.expand_dims(upper_out,axis=1)),axis=1),
                                                dims=ds_all_p[m][v].dims)
        
    return ds_all_p

def resample_flux(ds_all: dict[str,xr.Dataset],
                  resample: list[str],
                  resample_uncert_correlation: bool = False
                  )-> dict[str,xr.Dataset] :
    """
    Resample and realign the reampled datasets.

    Args:
        ds_all: xarray datasets of fluxes, scaled and sliced between 
            chosen dates.
        resample: Option to be passed to resample built-in function of xarray Dataset. For yearly average, 'YS' option should be used; 'QS-DEC' for seasonaly average.
            See http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
        resample_uncert_correlation: If True, calculates the resampled uncertainty as the mean from all averaged periods. 
            If False, recalculates uncertainty assuming no correlation between all averaged periods, by taking the square root of the summed variances, divided by the number of averaging periods.
    Returns:
        ds_all_p: resampled datasets
    """

    # Check resample option
    # I don't agree with this renaming, I think we should keep the same names as xarray resample function, that way we can easily use any of the many available period
    # Besides there is a link to the xarray periods list in the description of the parameter 'resample'
    rtime = []

    for resample_val in resample:
        # make it possible also to use pandas stuff

        if resample_val == 'year':
            rtime.append('YS')

        elif resample_val == 'season':
            rtime.append('QS-DEC')

        elif resample_val is None:
            rtime.append(None)

        else:
            raise ValueError(f"'{resample_val}' is not available for resample. Try 'year' or 'season' or None.")
        
    ds_all_original = {m:ds_all[m].copy() for m in ds_all.keys()}
            
    ds_all_p = calculate_resampled_flux(ds_all,rtime)
    
    if not resample_uncert_correlation:
        ds_all_p = calculate_resampled_uncertainty(ds_all_original,ds_all_p,rtime)
    
    # shift timestamps of averaged data forwards to centre of inversion period
    for im,m in enumerate(ds_all.keys()):

        if rtime[im] is not None and ds_all_original[m].time.size > ds_all_p[m].time.size:

            # I feel like there is better ways of doing this but didn't find them -> check openGHG_inversions
            # in fact do it on the plot axis (or not)
            date_list_for_offset = pd.date_range(start = "2018-01-01", 
                                                 end = "2023-01-01" , 
                                                 freq = rtime[im])
            offset = (date_list_for_offset[1:]-date_list_for_offset[:-1]).mean()/2
            ds_all_p[m]['time'] = ds_all_p[m]['time'].values + offset

    return {m+'_resample': ds for m,ds in ds_all_p.items()}