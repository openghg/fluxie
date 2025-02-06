import xarray as xr 

def calc_rolling_mean(ds : xr.Dataset,
                      time_period : int = 3
                      )->xr.Dataset:
    ds_mean = ds.rolling(time=time_period,center=True).mean()
    return ds_mean.dropna(dim='time',how='all')