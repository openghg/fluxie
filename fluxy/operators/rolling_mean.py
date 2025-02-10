import xarray as xr 

def calc_rolling_mean(ds : xr.Dataset,
                      time_period : int = 3
                      )->xr.Dataset:
    
    ds_mean = ds.rolling(time=time_period,center=True).mean()
    ds_mean = ds_mean.dropna(dim='time',how='all')

    return xr.concat([ds.isel(time=0),ds_mean,ds.isel(time=-1)],
                     dim='time')