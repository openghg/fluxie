import numpy as np

def align_dataset(ds_list: list[xr.Dataset]
                  )->list[xr.Dataset]: 
    """
    Check time coord and align the time coord of a list of xarray datasets on the time coord of the first of the list if not equals.

    Args:
        ds_list: list of xarray datasets to be temporarily aligned
    Returns:
        ds_ouput: list of xarray datasets temporarily aligned
    """
    time_dim_equal = [ds_list[0].time.equals(x.time) for x in ds_list[1:]]

    if all(time_dim_equal):
        ds_ouput = ds_list
    else :
        # Infer period of first dataset 
        dtime = ds_list[0].time.values[1:] - ds_list[0].time.values[:-1]
        if any(abs(dtime-np.median(dtime))>0.1*np.median(dtime)):
            raise ValueError('Unable to infer period from dataset')
        period = np.median(dtime)
        
        aligned_ds_list = [ds_list[0],]
        for ds_p in ds_list[1:]:

            if ds_list[0].time.equals(ds_p.time):
                aligned_ds_list.append(ds_p)
                continue

            diff_time = abs(ds_list[0].time.values - ds_p.time.values)
            if any(diff_time>0.1*period):
                raise ValueError(f'Time dimensions seemed to be too strong to combined (period of ref dataset: {period.astype("timedelta64[D]")}, max diff: {max(diff_time).astype("timedelta64[D]")})')

            ds_aligned = ds_p  
            ds_aligned['time'] = ds_list[0].time
            aligned_ds_list.append(ds_p)
        ds_ouput = aligned_ds_list

    return ds_ouput