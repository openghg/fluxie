import numpy as np
import xarray as xr

def align_dataset(ds_list: list[xr.Dataset])->list[xr.Dataset]: 
    """
    Check the time coordinates of a list of xarray datasets and, if they differ, align them with the time coordinate of the first dataset in the list.

    Args:
        ds_list: list of xarray datasets to be time-aligned
    Returns:
        aligned_ds_list: list of xarray datasets time-aligned
    """
    time_dim_equal = [ds_list[0].time.equals(x.time) for x in ds_list[1:]]

    if all(time_dim_equal):
        return ds_list
    
    # Infer period of first dataset 
    dtime = ds_list[0].time.values[1:] - ds_list[0].time.values[:-1]
    if any(abs(dtime - np.median(dtime)) > 0.1 * np.median(dtime)):
        raise ValueError("Unable to infer period from dataset")
    period = np.median(dtime)
    
    aligned_ds_list = [ds_list[0]]

    for ds_p in ds_list[1:]:
        if ds_list[0].time.equals(ds_p.time):
            aligned_ds_list.append(ds_p)
            continue

        diff_time = abs(ds_list[0].time.values - ds_p.time.values)
        if any(diff_time > 0.1 * period):
            raise ValueError(
                f"Time dimensions seem to be too different between the datasets for them to be combined "
                + f'(period of reference dataset: {period.astype("timedelta64[D]")}, max difference: {max(diff_time).astype("timedelta64[D]")})'
            )
        
        ds_aligned = ds_p  
        ds_aligned["time"] = ds_list[0].time
        aligned_ds_list.append(ds_p)

    return aligned_ds_list