import numpy as np

def calc_rolling_mean(ds):
    """
    Calculate rolling mean of a list of numpy array using numpy.convolv
    (see https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy).
    
    Args:
        list_data : list of numpy array of dtype float or numpy.datetime64
        
    Return
        averaged_data : list of numpy array containing the averaged data.
    """
    rolling_mean = 3
    averaged_data = list()
    for data in ds:
        if np.issubdtype(data.dtype, np.datetime64):
            tmp = (np.convolve(data.astype(int), np.ones(rolling_mean), 'valid') / rolling_mean
                  ).astype(data.dtype)
            tmp = np.concatenate([[data[0],],tmp,[data[-1],]])
        else : 
            tmp = np.convolve(data, np.ones(rolling_mean), 'valid') / rolling_mean
            tmp = np.concatenate([[np.mean(data[:2]),],tmp,[np.mean(data[-2:]),]])
        averaged_data.append(tmp)
    return averaged_data
 