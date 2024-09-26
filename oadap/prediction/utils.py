import numpy as np
import pandas as pd
import PyCO2SYS
from scipy.special import erfc
from typing import Tuple


def grid_to_swath(
    grid: np.ndarray, values: np.ndarray, time: pd.DatetimeIndex, remove_nans: bool = True
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Convert a grid of values to a swath dataset.
    Arguments:
    grid: 2D array of shape (n, 2) where each row is a pair of coordinates (longitude, latitude)
    values: 2D array of shape (n, m) where n is the number of points and m is the time dimension
    time: 1D array of datetime objects (m)
    remove_nans: If True, removes NaN values from the output

    Returns:
    xy: 2D array of shape (k, 2) where each row is a pair of coordinates (longitude, latitude)
    values: 1D array of shape (k,) where k is the number of non-NaN points if remove_nan is True, otherwise k = n*m
    time: 1D array of datetime objects (k)
    """
    n, m = values.shape

    # Repeat grid coordinates for each time step
    xy = np.repeat(grid, m, axis=0)

    # Flatten values array
    flat_values = values.ravel()

    # Repeat time array for each grid point
    repeated_time = np.tile(time, n)

    if remove_nans:
        mask = ~np.isnan(flat_values)
        xy = xy[mask]
        flat_values = flat_values[mask]
        repeated_time = repeated_time[mask]
    
    repeated_time = pd.DatetimeIndex(repeated_time)

    return flat_values, xy, repeated_time # type: ignore


def chauvenet(array):
    mean = array.mean()  # Mean of incoming array
    stdv = array.std()  # Standard deviation
    N = len(array)  # Length of array
    criterion = 1.0 / (2 * N)  # Chauvenet's criterion
    d = abs(array - mean) / stdv  # Distance of a value to mean in stdv's
    prob = erfc(d)  # Area normal dist.
    return prob >= criterion  # Use boolean array outside this function


def RMSE(yt, yp):
    return np.round(np.sqrt(np.nanmean(np.square(yt.ravel() - yp.ravel()))), 3)


def convert_pH(data, input_scale="sws", output_scale="nbs"):
    input_scale_map = {"total": 1, "sws": 2, "free": 3, "nbs": 4}
    input_scale_val = input_scale_map[input_scale]
    output_scale_val = f"pH_{output_scale}_out"

    # Define input and output conditions
    kwargs = dict(
        par1_type=1,  # The first parameter supplied is of type "1", which means "alkalinity"
        par1=data["TA"],  # value of the first parameter
        par2_type=3,  # The second parameter supplied is of type "3", which means "pH"
        par2=data["pH"],  # value of the second parameter
        salinity=data["Salinity"],  # Salinity of the sample
        temperature=data["Temperature"],  # Temperature at input conditions
        temperature_out=25,  # Temperature at output conditions
        pressure=data["Depth"],  # Pressure at input conditions
        pressure_out=0,  # Pressure at output conditions
        opt_pH_scale=input_scale_val,  # pH scale at which the input pH is reported ("1" means "Total Scale")
        opt_k_carbonic=1,  # Choice of H2CO3 and HCO3- dissociation constants K1 and K2 ("4" means "Mehrbach refit")
        opt_k_bisulfate=1,  # Choice of HSO4- dissociation constant KSO4 ("1" means "Dickson")
        opt_total_borate=2,  # Choice of boron:sal ("1" means "Uppstrom")
    )

    # Run PyCO2SYS
    results = PyCO2SYS.sys(**kwargs)
    return results[output_scale_val]
