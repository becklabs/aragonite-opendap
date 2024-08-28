import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import PyCO2SYS
from scipy.special import erf, erfc


def closest_point_grid(lat, lon, lat_grid, lon_grid):
    coordinates = pd.DataFrame({'longitude': lon_grid.ravel(), 'latitude': lat_grid.ravel()})[['latitude', 'longitude']].to_numpy()
    tree = KDTree(coordinates)
    dist, idx = tree.query([lat, lon])
    index_pair = np.unravel_index(idx, lat_grid.shape)

    return index_pair

def adjacent_grid_points(i, j):
    left = (i, j+1)
    right = (i, j-1)
    above = (i-1, j)
    below = (i+1, j)
    
    return (left, right, above, below)

def closest_point(point, points):
    points = np.asarray(points)
    dist_2 = np.sum((points - point)**2, axis=1)
    return np.argmin(dist_2)

def chauvenet(array):
    mean = array.mean()           # Mean of incoming array
    stdv = array.std()            # Standard deviation
    N = len(array)                # Length of array
    criterion = 1.0/(2*N)         # Chauvenet's criterion
    d = abs(array-mean)/stdv      # Distance of a value to mean in stdv's
    prob = erfc(d)        # Area normal dist.    
    return prob >= criterion      # Use boolean array outside this function


def RMSE(yt, yp):
    return np.round(np.sqrt(np.nanmean(np.square(yt.ravel() - yp.ravel()))),3)

def convert_pH(data, input_scale = 'sws' , output_scale = 'nbs'):
    input_scale_map = {
        'total': 1,
        'sws': 2,
        'free': 3,
        'nbs': 4
    }
    input_scale_val = input_scale_map[input_scale]
    output_scale_val = f'pH_{output_scale}_out'


    # Define input and output conditions
    kwargs = dict(
        par1_type = 1,  # The first parameter supplied is of type "1", which means "alkalinity"
        par1 = data['TA'],  # value of the first parameter
        par2_type = 3,  # The second parameter supplied is of type "3", which means "pH"
        par2 = data['pH'],  # value of the second parameter
        salinity = data['Salinity'],  # Salinity of the sample
        temperature = data['Temperature'],  # Temperature at input conditions
        temperature_out = 25,  # Temperature at output conditions
        pressure = data['Depth'],  # Pressure at input conditions
        pressure_out = 0,  # Pressure at output conditions
        opt_pH_scale = input_scale_val,  # pH scale at which the input pH is reported ("1" means "Total Scale")
        opt_k_carbonic = 1,  # Choice of H2CO3 and HCO3- dissociation constants K1 and K2 ("4" means "Mehrbach refit")
        opt_k_bisulfate = 1,  # Choice of HSO4- dissociation constant KSO4 ("1" means "Dickson")
        opt_total_borate = 2,  # Choice of boron:sal ("1" means "Uppstrom")
    )

    # Run PyCO2SYS
    results = PyCO2SYS.sys(**kwargs)
    return results[output_scale_val]