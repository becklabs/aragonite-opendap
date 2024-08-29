from typing import List
import numpy as np

import geopandas as gpd

from oadap.utils import load_mat

from oadap.prediction.tcn.preprocessing import find_neighbors

data_dir = "data/"
lon = load_mat(data_dir + "FVCOM/x.mat")
lat = load_mat(data_dir + "FVCOM/y.mat")
xy = np.column_stack((lon, lat))

# Load the Natural Earth 10m coastline data
coastline = gpd.read_file(data_dir + "shp/ne_10m_coastline/ne_10m_coastline.shp")

neighbor_inds = find_neighbors(
    xy=xy, coastline=coastline, min_distance=0.01, max_distance=0.04
)

np.save(data_dir + "FVCOM/neighbor_inds_new.npy", neighbor_inds)
