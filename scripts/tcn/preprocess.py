import logging
import os

import numpy as np
import pandas as pd

from oadap.prediction.tcn.preprocessing import (
    calculate_climatology,
    calculate_climatology_days,
    group,
    svd_decompose,
    reconstruct_T,
    uniform_sampling,
    create_windows,
)

from oadap.utils import load_mat

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# VARIABLES
data_dir = "data/FVCOM/"
# array_path = data_dir + "temperature/temp.mat"
array_path = data_dir + "salinity/salinity.mat"


# output_dir = data_dir + "preprocessed/temperature/sample/"
output_dir = data_dir + "preprocessed/salinity/sample/"

artifacts_only = True
rewrite = True
window_size = 20
stride = 1
sampling_rate = 1
n_samples = 1200
# n_days = 2192  #  nt - 365 * 3
n_days = 3268

if not rewrite and os.path.exists(output_dir):
    logger.info(f"Output directory {output_dir} already exists. Skipping.")
    exit(0)

# Load Data
logger.info("Loading data")

lon = load_mat(data_dir + "x.mat")
lat = load_mat(data_dir + "y.mat")

h = load_mat(data_dir + "h.mat")  # Distance from the surface to the ocean floor
siglay = load_mat(data_dir + "siglay.mat")
time = pd.date_range(start="1/1/2005", end="12/31/2013")

neighbors = np.load(data_dir + "neighbor_inds.npy").astype(np.int32)

nx = lon.shape[0]
nt = time.shape[0]
nz = siglay.shape[1]
n_neighbors = neighbors.shape[-1]

T = load_mat(array_path)
T = T.transpose((0, 2, 1))  # (nx, nt, nz)
assert T.shape == (nx, nt, nz)
temp_surface = T[..., 0]

# Calculate q, phi of anomalies
logger.info("Calculating SVD decomposition")
climatology_days_3d = calculate_climatology_days(time=time, data=T)
climatology_3d, anomalies_3d = calculate_climatology(time=time, data=T)
q_anomaly, phi_anomaly, T_bar_anomaly, _ = svd_decompose(
    anomalies_3d, n_modes=2, check=False, align=True
)

# Calculate climatology and anomalies of the surface
logger.info("Grouping with neighbors")
climatology_surface, anomalies_surface = calculate_climatology(
    time=time, data=temp_surface[..., np.newaxis]
)
climatology_surface = climatology_surface.squeeze()
anomalies_surface = anomalies_surface.squeeze()

# Group points with their neighbors
anomalies_grouped = group(data=anomalies_surface, neighbor_inds=neighbors)
climatology_grouped = group(data=climatology_surface, neighbor_inds=neighbors)

# Select uniform sample
logger.info("Selecting uniform sample")
xy = np.column_stack((lon, lat))

has_neighbors_mask = neighbors[:, 0] != -1
has_neighbor_inds = np.where(has_neighbors_mask)[0]

samples, sample_inds = uniform_sampling(
    xy[has_neighbors_mask], n_samples=n_samples, random_state=42
)
sample_mask = np.zeros(nx, dtype=bool)
sample_mask[has_neighbor_inds[sample_inds]] = True


# Build Inputs
logger.info("Building inputs")
X = np.stack(
    (
        anomalies_grouped[..., 0],
        anomalies_grouped[..., 1],
        anomalies_grouped[..., 2],
        anomalies_grouped[..., 3],
        anomalies_grouped[..., 4],
        climatology_surface,
        np.repeat(lon, nt, axis=1),
        np.repeat(lat, nt, axis=1),
        np.repeat(h, nt, axis=1),
    ),
    axis=2,
)

X_sample = X[sample_mask, -n_days:]
X_windowed_sample = create_windows(
    X_sample, window_size=window_size, sampling_rate=sampling_rate, stride=stride
)

# Build Outputs
logger.info("Building outputs")
y_sample = np.dstack(
    (T_bar_anomaly[sample_mask, -n_days:], q_anomaly[sample_mask, -n_days:])
)
y_sample = create_windows(
    y_sample,
    window_size=window_size,
    sampling_rate=sampling_rate,
    stride=stride,
    last_only=True,
)

# Build aux arrays
T_sample = reconstruct_T(
    phi_anomaly[sample_mask],
    q_anomaly[sample_mask, -n_days:],
    T_bar_anomaly[sample_mask, -n_days:],
)
T_sample = create_windows(
    T_sample,
    window_size=window_size,
    sampling_rate=sampling_rate,
    stride=stride,
    last_only=True,
)

phi_sample = phi_anomaly[sample_mask]

xy_sample = xy[sample_mask]

# Save
logger.info("Saving data")
train_output_dir = output_dir + "train/"
artifacts_output_dir = output_dir + "artifacts/"

for path in [train_output_dir, artifacts_output_dir]:
    if not os.path.exists(path):
        os.makedirs(path)

# Save training data
np.save(train_output_dir + "X.npy", X_windowed_sample)
np.save(train_output_dir + "y.npy", y_sample)

# Save artifacts
np.save(artifacts_output_dir + "T.npy", T_sample)
np.save(artifacts_output_dir + "phi.npy", phi_sample)
np.save(artifacts_output_dir + "xy.npy", xy_sample)
np.save(artifacts_output_dir + "climatology_days.npy", climatology_days_3d[sample_mask])

# Log the total size of the arrays in GB
total_size = sum(
    [
        arr.nbytes
        for arr in [
            X_windowed_sample,
            y_sample,
            T_sample,
            phi_sample,
            xy_sample,
            climatology_days_3d,
        ]
    ]
)
total_size_gb = total_size / 1e9
logger.info(f"Total size of the arrays: {total_size_gb:.2f} GB")
