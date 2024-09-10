import os
import logging
import argparse
import numpy as np
import pandas as pd
from oadap.providers.opendap import GHRSSTL4

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Find the SST gridpoints and their neighbors to be used as the prediction domain"
    )
    parser.add_argument(
        "--fvcom_xy_file",
        default="./data/FVCOM/preprocessed/temperature/all/artifacts/xy.npy",
        help="Path to FVCOM gridpoints file",
    )
    parser.add_argument(
        "--output_dir",
        default="./data/EarthData/sst_grid/",
        help="Output path for neighbors.npy and xy.npy",
    )
    return parser.parse_args()


args = parse_arguments()
fvcom_xy = np.load(args.fvcom_xy_file)

lat_min = np.min(fvcom_xy[:, 1])
lat_max = np.max(fvcom_xy[:, 1])
lon_min = np.min(fvcom_xy[:, 0])
lon_max = np.max(fvcom_xy[:, 0])


temperature_opendap = GHRSSTL4(
    earthdata_username=os.environ["EARTHDATA_USERNAME"],
    earthdata_password=os.environ["EARTHDATA_PASSWORD"],
)

logging.info("Pulling SST grid from GHRSSTL4")
dummy_date = pd.Timestamp("2023-01-01")
temp, xy, time = temperature_opendap.subset(
    lat_min=lat_min,
    lat_max=lat_max,
    lon_min=lon_min,
    lon_max=lon_max,
    start=dummy_date,
    end=dummy_date,
)

nx, ny, _ = xy.shape

logging.info("Finding neighbors")
n_neighbors = 4
neighbors = np.ones((nx, ny, n_neighbors)) * -1
neighbors = neighbors.astype(int)

ocean_mask = ~np.isnan(temp[..., 0])

for i in range(nx):
    for j in range(ny):
        if not ocean_mask[i, j]:
            continue
        # Northeast neighbor
        if i + 1 < nx and j + 1 < ny and ocean_mask[i + 1, j + 1]:
            neighbors[i, j, 0] = np.ravel_multi_index((i + 1, j + 1), (nx, ny))
        # Northwest neighbor
        if i - 1 >= 0 and j + 1 < ny and ocean_mask[i - 1, j + 1]:
            neighbors[i, j, 1] = np.ravel_multi_index((i - 1, j + 1), (nx, ny))
        # Southeast neighbor
        if i + 1 < nx and j - 1 >= 0 and ocean_mask[i + 1, j - 1]:
            neighbors[i, j, 2] = np.ravel_multi_index((i + 1, j - 1), (nx, ny))
        # Southwest neighbor
        if i - 1 >= 0 and j - 1 >= 0 and ocean_mask[i - 1, j - 1]:
            neighbors[i, j, 3] = np.ravel_multi_index((i - 1, j - 1), (nx, ny))

os.makedirs(args.output_dir, exist_ok=True)
np.save(os.path.join(args.output_dir, "neighbors.npy"), neighbors)
np.save(os.path.join(args.output_dir, "xy.npy"), xy)
