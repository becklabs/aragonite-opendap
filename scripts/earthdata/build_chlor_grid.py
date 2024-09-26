import os
import logging
import argparse
import numpy as np
import pandas as pd
from oadap.providers.opendap import AQUAMODISCHLORL3

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Find the chlorophyll gridpoints not on land"
    )
    parser.add_argument(
        "--fvcom_xy_file",
        default="./data/FVCOM/preprocessed/temperature/all/artifacts/xy.npy",
        help="Path to FVCOM gridpoints file",
    )
    parser.add_argument(
        "--land_shp",
        default="./data/shp/ne_10m_land/ne_10m_land.shp",
        help="Path to land shapefile",
    )
    parser.add_argument(
        "--output_dir",
        default="./data/EarthData/chlor_grid/",
        help="Output path for ocean_mask.npy",
    )
    return parser.parse_args()

def create_bounding_box(lat_min, lat_max, lon_min, lon_max):
    """Create a bounding box around the given lat/lon with a buffer (in degrees)"""
    return box(lon_min, lat_min, lon_max, lat_max)

def is_on_land(lat, lon, land):
    """Check if a point is on land"""
    point = Point(lon, lat)
    return land.contains(point).any()

def is_on_land_mask(xy, land):
    """Check if a list of points are on land"""
    return np.array([is_on_land(lat, lon, land) for lon, lat in xy])

args = parse_arguments()
fvcom_xy = np.load(args.fvcom_xy_file)

lat_min = np.min(fvcom_xy[:, 1])
lat_max = np.max(fvcom_xy[:, 1])
lon_min = np.min(fvcom_xy[:, 0])
lon_max = np.max(fvcom_xy[:, 0])


chlor_opendap = AQUAMODISCHLORL3(
)

logging.info("Pulling Chlor grid from AQUAMODISCHLORL3")
dummy_date = pd.Timestamp("2023-01-01")
chlor, xy, time = chlor_opendap.subset(
    lat_min=lat_min,
    lat_max=lat_max,
    lon_min=lon_min,
    lon_max=lon_max,
    start=dummy_date,
    end=dummy_date,
)

nx, ny, _ = xy.shape

logging.info("Creating ocean mask")
world = gpd.read_file(args.land_shp)

bbox = create_bounding_box(lat_min, lat_max, lon_min, lon_max)
filtered_world = world[world.intersects(bbox)]

ocean_mask = ~is_on_land_mask(xy.reshape(-1, 2), filtered_world)
ocean_mask = ocean_mask.reshape(nx, ny)


os.makedirs(args.output_dir, exist_ok=True)
np.save(os.path.join(args.output_dir, "ocean_mask.npy"), ocean_mask)

