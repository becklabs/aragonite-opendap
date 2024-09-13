from typing import List
import numpy as np
import geopandas as gpd
from oadap.utils import load_mat
from oadap.prediction.tcn.preprocessing import find_neighbors
import argparse
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Find the FVCOM neighbor indices")
    parser.add_argument(
        "--x_file",
        default="./data/FVCOM/x.mat",
        help="Path to FVCOM x coordinates file",
    )
    parser.add_argument(
        "--y_file",
        default="./data/FVCOM/y.mat",
        help="Path to FVCOM y coordinates file",
    )
    parser.add_argument(
        "--coastline_shp",
        default="./data/shp/ne_10m_coastline/ne_10m_coastline.shp",
        help="Path to coastline shapefile",
    )
    parser.add_argument(
        "--min_distance",
        type=float,
        default=0.01,
        help="Minimum distance for neighbor search (km)",
    )
    parser.add_argument(
        "--max_distance",
        type=float,
        default=0.04,
        help="Maximum distance for neighbor search (km)",
    )
    parser.add_argument(
        "--output_file",
        default="./data/FVCOM/neighbor_inds.npy",
        help="Output file for neighbor indices",
    )
    return parser.parse_args()


args = parse_arguments()

lon = load_mat(args.x_file)
lat = load_mat(args.y_file)
xy = np.column_stack((lon, lat))

# Load the Natural Earth 10m coastline data
coastline = gpd.read_file(args.coastline_shp)

logging.info("Finding neighbors")
neighbor_inds = find_neighbors(
    xy=xy,
    coastline=coastline,
    min_distance=args.min_distance,
    max_distance=args.max_distance,
)

np.save(args.output_file, neighbor_inds)
