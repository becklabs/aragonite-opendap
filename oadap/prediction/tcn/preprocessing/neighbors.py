import numpy as np
from typing import List
from shapely.geometry import Point, LineString # type: ignore

from typing import List
import numpy as np

from shapely.ops import unary_union # type: ignore


def is_path_over_land(point1, point2, coastline_geometry):
    """Check if the path between two points crosses the coastline"""
    line = LineString([point1, point2])
    return line.intersects(coastline_geometry)


def accessible_mask(point: Point, candidates: List[Point], coastline_geometry):
    return [not is_path_over_land(point, p, coastline_geometry) for p in candidates]


def northeast_mask(point: np.ndarray, candidates: np.ndarray):
    return (candidates[:, 0] > point[0]) & (candidates[:, 1] > point[1])


def northwest_mask(point, candidates):
    return (candidates[:, 0] < point[0]) & (candidates[:, 1] > point[1])


def southeast_mask(point, candidates):
    return (candidates[:, 0] > point[0]) & (candidates[:, 1] < point[1])


def southwest_mask(point, candidates):
    return (candidates[:, 0] < point[0]) & (candidates[:, 1] < point[1])


def find_neighbors(xy, coastline, min_distance=0.01, max_distance=0.04):
    """
    Implementation of the neighbor finding algorithm

    1. Select points that are (min_distance < x < max_distance) from x (in decimal degrees)
    2. Select points that are not separated by land from x
    3. Find the closest NE, NW, SE, SW points to x from selected points

    Note: If no neighbors are found in at least one direction, the neighbor indices are set to -1
    """
    x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
    y_min, y_max = xy[:, 1].min(), xy[:, 1].max()

    local_coastline = unary_union(coastline.cx[x_min:x_max, y_min:y_max].geometry)

    neighbor_inds = np.empty((xy.shape[0], 4))
    for p in range(len(xy)):
        point = xy[p]

        # Find points that are (min_distance < x < max_distance) from x
        distances = np.linalg.norm(xy - point, axis=1)
        candidate_inds = np.where(
            (distances > min_distance) & (distances < max_distance)
        )[0]
        candidates = xy[candidate_inds]

        # Find points that are not separated by land from x
        accessible_inds = np.where(
            accessible_mask(
                Point(*point), [Point(*p) for p in candidates], local_coastline
            )
        )[0]
        accessible_candidates = candidates[accessible_inds]

        accessible_distances = distances[candidate_inds][accessible_inds]
        accessible_candidates_sorted_inds = np.argsort(accessible_distances)
        accessible_candidates_sorted = accessible_candidates[
            accessible_candidates_sorted_inds
        ]

        sorted_candidate_inds = candidate_inds[accessible_inds][
            accessible_candidates_sorted_inds
        ]

        # Find the closest NE, NW, SE, SW points

        ne_inds = np.where(northeast_mask(point, accessible_candidates_sorted))[0]
        nw_inds = np.where(northwest_mask(point, accessible_candidates_sorted))[0]
        se_inds = np.where(southeast_mask(point, accessible_candidates_sorted))[0]
        sw_inds = np.where(southwest_mask(point, accessible_candidates_sorted))[0]

        if (
            ne_inds.size == 0
            or nw_inds.size == 0
            or se_inds.size == 0
            or sw_inds.size == 0
        ):
            neighbor_inds[p] = [-1, -1, -1, -1]
        else:
            ne_neighbor = sorted_candidate_inds[ne_inds[0]]
            nw_neighbor = sorted_candidate_inds[nw_inds[0]]
            se_neighbor = sorted_candidate_inds[se_inds[0]]
            sw_neighbor = sorted_candidate_inds[sw_inds[0]]
            neighbor_inds[p] = [ne_neighbor, nw_neighbor, se_neighbor, sw_neighbor]

    return neighbor_inds

def group(data: np.ndarray, neighbor_inds: np.ndarray) -> np.ndarray:
    """
    Group data points with their neighbors specified in a neighbor_inds array.
    Missing neighbors (represented by -1) are replaced with np.nan.
    """
    nx = data.shape[0]
    nt = data.shape[1]
    n_neighbors = neighbor_inds.shape[-1]
    grouped = np.full((nx, nt, n_neighbors + 1), np.nan)
    
    for i, loc in enumerate(data):
        grouped[i, :, 0] = loc
        valid_neighbors = neighbor_inds[i] != -1
        grouped[i, :, 1:][..., valid_neighbors] = data[neighbor_inds[i][valid_neighbors]].T
    
    return grouped
