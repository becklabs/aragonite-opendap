from typing import Tuple, Optional
import numpy as np


def uniform_sampling(
    coordinates: np.ndarray, n_samples: int, random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniformly sample points from a set of coordinates.
    """

    rng = np.random.default_rng(random_state)

    # Convert coordinates to numpy array
    coords = np.array(coordinates)

    # Determine grid size
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    # Calculate number of cells in each dimension
    n_cells = int(np.sqrt(n_samples))

    # Create grid
    x_edges = np.linspace(x_min, x_max, n_cells + 1)
    y_edges = np.linspace(y_min, y_max, n_cells + 1)

    # Initialize lists to store sampled points and their indices
    sampled_points = []
    sampled_indices = []

    # Iterate through grid cells
    for i in range(n_cells):
        for j in range(n_cells):
            # Find points in current cell
            mask = (
                (coords[:, 0] >= x_edges[i])
                & (coords[:, 0] < x_edges[i + 1])
                & (coords[:, 1] >= y_edges[j])
                & (coords[:, 1] < y_edges[j + 1])
            )
            cell_points = coords[mask]
            cell_indices = np.where(mask)[0]

            # If cell contains points, select one randomly
            if len(cell_points) > 0:
                random_index = rng.integers(len(cell_points))
                sampled_points.append(cell_points[random_index])
                sampled_indices.append(cell_indices[random_index])

    return np.array(sampled_points), np.array(sampled_indices)
