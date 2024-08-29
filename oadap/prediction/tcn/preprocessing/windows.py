from typing import Tuple, Optional

import numpy as np
import pandas as pd # type: ignore


def create_windows_simple(arr: np.ndarray, window_size: int) -> np.ndarray:
    """
    Create time dimension windows of size window_size for the given array
    """
    nx, nt, nf = arr.shape
    n_windows = nt - window_size  # Exclude last timestep from window

    sequences = np.empty((nx, n_windows, window_size, nf))
    for start_idx in range(n_windows):
        end_idx = start_idx + window_size
        sequences[:, start_idx, :, :] = arr[:, start_idx:end_idx, :]

    return sequences

def create_windows(
    arr: np.ndarray, window_size: int, sampling_rate: int = 1, stride: int = 1, last_only: bool = False
) -> np.ndarray:
    """
    Create windows of size window_size for the given array along the second to last axis
    
    Parameters:
    arr (numpy.ndarray): Input array of arbitrary shape with at least 2 dimensions
    window_size (int): Size of the window
    sampling_rate (int): Number of steps to skip between each sample within a window
    stride (int): Number of steps to move between the start of each window
    
    Returns:
    numpy.ndarray: Array of windowed sequences
    """
    # Ensure the input array has at least 2 dimensions
    if arr.ndim < 2:
        raise ValueError("Input array must have at least 2 dimensions")
    
    # Get the shape of the input array
    input_shape = arr.shape
    
    # The second to last dimension is the one we'll window over
    nt = input_shape[-2]
    
    effective_window_size = (window_size - 1) * sampling_rate + 1
    n_windows = (nt - effective_window_size) // stride + 1
    
    # Create the output shape: original dimensions + number of windows + window size
    if last_only:
        output_shape = input_shape[:-2] + (n_windows,) + (input_shape[-1],)
    else:
        output_shape = input_shape[:-2] + (n_windows, window_size) + (input_shape[-1],)
    
    # Create an empty array to store the sequences
    sequences = np.empty(output_shape, dtype=arr.dtype)
    
    # Create windows
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + effective_window_size
        window = arr[..., start_idx:end_idx:sampling_rate, :]
        if last_only:
            sequences[..., i, :] = window[..., -1, :]
        else:
            sequences[..., i, :, :] = window
    return sequences



