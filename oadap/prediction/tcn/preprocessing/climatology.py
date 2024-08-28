from typing import Tuple

import numpy as np
import pandas as pd  # type: ignore


def calculate_climatology_days(time: pd.DatetimeIndex, data: np.ndarray) -> np.ndarray:
    """
    Calculate the climatology for the given 3D data.
    Returns an array of shape (nx, 366, nz) where the second dimension has 366 days to include Feb 29.
    """
    nx, nt, nz = data.shape
    days_of_year = time.dayofyear
    assert days_of_year.shape == (nt,), "Time index must match data"

    climatology_days = np.zeros((nx, 365, nz))
    day_counts = np.zeros(365)

    # Calculate climatology
    for day in range(1, 365 + 1):
        if day > 59:
            day_indices = np.where(
                ((days_of_year == day) & (~time.is_leap_year))
                | ((days_of_year == day + 1) & time.is_leap_year)
            )[0]
        else:
            day_indices = np.where(days_of_year == day)[0]

        if day_indices.size > 0:
            samples = data[:, day_indices, :]
            climatology_days[:, day - 1, :] = np.sum(samples, axis=1)
            day_counts[day - 1] = samples.shape[1]

    climatology_days /= day_counts[:, np.newaxis]

    # Calculate leap day climatology
    feb29_indices = np.where((days_of_year == 60) & time.is_leap_year)[0]
    feb29_samples = data[:, feb29_indices, :]
    feb29_climatology = np.sum(feb29_samples, axis=1) / feb29_samples.shape[1]

    leap_climatology_days = np.zeros((nx, 366, nz))
    leap_climatology_days[:, :59, :] = climatology_days[:, :59, :]
    leap_climatology_days[:, 59, :] = feb29_climatology
    leap_climatology_days[:, 60:, :] = climatology_days[:, 59:, :]

    return leap_climatology_days


def calculate_climatology(
    time: pd.DatetimeIndex, data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the climatology and anomalies for the given 3D data
    """
    nx, nt, nz = data.shape
    days_of_year = time.dayofyear
    assert days_of_year.shape == (nt,), "Time index must match data"

    climatology_days = calculate_climatology_days(time=time, data=data)

    # Create the final climatology array matching the original data shape
    climatology = np.zeros_like(data)
    for i, date in enumerate(time):
        day_of_year = date.dayofyear
        day_index = day_of_year - 1

        if day_of_year > 59 and not date.is_leap_year:
            climatology[:, i, :] = climatology_days[:, day_index + 1, :]
        else:
            climatology[:, i, :] = climatology_days[:, day_index, :]

    anomalies = data - climatology
    return climatology, anomalies


def reconstruct_data(
    anomalies: np.ndarray, time: pd.DatetimeIndex, climatology_days: np.ndarray
):
    nx, nt, nz = anomalies.shape
    days_of_year = time.dayofyear
    assert days_of_year.shape == (nt,), "Time index must match data"
    assert climatology_days.shape == (nx, 366, nz), "Climatology days must match data"

    climatology = np.zeros_like(anomalies)
    for i, date in enumerate(time):
        day_of_year = date.dayofyear
        day_index = day_of_year - 1

        if day_of_year > 59 and not date.is_leap_year:
            climatology[:, i, :] = climatology_days[:, day_index + 1, :]
        else:
            climatology[:, i, :] = climatology_days[:, day_index, :]

    data = anomalies + climatology
    return data
