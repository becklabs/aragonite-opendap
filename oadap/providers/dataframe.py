from typing import Tuple
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class SwathDataProvider(ABC):
    @abstractmethod
    def subset(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        """
        Subset the swath data to a bounding box and time range.

        Parameters:
        - lat_min, lat_max, lon_min, lon_max: float
            Bounding box coordinates
        - start, end: pd.Timestamp
            Start and end dates for the time range

        Returns:
        - data (n_samples,)
        - xy coordinates (n_samples, 2)
        - time range (n_samples)
        """
        ...


class DataFrameProvider(SwathDataProvider):
    def __init__(
        self,
        data_dir: str,
        lat_col: str,
        lon_col: str,
        time_col: str,
        variable_col: str,
    ):
        self.df = pd.read_csv(data_dir)
        self.df[time_col] = pd.to_datetime(self.df[time_col])
        self.df = self.df.dropna(subset=[lat_col, lon_col, time_col, variable_col])

        self.lat_col = lat_col
        self.lon_col = lon_col
        self.time_col = time_col
        self.variable_col = variable_col

    def subset(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        subset_df = self.df[
            (self.df[self.lat_col] >= lat_min)
            & (self.df[self.lat_col] <= lat_max)
            & (self.df[self.lon_col] >= lon_min)
            & (self.df[self.lon_col] <= lon_max)
            & (self.df[self.time_col] >= start)
            & (self.df[self.time_col] <= end)
        ]
        return (
            subset_df[[self.variable_col]].values,
            subset_df[[self.lon_col, self.lat_col]].values,
            subset_df[self.time_col],
        )


class MWRASalinity(DataFrameProvider):
    def __init__(self, data_dir: str):
        super().__init__(
            data_dir,
            lat_col="Latitude",
            lon_col="Longitude",
            time_col="Date",
            variable_col="Salinity",
        )
