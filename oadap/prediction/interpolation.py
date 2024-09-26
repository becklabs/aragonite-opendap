import numpy as np
import pandas as pd
import datetime
from abc import ABC, abstractmethod
from typing import Callable, Union, Tuple, Dict, Any
from pykrige.ok3d import OrdinaryKriging3D
from sklearn.preprocessing import StandardScaler


class Interpolator(ABC):
    """
    Interface for Interpolators
    """

    @abstractmethod
    def fit(self, xy: np.ndarray, values: np.ndarray, time: pd.DatetimeIndex):
        """
        Fit the model to the data

        Arguments:
        xy: 2D array of shape (n, 2) where each row is a pair of coordinates (longitude, latitude)
        values: 1D array of shape (n,) where n is the number of points
        time: 1D array of datetime objects (n)
        """

    @abstractmethod
    def predict(self, grid: np.ndarray, days: pd.DatetimeIndex) -> np.ndarray:
        """
        Interpolate values at grid points for each day in days

        Arguments:
        grid: 2D array of shape (n, 2) where each row is a pair of coordinates (longitude, latitude)
        days: 1D array of datetime objects (m)

        Returns:
        2D array of shape (m, n) where each row is the interpolated values at grid points for a day
        """


class KrigingInterpolator(Interpolator):
    """
    Kriging Interpolation with a specified variogram model
    """

    def __init__(self, **model_kwargs: Any):
        self.time_range: Union[Tuple, None] = None
        self.model_kwargs = model_kwargs

    def fit(self, xy: np.ndarray, values: np.ndarray, time: pd.DatetimeIndex):
        assert values.shape[0] == xy.shape[0]
        assert len(time) == xy.shape[0]
        assert xy.shape[1] == 2

        self.time_range = (time.min(), time.max())

        self.model = OrdinaryKriging3D(
            xy[:, 0],  # x (longitude)
            xy[:, 1],  # y (latitude)
            (time - self.time_range[0]).days.values,  # t (time in days since start)
            values,
            verbose=False,
            enable_plotting=False,
            **self.model_kwargs,  # type: ignore
        )

    def predict(self, grid: np.ndarray, days: pd.DatetimeIndex) -> np.ndarray:
        assert (
            grid.shape[1] == 2
        ), "grid should have shape (n, 2) where n is the number of points"
        assert self.time_range is not None, "Model not fitted"

        day_indices = (days - self.time_range[0]).days.values

        x = np.tile(grid[:, 0], len(days))
        y = np.tile(grid[:, 1], len(days))
        t = np.repeat(day_indices, len(grid))

        predictions, _ = self.model.execute("points", x, y, t)

        return predictions.reshape(len(days), len(grid)).T

class ScaledKrigingInterpolator(Interpolator):
    """
    Kriging Interpolation with a specified variogram model and input/output scaling.
    """
    def __init__(self, **model_kwargs: Any):
        self.time_min: Union[pd.Timestamp, None] = None
        self.model_kwargs = model_kwargs
        self.xy_scaler = StandardScaler()
        self.time_scaler = StandardScaler()
        self.value_scaler = StandardScaler()

    def fit(self, xy: np.ndarray, values: np.ndarray, time: pd.DatetimeIndex):
        assert values.shape[0] == xy.shape[0], "Mismatch in number of samples between xy and values."
        assert len(time) == xy.shape[0], "Mismatch in number of samples between time and xy."
        assert xy.shape[1] == 2, "Input xy should have shape (n_samples, 2)."

        self.time_min = time.min()
        assert self.time_min is not None, "Time range not defined."

        xy_scaled = self.xy_scaler.fit_transform(xy)

        time_numeric = (time - self.time_min).days.values.reshape(-1, 1)
        time_scaled = self.time_scaler.fit_transform(time_numeric).flatten()

        values_scaled = self.value_scaler.fit_transform(values.reshape(-1, 1)).flatten()

        self.model = OrdinaryKriging3D(
            xy_scaled[:, 0],  # Scaled x (longitude)
            xy_scaled[:, 1],  # Scaled y (latitude)
            time_scaled,      # Scaled t (time in days since self.time_min)
            values_scaled,
            verbose=False,
            enable_plotting=False,
            **self.model_kwargs # type: ignore
        )

    def predict(self, grid: np.ndarray, days: pd.DatetimeIndex) -> np.ndarray:
        assert grid.shape[1] == 2, "Grid should have shape (n_points, 2)."
        assert self.time_min is not None, "Model has not been fitted yet."

        grid_scaled = np.array(self.xy_scaler.transform(grid))

        days_numeric = (days - self.time_min).days.values.reshape(-1, 1)
        days_scaled = np.array(self.time_scaler.transform(days_numeric)).flatten()

        x = np.tile(grid_scaled[:, 0], len(days))
        y = np.tile(grid_scaled[:, 1], len(days))
        t = np.repeat(days_scaled, len(grid))

        predictions_scaled, _ = self.model.execute('points', x, y, t)

        predictions = np.array(self.value_scaler.inverse_transform(predictions_scaled.reshape(-1, 1))).flatten()

        return predictions.reshape(len(days), len(grid)).T


class TimeSubsetInterpolator(Interpolator):
    """
    An Interpolator that trains a model for each day in a specified date range,
    using data from that day and up to `n_days` before and after,
    ensuring each model includes at least one data point.
    """

    def __init__(
        self,
        n_days: int,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        interpolator_factory: Callable[..., Interpolator],
    ):
        self.n_days = n_days
        self.interpolator_factory = interpolator_factory
        self.start_date = start_date
        self.end_date = end_date
        self.models: Dict[datetime.date, Interpolator] = {}

    def fit(self, xy: np.ndarray, values: np.ndarray, time: pd.DatetimeIndex):
        assert (
            values.shape[0] == xy.shape[0]
        ), "Mismatch in number of values and xy points"
        assert xy.shape[1] == 2, "xy should have shape (n_samples, 2)"

        days = pd.date_range(self.start_date, self.end_date, freq="D")

        data_days = time.normalize().unique().sort_values()

        for day in days:
            # Always include data from the current day
            collected_days = [day.normalize()]

            # Collect up to n_days before the current day that have data
            days_before = data_days[data_days < day.normalize()].sort_values(
                ascending=False
            )
            days_before = days_before[: self.n_days]

            # Collect up to n_days after the current day that have data
            days_after = data_days[data_days > day.normalize()].sort_values()
            days_after = days_after[: self.n_days]

            collected_days = list(days_before) + collected_days + list(days_after)

            subset_idx = time.normalize().isin(collected_days)

            if not subset_idx.any():
                continue  # Skip if no data points are found

            print(
                f"Fitting model for {day.date()} with {subset_idx.sum()} points from {[d.date().strftime('%Y-%m-%d') + f'({time.isin([d]).sum()})' for d in collected_days]}"
            )

            model = self.interpolator_factory()
            model.fit(xy[subset_idx], values[subset_idx], time[subset_idx])

            self.models[day.date()] = model

    def predict(self, grid: np.ndarray, days: pd.DatetimeIndex) -> np.ndarray:
        assert grid.shape[1] == 2, "grid should have shape (n_points, 2)"

        predictions = []
        for day in days:
            day_date = day.normalize().date()
            if day_date not in self.models:
                raise ValueError(f"Model for {day_date} not fitted")
            model = self.models[day_date]
            predictions.append(model.predict(grid, pd.DatetimeIndex([day])))

        return np.hstack(predictions)
