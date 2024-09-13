import numpy as np
import pandas as pd
import datetime
from abc import ABC, abstractmethod
from typing import Callable, Union, Tuple, Dict
from pykrige.ok3d import OrdinaryKriging3D
from scipy.interpolate import LinearNDInterpolator
from sklearn.neighbors import KDTree
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
    def __init__(self, **model_kwargs: Dict):
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
            **self.model_kwargs # type: ignore
        )
    
    def predict(self, grid: np.ndarray, days: pd.DatetimeIndex) -> np.ndarray:
        assert grid.shape[1] == 2, "grid should have shape (n, 2) where n is the number of points"
        assert self.time_range is not None, "Model not fitted"
        
        day_indices = (days - self.time_range[0]).days.values
        
        x = np.tile(grid[:, 0], len(days))
        y = np.tile(grid[:, 1], len(days))
        t = np.repeat(day_indices, len(grid))
        
        predictions, _ = self.model.execute('points', x, y, t)
        
        return predictions.reshape(len(days), len(grid)).T


class ScaledKrigingInterpolator(Interpolator):
    """
    Kriging Interpolation with a specified variogram model and input/output scaling.
    """
    def __init__(self, **model_kwargs: Dict):
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


class Fast3DInterpolator(Interpolator):
    """
    A faster alternative to kriging using LinearNDInterpolator and KDTree
    for 3D interpolation (x, y, time)
    """
    def __init__(self, n_neighbors: int = 10):
        self.n_neighbors = n_neighbors
        self.interpolator: Union[LinearNDInterpolator, None] = None
        self.kdtree: Union[KDTree, None] = None
        self.time_range: Union[Tuple, None] = None

    def fit(self, xy: np.ndarray, values: np.ndarray, time: pd.DatetimeIndex):
        assert values.shape[0] == xy.shape[0]
        assert len(time) == xy.shape[0]
        assert xy.shape[1] == 2

        self.time_range = (time.min(), time.max())
        time_days = (time - self.time_range[0]).days.values

        points_3d = np.column_stack((xy, time_days.reshape(-1, 1)))

        self.interpolator = LinearNDInterpolator(points_3d, values, fill_value=np.nan)

        self.kdtree = KDTree(points_3d)

    def predict(self, grid: np.ndarray, days: pd.DatetimeIndex) -> np.ndarray:
        assert grid.shape[1] == 2, "grid should have shape (n, 2) where n is the number of points"
        assert self.time_range is not None, "Model not fitted"
        assert self.interpolator is not None, "Model not fitted"
        assert self.kdtree is not None, "Model not fitted"

        day_indices = (days - self.time_range[0]).days.values

        query_points = np.column_stack((
            np.tile(grid[:, 0], len(days)),
            np.tile(grid[:, 1], len(days)),
            np.repeat(day_indices, len(grid))
        ))

        results = self.interpolator(query_points)

        # Handle NaN values using KNN
        nan_mask = np.isnan(results)
        if np.any(nan_mask):
            nan_points = query_points[nan_mask]
            _, indices = self.kdtree.query(nan_points, k=self.n_neighbors)
            results[nan_mask] = np.nanmean(results[indices], axis=1)

        return results.reshape(len(days), len(grid)).T

class TimeSubsetInterpolator(Interpolator):
    """
    Interpolates data using a given time subset
    Assumes time is on a regular daily interval
    """
    def __init__(self, n_days: int, interpolator_factory: Callable[..., Interpolator]):
        self.n_days = n_days
        self.interpolator_factory = interpolator_factory
        self.models: Dict[datetime.date, Interpolator] = {}

    def fit(self, xy: np.ndarray, values: np.ndarray, time: pd.DatetimeIndex):
        assert values.shape[0] == xy.shape[0]
        assert xy.shape[1] == 2

        days = time.unique().sort_values()
        for i in range(self.n_days, len(days) - self.n_days):
            subset = days[i-self.n_days:i+self.n_days+1]
            subset_idx = time.isin(subset)
            print(f"Fitting model for {days[i].date()} with {sum(subset_idx)} points")

            model = self.interpolator_factory()
            model.fit(xy[subset_idx], values[subset_idx], time[subset_idx])

            self.models[days[i].date()] = model
    
    def predict(self, grid: np.ndarray, days: pd.DatetimeIndex) -> np.ndarray:
        assert grid.shape[1] == 2, "grid should have shape (n, 2) where n is the number of points"
        
        predictions = []
        for day in days:
            if day.date() not in self.models:
                raise ValueError(f"Model for {day.date()} not fitted")
            model = self.models[day.date()]
            predictions.append(model.predict(grid, pd.DatetimeIndex([day])))
        
        return np.hstack(predictions)

class PointsSubsetInterpolator(Interpolator):
    """
    Interpolates data using a given number of points around each day.
    First uses points from the day itself, then from surrounding days in order of proximity.
    """
    def __init__(self, n_points: int, interpolator_factory: Callable[..., Interpolator]):
        self.n_points = n_points
        self.interpolator_factory = interpolator_factory
        self.models: Dict[datetime.date, Interpolator] = {}
    
    def fit(self, xy: np.ndarray, values: np.ndarray, time: pd.DatetimeIndex):
        assert values.shape[0] == xy.shape[0], "values and xy must have the same number of rows"
        assert xy.shape[1] == 2, "xy should have shape (n_samples, 2)"
        
        # Ensure time is in date format
        time_dates = time.normalize()
        days = np.array(sorted(time_dates.unique()))
        
        for i, current_day in enumerate(days):
            included_days = [current_day]
            subset_idx = (time_dates == current_day)
            
            offset = 1
            while subset_idx.sum() < self.n_points and (i - offset >= 0 or i + offset < len(days)):
                # Include previous day
                if i - offset >= 0:
                    prev_day = days[i - offset]
                    included_days.append(prev_day)
                # Include next day
                if i + offset < len(days):
                    next_day = days[i + offset]
                    included_days.append(next_day)
                subset_idx = time_dates.isin(included_days)
                offset += 1

            num_points = subset_idx.sum()
            if num_points == 0:
                raise ValueError(f"No data points available to fit model for {current_day.date()}")
            
            print(f"Fitting model for {current_day.date()} with {num_points} points")
            model = self.interpolator_factory()
            model.fit(xy[subset_idx], values[subset_idx], time[subset_idx])
            self.models[current_day.date()] = model
        
    def predict(self, grid: np.ndarray, days: pd.DatetimeIndex) -> np.ndarray:
        assert grid.shape[1] == 2, "grid should have shape (n_points, 2)"
        
        predictions = []
        for day in days:
            day_date = day.normalize().date()
            if day_date not in self.models:
                raise ValueError(f"Model for {day_date} not fitted")
            model = self.models[day_date]
            day_prediction = model.predict(grid, pd.DatetimeIndex([day]))
            predictions.append(day_prediction)
        
        return np.hstack(predictions)