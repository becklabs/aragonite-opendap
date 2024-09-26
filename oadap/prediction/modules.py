import pandas as pd
import joblib
import PyCO2SYS

import numpy as np
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial import KDTree

from .tcn.preprocessing import group, create_windows, reconstruct_field, reconstruct_data
from .tcn.utils import load_config, set_device, load_model


class TCNModule:

    def __init__(
        self,
        grid_xy: np.ndarray,
        grid_neighbor_inds: np.ndarray,
        fvcom_xy: np.ndarray,
        fvcom_h: np.ndarray,
        fvcom_phi: np.ndarray,
        fvcom_climatologies: np.ndarray,
        config_path: str,
        checkpoint_path: str,
    ):

        self.config = load_config(config_path)
        self.device = set_device(self.config["inference"]["device"])
        self.model, self.X_scaler, self.y_scaler = load_model(
            config=self.config, checkpoint_path=checkpoint_path, device=self.device
        )

        self.grid_xy = grid_xy.reshape(-1, 2)  # (gp, 2)
        self.grid_neighbor_inds = grid_neighbor_inds.reshape(
            -1, self.config["features"]["n_neighbors"]
        )  # (gp, n_neighbors)
        self.grid_neighbors_mask = np.all(
            self.grid_neighbor_inds != -1, axis=1
        )  # (gp,)

        assert self.grid_xy.shape[0] == self.grid_neighbor_inds.shape[0]

        self.fvcom_xy = fvcom_xy  # (np, 2)
        self.fvcom_h = fvcom_h  # (np, 1)
        self.fvcom_phi = fvcom_phi  # (np, nz, 2)
        self.fvcom_day_climatologies = fvcom_climatologies  # (np, n_days, nz)

        tree = KDTree(self.fvcom_xy)
        dist, self.grid_idx_map = tree.query(self.grid_xy)

    def predict(
        self, surface: np.ndarray, time: pd.DatetimeIndex
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:

        nx, nt = surface.shape
        assert nx == self.grid_xy.shape[0]
        assert nt == len(time)

        climatologies = self.fvcom_day_climatologies[:, time.day_of_year - 1, :]
        climatologies_surface = climatologies[:, :, 0]
        anomalies_surface = surface - climatologies_surface[self.grid_idx_map]

        anomalies_grouped = group(
            data=anomalies_surface, neighbor_inds=self.grid_neighbor_inds
        )

        X = np.stack(
            (
                anomalies_grouped[self.grid_neighbors_mask][..., 0],
                anomalies_grouped[self.grid_neighbors_mask][..., 1],
                anomalies_grouped[self.grid_neighbors_mask][..., 2],
                anomalies_grouped[self.grid_neighbors_mask][..., 3],
                anomalies_grouped[self.grid_neighbors_mask][..., 4],
                climatologies_surface[self.grid_idx_map][self.grid_neighbors_mask],
                np.repeat(
                    self.fvcom_xy[self.grid_idx_map][self.grid_neighbors_mask][:, 0][
                        ..., np.newaxis
                    ],
                    nt,
                    axis=1,
                ),
                np.repeat(
                    self.fvcom_xy[self.grid_idx_map][self.grid_neighbors_mask][:, 1][
                        ..., np.newaxis
                    ],
                    nt,
                    axis=1,
                ),
                np.repeat(
                    self.fvcom_h[self.grid_idx_map][self.grid_neighbors_mask],
                    nt,
                    axis=1,
                ),
            ),
            axis=2,
        )

        X_windowed = create_windows(
            X,
            window_size=self.config["features"]["window_size"],
            sampling_rate=self.config["features"]["sampling_rate"],
            stride=self.config["features"]["stride"],
        )
        nx, nw, nt, nf = X_windowed.shape

        time_windowed = pd.to_datetime(
            create_windows(
                np.array(time)[..., np.newaxis],
                window_size=self.config["features"]["window_size"],
                sampling_rate=self.config["features"]["sampling_rate"],
                stride=self.config["features"]["stride"],
                last_only=True,
            ).flatten()
        )

        X_windowed = X_windowed.reshape(nx * nw * nt, nf)
        X_windowed_scaled = self.X_scaler.transform(X_windowed)
        X_windowed_scaled = X_windowed_scaled.reshape(nx * nw, nt, nf)

        dataset = TensorDataset(torch.FloatTensor(X_windowed_scaled))
        dataloader = DataLoader(
            dataset, batch_size=self.config["inference"]["batch_size"]
        )

        preds: List[np.ndarray] = []

        with torch.no_grad():
            for batch in dataloader:
                X_batch = batch[0].to(self.device)
                y_pred = self.model(X_batch)
                preds.append(y_pred.cpu().numpy())

        preds_arr = np.vstack(preds)
        preds_arr = self.y_scaler.inverse_transform(preds_arr)

        preds_arr = preds_arr.reshape(nx, nw, -1)

        T_bar = preds_arr[..., 0]
        q = preds_arr[..., 1:]

        pred_anomalies = reconstruct_field(
            phi=self.fvcom_phi[self.grid_idx_map][self.grid_neighbors_mask],
            mu=T_bar,
            q=q,
        )
        pred_data = reconstruct_data(
            anomalies=pred_anomalies,
            time=time_windowed,
            climatology_days=self.fvcom_day_climatologies[self.grid_idx_map][
                self.grid_neighbors_mask
            ],
        )

        return pred_data, self.grid_xy[self.grid_neighbors_mask], time_windowed


class TAlkRegressionModule:
    def __init__(self, checkpoint_path: str):
        self.model = joblib.load(checkpoint_path)

    def predict(self, salinity: np.ndarray) -> np.ndarray:
        input_shape = salinity.shape
        X = salinity.reshape(-1, 1)
        y_pred = self.model.predict(X)
        y_pred = np.array(y_pred).reshape(input_shape)
        return y_pred

    def predict_df(self, data: pd.DataFrame, salinity_col: str) -> pd.Series:
        X = data[salinity_col].values.reshape(-1, 1) # type: ignore
        y_pred = self.model.predict(X)
        return pd.Series(y_pred)


class DICRegressionModule:
    SEASONS = {
        "October - March": lambda column: column.dt.month.isin([10, 11, 12, 1, 2, 3]),
        "April - June": lambda column: column.dt.month.isin([4, 5, 6]),
        "July - September": lambda column: column.dt.month.isin([7, 8, 9]),
    }

    def __init__(self, checkpoint_path: str):
        self.checkpoint = joblib.load(checkpoint_path)

    def predict(
        self,
        salinity_field: np.ndarray,  # (nx, nt, nz)
        temperature_field: np.ndarray,  # (nx, nt, nz)
        chlorophyll_surface: np.ndarray,  # (nx, nt)
        time: pd.DatetimeIndex,  # (nt,)
    ):
        nx, nt, nz = salinity_field.shape

        # Broadcast chlorophyll_surface to match the shape of salinity and temperature fields
        chlorophyll_field = np.broadcast_to(
            chlorophyll_surface[:, :, np.newaxis], (nx, nt, nz)
        )

        # Stack the features along the last axis to create X
        X = np.stack([temperature_field, salinity_field, chlorophyll_field], axis=-1)  # shape (nx, nt, nz, 3)

        # Reshape X to 2D array of shape (n_samples, 3)
        X_flat = X.reshape(-1, 3)
        n_samples = X_flat.shape[0]

        # Repeat the time array to match the number of samples
        # Each time value is repeated nx * nz times
        time_repeated = np.repeat(time.values, nx * nz)
        time_flat = pd.to_datetime(time_repeated)

        # Create a mask for each season
        season_masks = {season: self.SEASONS[season](time_flat) for season in self.SEASONS}

        # Initialize result array
        result_flat = np.full(n_samples, np.nan)

        # Loop over seasons
        for season in self.SEASONS:
            mask = season_masks[season]
            if not np.any(mask):
                continue  # Skip if no data for this season

            # Select data for the current season
            X_season = X_flat[mask]
            X_season_transformed = self._transform(X_season, season=season)

            # Make predictions
            y_pred, _ = self.checkpoint[season]["model"].predict(X_season_transformed)
            y_pred = self._inverse_transform(y_pred, season=season)

            # Assign predictions to the result array
            result_flat[mask] = y_pred

        # Reshape result back to (nx, nt, nz)
        result = result_flat.reshape(nx, nt, nz)
        return result

    def predict_df(
        self,
        data: pd.DataFrame,
        salinity_col: str,
        temperature_col: str,
        chlorophyll_col: str,
        time_col: str,
    ):
        result = pd.Series(index=data.index, dtype=float)
        for season in self.SEASONS:
            mask = self.SEASONS[season](data[time_col])
            X = data[mask][[temperature_col, salinity_col, chlorophyll_col]].values
            X = self._transform(X, season=season)
            y_pred, _ = self.checkpoint[season]["model"].predict(X)
            y_pred = self._inverse_transform(y_pred, season=season).reshape(-1)
            result.loc[mask] = y_pred
        return result

    def _transform(self, X, season: str):
        season_checkpoint = self.checkpoint[season]
        Xm = season_checkpoint["Xm"]
        Xs = season_checkpoint["Xs"]
        return (X - Xm) / Xs

    def _inverse_transform(self, y, season: str):
        season_checkpoint = self.checkpoint[season]
        ym = season_checkpoint["ym"]
        ys = season_checkpoint["ys"]
        return y * ys + ym


class CO2SYSAragoniteModule:
    def __init__(self):
        pass

    def predict(
        self,
        salinity_field: np.ndarray,
        talk_field: np.ndarray,
        dic_field: np.ndarray,
        temperature_field: np.ndarray,
        depth: np.ndarray, # (nx, nz)
    ) -> np.ndarray:
        nx, nt, nz = salinity_field.shape
        depth_expanded = np.broadcast_to(depth[:, np.newaxis, :], (nx, nt, nz))

        co2sys = PyCO2SYS.sys(
            par1_type=1,  # Total Alkalinity input 1
            par1=talk_field.reshape(-1),  # value of the first parameter
            par2_type=2,  # DIC input 2
            par2=dic_field.reshape(-1),  # value of the second parameter
            salinity=salinity_field.reshape(-1), # type: ignore  # practical salinity (default 35)
            temperature=25,  # temperature at which par1 and par2 arguments are provided in °C (default 25 °C)
            pressure=0,  # water pressure at which par1 and par2 arguments are provided in dbar (default 0 dbar)
            temperature_out=temperature_field.reshape(-1),  # temperature at which results will be calculated in °C
            pressure_out=depth_expanded.reshape(-1),  # water pressure at which results will be calculated in dbar
            opt_pH_scale=4,  # NBS 4
            opt_k_carbonic=1,  # Roy et al 1993 1
            opt_k_bisulfate=1,  # Dickson et al 1990 1
            opt_total_borate=2,  # Lee et al 2010 2
            opt_k_fluoride=1,  # Dickson and Riley 1979 1
        )
        return np.array(co2sys["saturation_aragonite_out"]).reshape(nx, nt, nz)

    def predict_df(
        self,
        data: pd.DataFrame,
        salinity_col: str,
        talk_col: str,
        dic_col: str,
        temperature_col: str,
        depth_col: str,
    ) -> pd.Series:
        co2sys = PyCO2SYS.sys(
            par1_type=1,  # Total Alkalinity input 1
            par1=data[talk_col],
            par2_type=2,  # DIC input 2
            par2=data[dic_col],
            salinity=data[salinity_col], # type: ignore  # practical salinity (default 35)
            temperature=25,  # temperature at which par1 and par2 arguments are provided in °C (default 25 °C)
            pressure=0,  # water pressure at which par1 and par2 arguments are provided in dbar (default 0 dbar)
            temperature_out=data[
                temperature_col
            ],  # temperature at which results will be calculated in °C
            pressure_out=data[
                depth_col
            ],  # water pressure at which results will be calculated in dbar
            opt_pH_scale=4,  # NBS 4
            opt_k_carbonic=1,  # Roy et al 1993 1
            opt_k_bisulfate=1,  # Dickson et al 1990 1
            opt_total_borate=2,  # Lee et al 2010 2
            opt_k_fluoride=1,  # Dickson and Riley 1979 1
        )
        return pd.Series(co2sys["saturation_aragonite_out"])
