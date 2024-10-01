import logging
import pandas as pd
import joblib
import PyCO2SYS

import numpy as np
from typing import Tuple, List
from tqdm import tqdm

from math import ceil

import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial import KDTree

from .tcn.preprocessing import (
    group,
    create_windows,
    reconstruct_field,
    reconstruct_data,
)
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

class DICRegressionModule:
    SEASONS = {
        "October - March": lambda column: column.month.isin([10, 11, 12, 1, 2, 3]),
        "April - June": lambda column: column.month.isin([4, 5, 6]),
        "July - September": lambda column: column.month.isin([7, 8, 9]),
    }

    def __init__(self, checkpoint_path: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.checkpoint = joblib.load(checkpoint_path)


    def predict(
        self,
        salinity_field: np.ndarray,
        temperature_field: np.ndarray,
        chlorophyll_surface: np.ndarray,
        time: pd.DatetimeIndex,
        batch_size: int = 100_000,
        pbar: bool = False,
    ) -> np.ndarray:
        """
        Predict the regression output based on input fields and time.

        Parameters:
        - salinity_field (np.ndarray): Salinity data with shape (nx, nt, nz).
        - temperature_field (np.ndarray): Temperature data with shape (nx, nt, nz).
        - chlorophyll_surface (np.ndarray): Chlorophyll surface data with shape (nx, nt).
        - time (pd.DatetimeIndex): Time indices with length nt.
        - batch_size (int): Number of samples to process in each batch.

        Returns:
        - result (np.ndarray): Predicted results with shape (nx, nt, nz).
        """

        nx, nt, nz = salinity_field.shape
        self.logger.debug(f"Unpacked shapes - nx: {nx}, nt: {nt}, nz: {nz}")

        # Broadcast chlorophyll_surface to match the shape of salinity and temperature fields
        chlorophyll_field = np.broadcast_to(
            chlorophyll_surface[:, :, np.newaxis], (nx, nt, nz)
        )

        # Stack the features along the last axis to create X
        X = np.stack([temperature_field, salinity_field, chlorophyll_field], axis=-1) # (nx, nt, nz, 3)

        # Reshape X to 2D array of shape (n_samples, 3)
        X_flat = X.reshape(-1, 3)
        n_samples = X_flat.shape[0]

        # Repeat the time array to match the number of samples
        time_repeated = np.repeat(time.values, nx * nz)
        time_flat = pd.to_datetime(time_repeated)

        # Create a mask for each season
        season_masks = {}
        for season, condition in self.SEASONS.items():
            mask = condition(time_flat)
            season_masks[season] = mask

        # Initialize result array
        result_flat = np.full(n_samples, np.nan)

        # Loop over seasons
        for season in self.SEASONS:
            mask = season_masks[season]
            if not np.any(mask):
                self.logger.warning(
                    f"No data available for season: {season}. Skipping."
                )
                continue  # Skip if no data for this season

            # Get the indices where mask is True
            mask_indices = np.where(mask)[0]
            total_mask_samples = len(mask_indices)
            total_batches = ceil(total_mask_samples / batch_size)

            # Loop over batches
            if pbar:
                batch_iterator = tqdm(range(total_batches), desc=f"Processing Batches for Season: {season}", unit="batch")
            else:
                batch_iterator = range(total_batches)

            for batch_num in batch_iterator:
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, total_mask_samples)

                # Get the batch indices
                batch_indices = mask_indices[start_idx:end_idx]

                # Select data for the current batch
                X_batch = X_flat[batch_indices]

                # Transform the data
                X_batch_transformed = self._transform(X_batch, season=season)

                # Make predictions
                model = self.checkpoint[season]["model"]
                y_pred, _ = model.predict(X_batch_transformed)

                # Inverse transform the predictions
                y_pred = self._inverse_transform(y_pred, season=season)

                # Flatten y_pred to 1D if necessary
                y_pred = y_pred.ravel()

                # Assign predictions to the result array
                result_flat[batch_indices] = y_pred

        # Reshape result back to (nx, nt, nz)
        result = result_flat.reshape(nx, nt, nz)
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
        """
        Initialize the CO2SYSAragoniteModule.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Initialized CO2SYSAragoniteModule.")

    def predict(
        self,
        salinity_field: np.ndarray,  
        talk_field: np.ndarray,  
        dic_field: np.ndarray,  
        temperature_field: np.ndarray,  
        depth: np.ndarray,  
        batch_size: int = 100_000,  
        pbar: bool = False,
    ) -> np.ndarray:
        """
        Predict saturation aragonite using PyCO2SYS with batch processing.

        Parameters:
        - salinity_field (np.ndarray): Salinity data with shape (nx, nt, nz).
        - talk_field (np.ndarray): Total Alkalinity data with shape (nx, nt, nz).
        - dic_field (np.ndarray): DIC data with shape (nx, nt, nz).
        - temperature_field (np.ndarray): Temperature data with shape (nx, nt, nz).
        - depth (np.ndarray): Depth data with shape (nx, nz).
        - batch_size (int): Number of samples to process in each batch.

        Returns:
        - np.ndarray: Saturation aragonite output with shape (nx, nt, nz).
        """
        self.logger.info("Starting prediction process with batch processing.")
        self.logger.debug(
            f"Input shapes - salinity_field: {salinity_field.shape}, "
            f"talk_field: {talk_field.shape}, dic_field: {dic_field.shape}, "
            f"temperature_field: {temperature_field.shape}, depth: {depth.shape}"
        )
        self.logger.debug(f"Batch size set to: {batch_size}")

        nx, nt, nz = salinity_field.shape
        self.logger.debug(f"Unpacked shapes - nx: {nx}, nt: {nt}, nz: {nz}")

        # Expand depth to match the shape of other fields
        self.logger.info("Broadcasting depth to match salinity and temperature fields.")
        depth_expanded = np.broadcast_to(depth[:, np.newaxis, :], (nx, nt, nz))
        self.logger.debug(
            f"depth_expanded shape after broadcasting: {depth_expanded.shape}"
        )

        # Reshape all fields to 1D arrays for processing
        self.logger.info("Reshaping input fields for batch processing.")
        talk_flat = talk_field.reshape(-1)
        dic_flat = dic_field.reshape(-1)
        salinity_flat = salinity_field.reshape(-1)
        temperature_out_flat = temperature_field.reshape(-1)
        pressure_out_flat = depth_expanded.reshape(-1)
        n_samples = salinity_flat.shape[0]
        self.logger.debug(f"Total number of samples: {n_samples}")

        # Calculate the number of batches
        total_batches = ceil(n_samples / batch_size)
        self.logger.info(f"Total batches to process: {total_batches}")

        # Initialize the result array
        result_flat = np.empty(n_samples, dtype=np.float64)
        self.logger.debug(f"Initialized result_flat with shape: {result_flat.shape}")

        # Process data in batches
        if pbar:
            batch_iterator = tqdm(range(total_batches), desc="Processing Batches", unit="batch")
        else:
            batch_iterator = range(total_batches)
        for batch_num in batch_iterator:
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            current_batch_size = end_idx - start_idx

            # Extract batch data
            batch_talk = talk_flat[start_idx:end_idx]
            batch_dic = dic_flat[start_idx:end_idx]
            batch_salinity = salinity_flat[start_idx:end_idx]
            batch_temperature_out = temperature_out_flat[start_idx:end_idx]
            batch_pressure_out = pressure_out_flat[start_idx:end_idx]

            self.logger.debug(
                f"Batch data shapes - talk: {batch_talk.shape}, dic: {batch_dic.shape}, "
                f"salinity: {batch_salinity.shape}, temperature_out: {batch_temperature_out.shape}, "
                f"pressure_out: {batch_pressure_out.shape}"
            )

            # Perform CO2SYS calculations
            try:
                co2sys = PyCO2SYS.sys(
                    par1_type=1,  # Total Alkalinity input 1
                    par1=batch_talk,  # Total Alkalinity
                    par2_type=2,  # DIC input 2
                    par2=batch_dic,  # DIC
                    salinity=batch_salinity, # Practical Salinity # type: ignore
                    temperature=25,  # Temperature at which par1 and par2 are provided in °C
                    pressure=0,  # Water pressure at which par1 and par2 are provided in dbar
                    temperature_out=batch_temperature_out,  # Output temperature in °C
                    pressure_out=batch_pressure_out,  # Output pressure in dbar
                    opt_pH_scale=4,  # NBS scale
                    opt_k_carbonic=1,  # Roy et al. 1993
                    opt_k_bisulfate=1,  # Dickson et al. 1990
                    opt_total_borate=2,  # Lee et al. 2010
                    opt_k_fluoride=1,  # Dickson and Riley 1979
                )
                self.logger.debug(f"PyCO2SYS.sys output keys: {co2sys.keys()}")
            except Exception as e:
                self.logger.exception(
                    f"Error during PyCO2SYS.sys execution for batch {batch_num + 1}: {e}"
                )
                raise

            # Extract saturation_aragonite_out and assign to result
            saturation_aragonite = co2sys["saturation_aragonite_out"]
            self.logger.debug(
                f"Saturation aragonite output shape: {saturation_aragonite.shape}"
            )

            # Assign the batch results to the result_flat array
            result_flat[start_idx:end_idx] = saturation_aragonite

        # Reshape the flat result back to (nx, nt, nz)
        result = result_flat.reshape(nx, nt, nz)
        self.logger.debug(f"Final result shape: {result.shape}")

        return result
