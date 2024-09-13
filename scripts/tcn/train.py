import os
import logging
import yaml
import numpy as np
import torch
import wandb
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader

from oadap.prediction.tcn.dataset import WindowedDataset
from oadap.prediction.tcn.model import (
    RegressionTCN,
    RegressionTCNv0,
    SpatialRegressionTCNv1,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a TCN model with the given config"
    )
    parser.add_argument(
        "--config", default="config/tcn/v0.yaml", help="Path to the configuration file"
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_device(device_preference):
    if device_preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_preference)


def load_data(config):
    data_dir = config["data"]["data_dir"]
    X = np.load(os.path.join(data_dir, config["data"]["train_file"]))
    y = np.load(os.path.join(data_dir, config["data"]["label_file"]))
    return X, y


def prepare_datasets(X, y, config, device):
    assert X.shape[0] == y.shape[0], "X and y must have the same number of locations"
    assert (
        X.shape[1] == y.shape[1]
    ), "X and y must have the same number of windows per location"
    nx, nw, nt, nf = X.shape
    _, _, nl = y.shape
    n_val_points = config["data"]["val_points"]
    np.random.seed(config["random_seed"])
    val_inds = np.random.choice(nx, n_val_points, replace=False)
    X_val, y_val = X[val_inds], y[val_inds]
    X, y = np.delete(X, val_inds, axis=0), np.delete(y, val_inds, axis=0)

    X = X.reshape(-1, nt, nf)
    y = y.reshape(-1, nl)
    X_val = X_val.reshape(-1, nt, nf)
    y_val = y_val.reshape(-1, nl)

    # Use the first year for training and the second for testing
    # train_inds = np.arange(X.shape[0])[:(nx - n_val_points) * 365]
    train_inds = np.random.choice(
        X.shape[0], int(X.shape[0] * config["data"]["train_split"]), replace=False
    )
    test_inds = np.setdiff1d(np.arange(X.shape[0]), train_inds)

    train_dataset = WindowedDataset(X[train_inds], y[train_inds], device=device)
    test_dataset = WindowedDataset(
        X[test_inds],
        y[test_inds],
        X_scaler=train_dataset.X_scaler,
        y_scaler=train_dataset.y_scaler,
        device=device,
    )
    val_dataset = WindowedDataset(
        X_val,
        y_val,
        X_scaler=train_dataset.X_scaler,
        y_scaler=train_dataset.y_scaler,
        device=device,
    )

    return train_dataset, test_dataset, val_dataset


def create_dataloaders(train_dataset, test_dataset, val_dataset, config):
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config["training"]["batch_size"]
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"]
    )
    return train_dataloader, test_dataloader, val_dataloader


def create_model(config, device):
    if config["model"]["type"] == "RegressionTCN":
        model = RegressionTCN(
            feature_dim=config["model"]["feature_dim"],
            output_dim=config["model"]["output_dim"],
            hidden_channels=config["model"]["hidden_channels"],
            network_depth=config["model"]["network_depth"],
            filter_width=config["model"]["filter_width"],
            dropout=config["model"]["dropout"],
            activation=config["model"]["activation"],
            use_skip_connections=config["model"]["use_skip_connections"],
        )
    elif config["model"]["type"] == "RegressionTCNv0":
        model = RegressionTCNv0(
            feature_dim=config["model"]["feature_dim"],
            output_dim=config["model"]["output_dim"],
            hidden_channels=config["model"]["hidden_channels"],
            network_depth=config["model"]["network_depth"],
            filter_width=config["model"]["filter_width"],
            dropout=config["model"]["dropout"],
            activation=config["model"]["activation"],
            use_skip_connections=config["model"]["use_skip_connections"],
        )
    elif config["model"]["type"] == "SpatialRegressionTCNv1":
        model = SpatialRegressionTCNv1(
            feature_dim=config["model"]["feature_dim"],
            spatial_dim=config["model"]["spatial_dim"],
            spatial_embedding_dim=config["model"]["spatial_embedding_dim"],
            time_dim=config["model"]["time_dim"],
            output_dim=config["model"]["output_dim"],
            hidden_channels=config["model"]["hidden_channels"],
            network_depth=config["model"]["network_depth"],
            filter_width=config["model"]["filter_width"],
            dropout=config["model"]["dropout"],
            activation=config["model"]["activation"],
            use_skip_connections=config["model"]["use_skip_connections"],
        )
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")

    model = model.to(device)
    # model = torch.compile(model)  # Use torch.compile for potential speedup
    return model


def load_checkpoint(checkpoint_path, model, optimizer, device):
    if not os.path.exists(checkpoint_path):
        logger.warning(
            f"Checkpoint not found at {checkpoint_path}. Starting from scratch."
        )
        return model, optimizer, None, None, 0

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    X_scaler = checkpoint["X_scaler"]
    y_scaler = checkpoint["y_scaler"]
    start_epoch = checkpoint["epoch"] + 1
    logger.info(f"Resumed training from epoch {start_epoch}")
    return model, optimizer, X_scaler, y_scaler, start_epoch


def train(
    model,
    train_dataloader,
    test_dataloader,
    val_dataloader,
    config,
    device,
    start_epoch=0,
):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"]["learning_rate"])
    )
    criterion = (
        nn.MSELoss() if config["training"]["loss_function"] == "mse" else nn.L1Loss()
    )

    if config["training"]["resume_from_checkpoint"]:
        checkpoint_path = os.path.join(
            config["training"]["checkpoint_dir"],
            config["training"]["resume_checkpoint"],
        )
        model, optimizer, X_scaler, y_scaler, start_epoch = load_checkpoint(
            checkpoint_path, model, optimizer, device
        )

    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dataloader)
        avg_test_loss = evaluate(model, test_dataloader, criterion)
        avg_val_loss = evaluate(model, val_dataloader, criterion)

        if epoch % config["wandb"]["log_interval"] == 0:
            wandb.log(
                {
                    "epoch": epoch,
                    "avg_train_loss": avg_train_loss,
                    "avg_test_loss": avg_test_loss,
                    # "mu_test_loss": avg_test_loss[0],
                    # "pca1_test_loss": avg_test_loss[1],
                    # "pca2_test_loss": avg_test_loss[2],
                    "avg_val_loss": avg_val_loss,
                }
            )

        if epoch % config["wandb"]["save_interval"] == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                train_dataloader.dataset.X_scaler,
                train_dataloader.dataset.y_scaler,
                config,
            )

        logger.info(
            f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )


def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            y_pred = model(X_batch)
            running_loss += criterion(y_pred, y_batch).item()
    return running_loss / len(dataloader)


def evaluate_per_channel(model, dataloader, criterion):
    model.eval()
    running_loss = {}
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            y_pred = model(X_batch)

            channels = y_pred.shape[-1]
            for i in range(channels):
                if i not in running_loss:
                    running_loss[i] = 0.0
                running_loss[i] += criterion(y_pred[:, i], y_batch[:, i]).item()
    for i in running_loss:
        running_loss[i] /= len(dataloader)
    return running_loss


def save_checkpoint(model, optimizer, epoch, X_scaler, y_scaler, config):
    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "X_scaler": X_scaler,
            "y_scaler": y_scaler,
        },
        checkpoint_path,
    )
    wandb.save(checkpoint_path)

    logger.info(f"Saved checkpoint and scalers for epoch {epoch}")


if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config)
    device = set_device(config["device"])
    logger.info(f"Using device: {device}")

    wandb.init(
        project=config["wandb"]["project"],
        config=config,
        resume=config["training"]["resume_from_checkpoint"],
    )

    X, y = load_data(config)
    train_dataset, test_dataset, val_dataset = prepare_datasets(X, y, config, device)
    train_dataloader, test_dataloader, val_dataloader = create_dataloaders(
        train_dataset, test_dataset, val_dataset, config
    )

    model = create_model(config, device)
    wandb.watch(model)

    train(model, train_dataloader, test_dataloader, val_dataloader, config, device)

    wandb.finish()
