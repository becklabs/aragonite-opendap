import torch
import yaml
from .model import RegressionTCN, RegressionTCNv0, SpatialRegressionTCNv1

def load_config(config_path):
    with open(config_path, 'r') as f:
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

def load_model(config, checkpoint_path, device):
    if config['model']['type'] == "RegressionTCN":
        model = RegressionTCN(
            feature_dim=config['model']['feature_dim'],
            output_dim=config['model']['output_dim'],
            hidden_channels=config['model']['hidden_channels'],
            network_depth=config['model']['network_depth'],
            filter_width=config['model']['filter_width'],
            dropout=config['model']['dropout'],
            activation=config['model']['activation'],
            use_skip_connections=config['model']['use_skip_connections']
        )
    elif config['model']['type'] == "RegressionTCNv0":
        model = RegressionTCNv0(
            feature_dim=config['model']['feature_dim'],
            output_dim=config['model']['output_dim'],
            hidden_channels=config['model']['hidden_channels'],
            network_depth=config['model']['network_depth'],
            filter_width=config['model']['filter_width'],
            dropout=config['model']['dropout'],
            activation=config['model']['activation'],
            use_skip_connections=config['model']['use_skip_connections']
        )
    elif config['model']['type'] == "SpatialRegressionTCNv1":
        model = SpatialRegressionTCNv1(
            feature_dim=config['model']['feature_dim'],
            spatial_dim=config['model']['spatial_dim'],
            spatial_embedding_dim=config['model']['spatial_embedding_dim'],
            time_dim=config['model']['time_dim'],
            output_dim=config['model']['output_dim'],
            hidden_channels=config['model']['hidden_channels'],
            network_depth=config['model']['network_depth'],
            filter_width=config['model']['filter_width'],
            dropout=config['model']['dropout'],
            activation=config['model']['activation'],
            use_skip_connections=config['model']['use_skip_connections']
        )
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, checkpoint['X_scaler'], checkpoint['y_scaler']