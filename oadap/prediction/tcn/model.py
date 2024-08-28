import torch
import torch.nn as nn
from typing import List
from pytorch_tcn import TCN # type: ignore

class RegressionTCNv0(nn.Module):
    """
    TCN model for regression tasks
    """
    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        hidden_channels: List[int],
        network_depth: int,
        filter_width: int,
        dropout: float,
        activation: str,
        use_skip_connections: bool,
        window_size: int = 20,
    ):
        super(RegressionTCNv0, self).__init__()

        self.tcn = TCN(
            num_inputs=feature_dim,
            num_channels=hidden_channels,
            kernel_size=filter_width,
            dropout=dropout,
            causal=True,
            activation=activation,
            kernel_initializer="xavier_uniform",
            use_skip_connections=use_skip_connections,
            input_shape="NLC",  # (batch, time, features)
            lookahead=0,
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.leaky_relu = nn.LeakyReLU()
        self.final_dense = nn.Linear(hidden_channels[-1], output_dim)

    def forward(self, x):
        # x shape: (batch_size, time_steps, features)
        x = self.tcn(x)  # (batch_size, time_steps, hidden_channels)

        x = x.transpose(1, 2)  # (batch_size, hidden_channels, time_steps)
        x = self.global_pool(x)  # (batch_size, hidden_channels, 1)

        x = x.squeeze(-1)  # (batch_size, hidden_channels)
        x = self.final_dense(x)  # (batch_size, output_dim)
        return x



class RegressionTCN(nn.Module):
    """
    TCN model for regression tasks
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        hidden_channels: List[int],
        network_depth: int,
        filter_width: int,
        dropout: float,
        activation: str,
        use_skip_connections: bool,
        window_size: int = 20,
    ):
        super(RegressionTCN, self).__init__()

        self.tcn = TCN(
            num_inputs=feature_dim,
            num_channels=hidden_channels,
            kernel_size=filter_width,
            dropout=dropout,
            causal=True,
            activation=activation,
            kernel_initializer="xavier_uniform",
            use_skip_connections=use_skip_connections,
            input_shape="NLC",  # (batch, time, features)
            lookahead=0,
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.leaky_relu = nn.LeakyReLU()
        # self.final_dense = nn.Linear(hidden_channels[-1], output_dim)
        # self.final_dense1 = nn.Linear(hidden_channels[-1] * window_size, hidden_channels[-1] * 2)
        # self.final_dense2 = nn.Linear(hidden_channels[-1] * 2, output_dim)

        self.final_dense1 = nn.Linear(hidden_channels[-1], hidden_channels[-1] * 2)
        self.final_dense2 = nn.Linear(hidden_channels[-1] * 2, hidden_channels[-1])
        self.final_dense3 = nn.Linear(hidden_channels[-1], output_dim)

    def forward(self, x):
        # x shape: (batch_size, time_steps, features)
        x = self.tcn(x)  # (batch_size, time_steps, hidden_channels)

        x = x.transpose(1, 2)  # (batch_size, hidden_channels, time_steps)
        x = self.global_pool(x)  # (batch_size, hidden_channels, 1)

        x = x.squeeze(-1)  # (batch_size, hidden_channels)

        x = self.leaky_relu(self.final_dense1(x))  # (batch_size, channels * 2)
        x = self.leaky_relu(self.final_dense2(x))  # (batch_size, channels)
        x = self.final_dense3(x)  # (batch_size, output_dim)

        return x


class SpatialRegressionTCNv0(nn.Module):
    """
    TCN model for regression tasks
    """

    def __init__(
        self,
        feature_dim: int,
        spatial_dim: int,
        time_dim: int,
        output_dim: int,
        hidden_channels: int,
        network_depth: int,
        filter_width: int,
        dropout: float,
        activation: str,
        use_skip_connections: bool,
    ):
        super(SpatialRegressionTCNv0, self).__init__()

        self.spatial_dim = spatial_dim
        self.time_dim = time_dim

        self.tcn = TCN(
            num_inputs=feature_dim,
            num_channels=[hidden_channels] * network_depth,
            kernel_size=filter_width,
            dropout=dropout,
            causal=True,
            activation=activation,
            kernel_initializer="xavier_uniform",
            use_skip_connections=use_skip_connections,
            input_shape="NLC",  # (batch, time, features)
            lookahead=0,
        )

        self.leaky_relu = nn.LeakyReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(hidden_channels + spatial_dim, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, output_dim)

    def forward(self, x):
        spatial_features = x[:, :, -self.spatial_dim :][:, 0, :]
        x = x[:, :, : -self.spatial_dim]

        # x shape: (batch_size, time_steps, features)
        x = self.tcn(x)  # (batch_size, time_steps, hidden_channels)

        x = x.transpose(1, 2)  # (batch_size, hidden_channels, time_steps)
        x = self.global_pool(x)  # (batch_size, hidden_channels, 1)

        # Add spatial features
        x = x.squeeze(-1)  # (batch_size, hidden_channels)

        x = torch.cat(
            [x, spatial_features], dim=1
        )  # (batch_size, hidden_channels + spatial_embedding_dim)

        x = self.leaky_relu(self.fc1(x))  # (batch_size, hidden_channels)
        x = self.fc2(x)  # (batch_size, output_dim)

        return x


class SpatialRegressionTCNv1(nn.Module):
    """
    TCN model for regression tasks
    """

    def __init__(
        self,
        feature_dim: int,
        spatial_dim: int,
        spatial_embedding_dim: int,
        time_dim: int,
        output_dim: int,
        hidden_channels: int,
        network_depth: int,
        filter_width: int,
        dropout: float,
        activation: str,
        use_skip_connections: bool,
    ):
        super(SpatialRegressionTCNv1, self).__init__()

        self.spatial_dim = spatial_dim
        self.time_dim = time_dim

        self.tcn = TCN(
            num_inputs=feature_dim,
            num_channels=[hidden_channels] * network_depth,
            kernel_size=filter_width,
            dropout=dropout,
            causal=True,
            activation=activation,
            kernel_initializer="xavier_uniform",
            use_skip_connections=use_skip_connections,
            input_shape="NLC",  # (batch, time, features)
            lookahead=0,
        )

        self.leaky_relu = nn.LeakyReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.spatial_head = nn.Linear(spatial_dim, spatial_embedding_dim)

        self.fc1 = nn.Linear(hidden_channels + spatial_embedding_dim, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, output_dim)

    def forward(self, x):
        spatial_features = x[:, :, -self.spatial_dim :][:, 0, :]
        x = x[:, :, : -self.spatial_dim]

        # x shape: (batch_size, time_steps, features)
        x = self.tcn(x)  # (batch_size, time_steps, hidden_channels)

        x = x.transpose(1, 2)  # (batch_size, hidden_channels, time_steps)
        x = self.global_pool(x)  # (batch_size, hidden_channels, 1)

        # Add spatial features
        x = x.squeeze(-1)  # (batch_size, hidden_channels)

        spatial_features = self.spatial_head(spatial_features)
        x = torch.cat(
            [x, spatial_features], dim=1
        )  # (batch_size, hidden_channels + spatial_embedding_dim)

        x = self.leaky_relu(self.fc1(x))  # (batch_size, hidden_channels)
        x = self.fc2(x)  # (batch_size, output_dim)

        return x
