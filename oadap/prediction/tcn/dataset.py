import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class TempDataset(Dataset):
    def __init__(self, X, y, X_scaler=None, y_scaler=None, device='cpu'):
        N, T, C = X.shape
        X = X.reshape(N * T, C)
        if X_scaler is None:
            X_scaler = StandardScaler()
            X = X_scaler.fit_transform(X)
        else:
            X = X_scaler.transform(X)
        X = X.reshape(N, T, C)
        
        if y_scaler is None:
            y_scaler = StandardScaler()
            y = y_scaler.fit_transform(y)
        else:
            y = y_scaler.transform(y)
        
        self.X_scaler = X_scaler
        self.y_scaler = y_scaler

        self.device = torch.device(device)
        self.X = torch.FloatTensor(X).to(self.device)
        self.y = torch.FloatTensor(y).to(self.device)

        self.device = device

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
