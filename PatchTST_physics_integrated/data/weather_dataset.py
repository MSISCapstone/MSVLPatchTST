import torch
from torch.utils.data import Dataset

class WeatherDataset(Dataset):
    def __init__(self, X, hours, context_window):
        self.X = X
        self.hours = hours
        self.L = context_window

    def __len__(self):
        return len(self.X) - self.L + 1

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.L]
        h = self.hours[idx : idx + self.L]
        return torch.from_numpy(x), torch.from_numpy(h)

X_all = df[feature_cols].astype("float32").values   # [N,20]
hour_all = df["date"].dt.hour.astype("int64").values
