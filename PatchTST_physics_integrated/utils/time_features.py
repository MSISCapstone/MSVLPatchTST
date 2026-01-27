import torch
import math

def add_hour_features(X, hour_idx):
    """
    X        : [B, L, C]
    hour_idx : [B, L]  (0–23)
    return   : [B, L, C+2]
    """
    hour_rad = 2 * math.pi * hour_idx.float() / 24.0
    hour_sin = torch.sin(hour_rad).unsqueeze(-1)
    hour_cos = torch.cos(hour_rad).unsqueeze(-1)

    return torch.cat([X, hour_sin, hour_cos], dim=-1)

