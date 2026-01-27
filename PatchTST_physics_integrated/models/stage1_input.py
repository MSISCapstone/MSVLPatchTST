import torch.nn as nn
from layers.revin import RevIN
from utils.time_features import add_hour_features

#  PROPOSED CHANNEL GROUPING (DATA-DRIVEN)
LONG_CHANNELS = [
    0,1,2,3,4,5,6,7,8,9,10,
    16,17,18,19
]

SHORT_CHANNELS = [
    11,12,13,14,15
]

AUX_CHANNELS = [20, 21]

class Stage1Input(nn.Module):
    def __init__(self):
        super().__init__()
        self.revin = RevIN(num_features=22)

    def forward(self, X_weather, hour_idx):
        """
        X_weather : [B, L, 20]
        hour_idx  : [B, L]
        """
        # Add cyclic hour features
        X = add_hour_features(X_weather, hour_idx)   # [B, L, 22]

        # RevIN normalization
        X = self.revin(X)

        # Channel grouping
        X_long  = X[:, :, LONG_CHANNELS]
        X_short = X[:, :, SHORT_CHANNELS]
        X_aux   = X[:, :, AUX_CHANNELS]

        return X_long, X_short, X_aux

