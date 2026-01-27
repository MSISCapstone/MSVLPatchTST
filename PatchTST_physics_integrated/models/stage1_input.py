import torch.nn as nn
from PatchTST_physics_integrated.layers.revin import RevIN
from PatchTST_physics_integrated.utils.time_features import add_hour_features

LONG_CHANNELS  = [0,1,2,3,4,5,6,7,8,9,10,16,17,18,19]
SHORT_CHANNELS = [11,12,13,14,15]
AUX_CHANNELS   = [20,21]

class Stage1Input(nn.Module):
    def __init__(self):
        super().__init__()
        self.revin = RevIN(num_features=22)

    def forward(self, X_weather, hour_idx):
        X = add_hour_features(X_weather, hour_idx)   # [B,L,22]
        X = self.revin(X)
        return X[:, :, LONG_CHANNELS], X[:, :, SHORT_CHANNELS], X[:, :, AUX_CHANNELS]
