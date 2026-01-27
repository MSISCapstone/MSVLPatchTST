import torch
import torch.nn as nn

from PatchTST_physics_integrated.models.stage2_long import LongEncoder
from PatchTST_physics_integrated.models.stage2_short import ShortEncoder
from PatchTST_physics_integrated.models.stage4_head import FusionForecastHead


class Stage4PhysicsModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Encoders
        self.long_encoder = LongEncoder(config)
        self.short_encoder = ShortEncoder(config)

        # Final forecasting head
        self.head = FusionForecastHead(
            d_model=config.d_model,
            out_dim=config.target_dim   # 1 or H
        )

    def forward(self, x_long, x_short):
        """
        x_long  : [B, T_long, C_long]
        x_short : [B, T_short, C_short]
        """

        F_long = self.long_encoder(x_long)     # [B, T1, D]
        F_short = self.short_encoder(x_short) # [B, T2, D]

        y_hat = self.head(F_long, F_short)     # [B, out_dim]
        return y_hat
