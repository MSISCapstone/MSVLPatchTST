import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignShortToLong(nn.Module):
    """
    Align short patch tokens to long patch grid using adaptive average pooling.

    Input :
      H_short [B, C_short, Np_short, D]
      Np_long (int)
    Output:
      H_short_aligned [B, C_short, Np_long, D]
    """
    def __init__(self):
        super().__init__()

    def forward(self, H_short: torch.Tensor, Np_long: int) -> torch.Tensor:
        B, C, Ns, D = H_short.shape

        # reshape to pool over patch dimension Ns
        x = H_short.permute(0, 1, 3, 2)          # [B, C, D, Ns]
        x = x.reshape(B * C * D, 1, Ns)          # [B*C*D, 1, Ns]

        x = F.adaptive_avg_pool1d(x, output_size=Np_long)  # [B*C*D, 1, Np_long]

        x = x.reshape(B, C, D, Np_long).permute(0, 1, 3, 2)  # [B, C, Np_long, D]
        return x

