import torch
import torch.nn as nn

class FusionForecastHead(nn.Module):
    """
    Fuses long+short representations and produces forecast output.

    Inputs:
      F_long  : [B, C_long,  Np, D]
      F_short : [B, C_short, Np, D]

    Output:
      y_hat   : [B, out_dim]
    """
    def __init__(self, d_model: int = 128, out_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, out_dim)

    def forward(self, F_long: torch.Tensor, F_short: torch.Tensor) -> torch.Tensor:
        # 1) concat channels
        F = torch.cat([F_long, F_short], dim=1)  # [B, C_total, Np, D]

        # 2) pool over channels and patches (mean)
        z = F.mean(dim=1).mean(dim=1)           # [B, D]

        # 3) head
        z = self.dropout(z)
        y_hat = self.fc(z)                      # [B, out_dim]
        return y_hat

