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

    class CrossGroupAttentionBridge(nn.Module):
    """
    Bi-directional cross attention between long and short channel groups
    at aligned patch resolution.

    Inputs:
      H_long  : [B, C_long,  Np, D]
      H_short : [B, C_short, Np, D]   (already aligned to same Np)

    Outputs:
      F_long  : [B, C_long,  Np, D]   (long enriched by short)
      F_short : [B, C_short, Np, D]   (short enriched by long)
    """
    def __init__(self, d_model=128, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn_long_q = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_short_q = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm_long = nn.LayerNorm(d_model)
        self.norm_short = nn.LayerNorm(d_model)

        self.ff_long = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ff_short = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, H_long, H_short):
        B, C_L, Np, D = H_long.shape
        _, C_S, _, _  = H_short.shape

        # reshape per patch: treat channels as "tokens"
        # Long tokens per patch: [B*Np, C_L, D]
        L = H_long.permute(0, 2, 1, 3).reshape(B*Np, C_L, D)
        S = H_short.permute(0, 2, 1, 3).reshape(B*Np, C_S, D)

        # 1) Long queries attend to Short (inject fast dynamics)
        L_attn, _ = self.attn_long_q(query=L, key=S, value=S)
        L = self.norm_long(L + L_attn)
        L = self.norm_long(L + self.ff_long(L))

        # 2) Short queries attend to Long (inject slow context)
        S_attn, _ = self.attn_short_q(query=S, key=L, value=L)
        S = self.norm_short(S + S_attn)
        S = self.norm_short(S + self.ff_short(S))

        # reshape back to [B, C, Np, D]
        F_long  = L.reshape(B, Np, C_L, D).permute(0, 2, 1, 3)
        F_short = S.reshape(B, Np, C_S, D).permute(0, 2, 1, 3)

        return F_long, F_short
