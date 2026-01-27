import torch
import torch.nn as nn
import torch.nn.functional as F

class ReplicationPad1dPatchifyShort(nn.Module):
    """
    Short-channel path: ReplicationPad1d -> Patchify
    Input : X_short [B, L, C_short]
    Output: patches [B, C_short, Np, patch_len]
    """
    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)

    def forward(self, x_short: torch.Tensor) -> torch.Tensor:
        B, L, C = x_short.shape
        x = x_short.permute(0, 2, 1)  # [B, C, L]

        if L < self.patch_len:
            pad_right = self.patch_len - L
        else:
            remainder = (L - self.patch_len) % self.stride
            pad_right = (self.stride - remainder) % self.stride

        if pad_right > 0:
            x = F.pad(x, (0, pad_right), mode="replicate")

        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return patches


class ShortPatchEmbedding(nn.Module):
    """
    Converts short-channel patches into tokens
    Input : patches [B, C_short, Np, patch_len]
    Output: tokens  [B, C_short, Np, d_model]
    """
    def __init__(self, patch_len: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, patches):
        B, C, Np, P = patches.shape
        patches = patches.reshape(B * C * Np, P)
        tokens = self.proj(patches)
        tokens = tokens.reshape(B, C, Np, -1)
        return tokens


class ShortChannelTransformer(nn.Module):
    """
    Transformer encoder for short-channel tokens
    Input : tokens [B, C_short, Np, d_model]
    Output: H_short [B, C_short, Np, d_model]
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

    def forward(self, tokens):
        B, C, Np, D = tokens.shape
        x = tokens.reshape(B * C, Np, D)
        x = self.encoder(x)
        H_short = x.reshape(B, C, Np, D)
        return H_short

