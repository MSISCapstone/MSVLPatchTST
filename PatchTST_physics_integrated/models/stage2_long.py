import torch
import torch.nn as nn
import torch.nn.functional as F

from PatchTST_physics_integrated.layers.PatchTST_backbone import PatchTST_backbone

class ReplicationPad1dPatchify(nn.Module):
    """
    Long-channel path: ReplicationPad1d -> unfold into patches.
    Input : X_long [B, L, C_long]
    Output: patches [B, C_long, Np, patch_len]
    """
    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)

    def forward(self, x_long: torch.Tensor) -> torch.Tensor:
        # x_long: [B, L, C]
        B, L, C = x_long.shape

        # We patch along time, easier if we go [B, C, L]
        x = x_long.permute(0, 2, 1)  # [B, C, L]

        # Compute padding so that (L_pad - patch_len) % stride == 0
        if L < self.patch_len:
            pad_right = self.patch_len - L
        else:
            remainder = (L - self.patch_len) % self.stride
            pad_right = (self.stride - remainder) % self.stride

        # ReplicationPad1d: replicate last value at the right
        if pad_right > 0:
            x = F.pad(x, (0, pad_right), mode="replicate")  # pad last dim (time)

        # Now extract patches using unfold
        # result shape: [B, C, Np, patch_len]
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        return patches




class LongChannelEncoder(nn.Module):
    """
    Long-channel encoder:
    ReplicationPad1d -> Patchify -> Patch Embedding -> PatchTST backbone
    """
    def __init__(
        self,
        patch_len: int,
        stride: int,
        c_long: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        from PatchTST_physics_integrated.layers.PatchTST_backbone import PatchTST_backbone
        self.patchify = ReplicationPad1dPatchify(patch_len, stride)

        # Linear projection: patch_len -> d_model
        self.patch_embed = nn.Linear(patch_len, d_model)

        # PatchTST backbone (channel-independent)
        self.encoder = PatchTST_backbone(
            c_in=1,                  # single channel at a time
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.c_long = c_long
        self.d_model = d_model

    def forward(self, x_long):
        """
        x_long: [B, L, C_long]
        return: H_long [B, C_long, Np, d_model]
        """
        # Patchify
        patches = self.patchify(x_long)   # [B, C_long, Np, patch_len]

        B, C, Np, P = patches.shape

        #  Prepare for embedding: merge B & C
        patches = patches.reshape(B * C, Np, P)   # [B*C, Np, patch_len]

        #Patch embedding
        tokens = self.patch_embed(patches)         # [B*C, Np, d_model]

        # Transformer encoder (PatchTST)
        enc = self.encoder(tokens)                  # [B*C, Np, d_model]

        # Restore channel dimension
        H_long = enc.reshape(B, C, Np, self.d_model)

        return H_long




