import torch
import torch.nn as nn
import torch.nn.functional as F

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

