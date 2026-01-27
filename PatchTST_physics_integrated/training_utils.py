import numpy as np
import torch


def validate(model, val_loader, criterion, device, target_indices):
    """
    Validation function for Physics-Integrated PatchTST.

    Supports:
      A) Old output:  outputs = model(batch_x) -> [B, pred_len, C]
      B) Fusion head: outputs = model(batch_x) -> [B, 1] or [B, pred_len] or [B, pred_len, 1]

    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run on
        target_indices: Indices of target channels (int or list/array)
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = []

    # Make target_indices always list-like
    if isinstance(target_indices, (int, np.integer)):
        target_indices_list = [int(target_indices)]
    else:
        target_indices_list = list(target_indices)

    # pred_len: try model.pred_len; else fall back to args if you store it; else infer later
    pred_len = getattr(model, "pred_len", None)

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            outputs = model(batch_x)

            # ---- TRUE targets always from batch_y last pred_len ----
            # batch_y is usually [B, label_len + pred_len, C]
            # If pred_len is not on model, infer from outputs when possible.
            if pred_len is None:
                # Infer pred_len
                if outputs.ndim == 3:
                    pred_len = outputs.shape[1]
                elif outputs.ndim == 2:
                    # If fusion output is [B,1] -> assume single-step
                    # If fusion output is [B,pred_len] -> pred_len = outputs.shape[1]
                    pred_len = outputs.shape[1]
                else:
                    raise ValueError(f"Cannot infer pred_len from outputs shape {outputs.shape}")

            y_true = batch_y[:, -pred_len:, target_indices_list]  # [B, pred_len, K]

            # ---- Align outputs to y_true ----
            if outputs.ndim == 3:
                # Old model: outputs [B, pred_len, C] -> select targets
                y_pred = outputs[:, -pred_len:, target_indices_list]  # [B, pred_len, K]
                loss = criterion(y_pred, y_true)

            elif outputs.ndim == 2:
                # Fusion head: outputs [B, out_dim]
                # This validate supports ONLY K=1 for fusion (single target)
                if y_true.shape[-1] != 1:
                    raise ValueError(
                        f"Fusion output is 2D {outputs.shape}, but y_true has {y_true.shape[-1]} targets. "
                        f"Fusion validate supports single target only (K=1)."
                    )

                y_true_1 = y_true.squeeze(-1)  # [B, pred_len]

                out_dim = outputs.shape[1]
                if out_dim == pred_len:
                    # Multi-horizon fusion: compare [B,pred_len] vs [B,pred_len]
                    loss = criterion(outputs, y_true_1)
                elif out_dim == 1:
                    # Single-step fusion: compare [B,1] vs last step [B,1]
                    loss = criterion(outputs, y_true_1[:, -1:].contiguous())
                else:
                    raise ValueError(
                        f"Fusion out_dim={out_dim} doesn't match pred_len={pred_len} and isn't 1. "
                        f"Use out_dim=1 or out_dim=pred_len."
                    )

            elif outputs.ndim == 3 and outputs.shape[-1] == 1:
                # Fusion head returns [B, pred_len, 1]
                if y_true.shape[-1] != 1:
                    raise ValueError("outputs last dim is 1 but y_true has multiple targets.")
                y_pred = outputs[:, -pred_len:, :]
                loss = criterion(y_pred, y_true)

            else:
                raise ValueError(f"Unsupported outputs shape: {outputs.shape}")

            total_loss.append(loss.item())

    model.train()
    return float(np.mean(total_loss)) if len(total_loss) > 0 else float("nan")
