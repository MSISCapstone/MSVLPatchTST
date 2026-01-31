#!/usr/bin/env python3
"""
Plotting helper for the original Weather experiments.

- Creates a 2x3 figure with the following layout:
  Row 1: 'p (mbar)', 'T (degC)', 'wv (m/s)'
  Row 2: 'max. wv (m/s)', 'rain (mm)', 'raining (s)'

- Reads predictions from a results folder (expects `pred.npy` saved by experiments)
- Reconstructs ground-truth test values by loading the dataset (uses same split logic as Dataset_Custom)
- Computes per-feature metrics (MAE, MSE, RMSE, RSE)
- Saves the figure and a CSV summary into `GIT_REPO_ROOT/outputs/results/{model_id_name}_{seq_len}_{pred_len}` (created if missing)

Usage:
  python main/plot_original.py \
    --model_id_name weather --seq_len 96 --pred_len 96 \
    [--results_src PATH] [--data_root PATH] [--data_file weather.csv]

If --results_src is provided, the script will look for `pred.npy` there. Otherwise it will try:
  GIT_REPO_ROOT/results/<setting>/pred.npy
then
  GIT_REPO_ROOT/output/results/<model_id_name>_<seq_len>_<pred_len>/pred.npy

"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import utilities from project
from types import SimpleNamespace

from PatchTST_supervised.data_provider.data_factory import data_provider
from PatchTST_supervised.utils.metrics import MAE, MSE, RMSE, RSE

FEATURES_TO_PLOT = [
    'p (mbar)',
    'T (degC)',
    'wv (m/s)',
    'max. wv (m/s)',
    'rain (mm)',
    'raining (s)'
]

PLOT_ORDER = FEATURES_TO_PLOT  # ordered as requested


def find_pred_file(results_src, git_root, model_id_name, seq_len, pred_len):
    # user-supplied
    if results_src:
        p = Path(results_src)
        if p.is_dir() and (p / 'pred.npy').exists():
            return p / 'pred.npy', p
        elif p.exists() and p.name.endswith('.npy'):
            return p, p.parent
        else:
            raise FileNotFoundError(f"No pred.npy found under provided results_src: {results_src}")

    # common locations: results/<setting>/pred.npy (as experiments do)
    # fallback to output/results/<model_id_name>_<seq_len>_<pred_len>/pred.npy
    # check output/results path first
    fallback = Path(git_root) / 'output' / 'results' / f"{model_id_name}_{seq_len}_{pred_len}"
    if (fallback / 'pred.npy').exists():
        return fallback / 'pred.npy', fallback

    # try results/<setting> (we don't know the exact setting name, pick the one that contains model_id_name and seq/pred)
    results_root = Path(git_root) / 'results'
    if results_root.exists():
        # pick candidate folders
        candidates = [p for p in results_root.iterdir() if p.is_dir() and model_id_name in p.name and f"sl{seq_len}" in p.name or f"pl{pred_len}" in p.name]
        # fallback more lenient search
        if not candidates:
            candidates = [p for p in results_root.iterdir() if p.is_dir() and model_id_name in p.name]
        # pick most recent
        if candidates:
            candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
            p = candidates[-1]
            if (p / 'pred.npy').exists():
                return p / 'pred.npy', p

    raise FileNotFoundError("Could not locate pred.npy in common locations. Please point --results_src to the folder that contains pred.npy")


def build_test_truths(git_root, root_path_name, data_path_name, seq_len, label_len, pred_len, batch_size=128, num_workers=0):
    # Build a fake args object for data_provider
    args = SimpleNamespace()
    args.data = 'custom'
    args.root_path = root_path_name
    args.data_path = data_path_name
    args.seq_len = seq_len
    args.label_len = label_len
    args.pred_len = pred_len
    args.features = 'M'
    args.target = 'OT'
    args.embed = 'timeF'
    args.freq = 'h'
    args.batch_size = batch_size
    args.num_workers = num_workers

    dataset, loader = data_provider(args, flag='test')

    trues = []
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
        # batched true values shaped (B, pred_len, D)
        # extract last pred_len dims and append
        f_dim = -1 if args.features == 'MS' else 0
        batch_y = batch_y[:, -pred_len:, f_dim:].numpy()
        trues.append(batch_y)
    if not trues:
        raise RuntimeError('No test batches found when reconstructing ground-truths')
    trues = np.concatenate(trues, axis=0)
    return trues, dataset


def compute_feature_indices(git_root, root_path_name, data_path_name, target='OT'):
    path = Path(root_path_name) / data_path_name
    df = pd.read_csv(path)
    cols = list(df.columns)
    if 'date' in cols:
        cols.remove('date')
    if target in cols:
        cols.remove(target)
    data_columns = cols + [target]
    return data_columns


def save_stats_and_plot(preds, trues, data_columns, out_dir, seq_len, pred_len):
    # preds and trues shapes: (N, pred_len, D)
    os.makedirs(out_dir, exist_ok=True)

    D = preds.shape[-1]

    # compute per-feature metrics and save
    metrics = []
    for idx, name in enumerate(data_columns):
        if idx >= D:
            break
        p = preds[..., idx]
        t = trues[..., idx]
        mae = MAE(p, t)
        mse = MSE(p, t)
        rmse = RMSE(p, t)
        rse = RSE(p, t)
        metrics.append({'feature': name, 'index': idx, 'mae': mae, 'mse': mse, 'rmse': rmse, 'rse': rse})
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(Path(out_dir) / 'per_feature_metrics.csv', index=False)

    # plot requested features in 2x3 grid
    plot_features = PLOT_ORDER
    # map to indices
    feature_indices = []
    for f in plot_features:
        if f in data_columns:
            feature_indices.append(data_columns.index(f))
        else:
            feature_indices.append(None)

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), constrained_layout=True)
    axes = axes.flatten()

    for ax, feat_name, feat_idx in zip(axes, plot_features, feature_indices):
        if feat_idx is None or feat_idx >= preds.shape[-1]:
            ax.text(0.5, 0.5, f'Feature not found: {feat_name}', ha='center', va='center')
            ax.set_title(feat_name)
            continue
        p = preds[..., feat_idx]  # (N, pred_len)
        t = trues[..., feat_idx]
        # mean across samples
        mean_p = np.nanmean(p, axis=0)
        mean_t = np.nanmean(t, axis=0)
        # compute metrics for display
        mae = MAE(p, t)
        mse = MSE(p, t)

        x = np.arange(pred_len)
        ax.plot(x, mean_t, label='Mean True', linewidth=2)
        ax.plot(x, mean_p, label='Mean Pred', linewidth=2)
        ax.set_title(f"{feat_name} (MAE={mae:.4f}, MSE={mse:.4f})")
        ax.legend()
        ax.set_xlabel('Horizon')

    fig_file = Path(out_dir) / f'prediction_grid_sl{seq_len}_pl{pred_len}.png'
    fig.savefig(fig_file, bbox_inches='tight')
    plt.close(fig)

    # Save a small summary text file
    with open(Path(out_dir) / 'summary.txt', 'w') as fh:
        fh.write('Per-feature metrics\n')
        fh.write(metrics_df.to_string(index=False))
        fh.write('\n')

    print(f"Saved plots and metrics to {out_dir}")


def save_model_architecture(git_root, found_dir, out_dir, seq_len, enc_in=21):
    """Attempt to load the checkpoint for the setting (found_dir.name) and save model architecture info.
    Saves:
      - model_architecture.txt (str(model))
      - model_architecture_summary.txt (torchinfo.summary output) if available
      - model_architecture_graph.png (torchviz) if available
    """
    import torch
    from pathlib import Path
    out_dir = Path(out_dir)
    ckpt_candidates = [
        Path(git_root) / 'output' / 'checkpoints' / found_dir.name / 'checkpoint.pth',
        Path(git_root) / 'checkpoints' / found_dir.name / 'checkpoint.pth',
        Path(git_root) / 'output' / 'checkpoints' / found_dir.name / 'checkpoint.pth',
        Path(git_root) / 'checkpoints' / found_dir.name / 'checkpoint.pth'
    ]
    ckpt_path = None
    for c in ckpt_candidates:
        if c.exists():
            ckpt_path = c
            break
    if ckpt_path is None:
        print(f"No checkpoint found for setting {found_dir.name} in expected locations; skipping model architecture save")
        return

    print(f"Found checkpoint at {ckpt_path} - attempting to load model for architecture dump")

    # Try to import model builder used by experiments
    try:
        from PatchTST_supervised.models import PatchTST as PatchTSTModelClass
    except Exception as e:
        # If direct import fails, try to import a Model name from a fallback location
        PatchTSTModelClass = None

    if PatchTSTModelClass is None:
        try:
            # fallback: try to import a module named PatchTST (some codebases provide .Model)
            import importlib
            m = importlib.import_module('PatchTST_supervised')
            PatchTSTModelClass = getattr(m, 'PatchTST', None)
        except Exception:
            PatchTSTModelClass = None

    def dump_checkpoint_param_summary(ckpt_path, out_dir):
        """Dump parameter names, shapes and counts from the checkpoint to text files."""
        import torch
        from pathlib import Path
        out_dir = Path(out_dir)
        try:
            state = torch.load(ckpt_path, map_location='cpu')
        except Exception as e:
            with open(out_dir / 'model_architecture.txt', 'w') as fh:
                fh.write(f"Failed to load checkpoint {ckpt_path}: {e}\n")
            return

        # Find a state dict inside the checkpoint if wrapped
        sd = None
        if isinstance(state, dict):
            for key in ('state_dict', 'model_state_dict', 'model', 'state'):
                if key in state and isinstance(state[key], dict):
                    sd = state[key]
                    break
            if sd is None and all(isinstance(v, (torch.Tensor,)) for v in state.values()):
                sd = state
        if sd is None:
            # try attribute-style
            if hasattr(state, 'state_dict'):
                sd = state.state_dict()
        if sd is None:
            with open(out_dir / 'model_architecture.txt', 'w') as fh:
                fh.write(f"Checkpoint loaded but no parameter dict found in {ckpt_path}\n")
                fh.write(str(type(state)) + "\n")
            return

        total = 0
        per_prefix = {}
        lines = []
        for name, val in sd.items():
            try:
                if isinstance(val, torch.Tensor):
                    shape = tuple(val.size())
                    nelems = val.numel()
                else:
                    # some checkpoints might store numpy arrays or other types
                    try:
                        arr = np.array(val)
                        shape = arr.shape
                        nelems = arr.size
                    except Exception:
                        shape = str(type(val))
                        nelems = 0
                total += nelems
                prefix = name.split('.')[0] if '.' in name else name
                per_prefix[prefix] = per_prefix.get(prefix, 0) + nelems
                lines.append(f"{name}\t{shape}\t{nelems}")
            except Exception as e:
                lines.append(f"{name}\t<error: {e}>")

        with open(out_dir / 'model_architecture.txt', 'w') as fh:
            fh.write(f"Checkpoint: {ckpt_path}\n")
            fh.write(f"Total params (approx): {total}\n\n")
            fh.write("Per-prefix parameter counts:\n")
            for k, v in sorted(per_prefix.items(), key=lambda x: -x[1]):
                fh.write(f"{k}: {v}\n")
            fh.write("\nFirst 100 parameter entries:\n")
            fh.write("name\tshape\tcount\n")
            fh.write("\n".join(lines[:100]))

        # write full parameter list separately
        with open(out_dir / 'model_parameters_list.txt', 'w') as fh:
            fh.write("name\tshape\tcount\n")
            fh.write("\n".join(lines))

        print(f"Wrote checkpoint parameter summary to {out_dir / 'model_architecture.txt'} and full list to {out_dir / 'model_parameters_list.txt'}")

    if PatchTSTModelClass is None:
        # Try dumping checkpoint parameters if we cannot import/instantiate model class
        print("Could not import PatchTST model class; extracting parameter summary from checkpoint")
        dump_checkpoint_param_summary(ckpt_path, out_dir)
        return

    # Build a minimal args namespace compatible with model construction
    from types import SimpleNamespace
    args = SimpleNamespace()
    args.enc_in = enc_in
    args.seq_len = seq_len
    args.label_len = 48
    args.pred_len = 96
    # set reasonable defaults often used in experiments
    args.patch_len = 16
    args.stride = 8
    args.d_model = 128
    args.n_heads = 16
    args.e_layers = 3
    args.d_layers = 1
    args.d_ff = 256
    args.fc_dropout = 0.2
    args.head_dropout = 0

    try:
        # exp_main used: model = PatchTST.Model(args)
        model = PatchTSTModelClass.Model(args)
        state = torch.load(ckpt_path, map_location='cpu')
        try:
            model.load_state_dict(state)
        except Exception:
            # sometimes checkpoint is nested under 'model' or other keys
            if isinstance(state, dict) and 'model' in state:
                model.load_state_dict(state['model'])
        # Save text representation
        with open(out_dir / 'model_architecture.txt', 'w') as fh:
            fh.write(str(model))

        # Try torchinfo summary
        try:
            from torchinfo import summary
            s = summary(model, input_size=(1, args.enc_in, args.seq_len), verbose=0)
            with open(out_dir / 'model_architecture_summary.txt', 'w') as fh:
                fh.write(str(s))
        except Exception as e:
            print('torchinfo not available or failed:', e)

        # Try torchviz graph
        try:
            from torchviz import make_dot
            import torch
            dummy = torch.zeros(1, args.enc_in, args.seq_len)
            model.eval()
            with torch.no_grad():
                y = model(dummy)
            dot = make_dot(y, params=dict(model.named_parameters()))
            dot.format = 'png'
            out_png = out_dir / 'model_architecture_graph'
            dot.render(str(out_png), cleanup=True)
        except Exception as e:
            print('torchviz graph generation failed or not available:', e)

        print(f"Saved model architecture files into {out_dir}")
    except Exception as e:
        print('Failed to instantiate or load model for architecture dump:', e)
        # fallback to checkpoint parsing
        dump_checkpoint_param_summary(ckpt_path, out_dir)




def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_id_name', default='weather')
    p.add_argument('--seq_len', type=int, default=96)
    p.add_argument('--label_len', type=int, default=48)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--results_src', default=None, help='Path to folder containing pred.npy (optional)')
    p.add_argument('--data_root', default=None, help='Path to datasets root (defaults to GIT_REPO_ROOT/datasets)')
    p.add_argument('--data_file', default='weather.csv')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=0)

    args = p.parse_args()

    # determine git root
    git_root = os.popen('git rev-parse --show-toplevel').read().strip()
    if not git_root:
        git_root = os.getcwd()

    root_path_name = args.data_root if args.data_root else os.path.join(git_root, 'datasets', 'weather')

    pred_file, found_dir = find_pred_file(args.results_src, git_root, args.model_id_name, args.seq_len, args.pred_len)

    preds = np.load(pred_file)
    # preds shape should be (N, pred_len, D)

    trues, dataset = build_test_truths(git_root, root_path_name, args.data_file, args.seq_len, args.label_len, args.pred_len, batch_size=args.batch_size, num_workers=args.num_workers)

    data_columns = compute_feature_indices(git_root, root_path_name, args.data_file)

    # Create outputs dir per user's request under /outputs/results/{model_id_name}_{seq_len}_{pred_len}
    out_dir = Path(git_root) / 'output' / 'test_results' / f"{args.model_id_name}_{args.seq_len}_{args.pred_len}"
    save_stats_and_plot(preds, trues, data_columns, out_dir, args.seq_len, args.pred_len)

    # Save model architecture files (text, optional torchinfo/torchviz)
    try:
        save_model_architecture(git_root, found_dir, out_dir, args.seq_len, enc_in=21)
    except Exception as e:
        print('Failed to save model architecture:', e)


if __name__ == '__main__':
    main()
