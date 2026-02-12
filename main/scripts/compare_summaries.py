"""Compare summaries from MSVLPatchTST and Original test results and write a Markdown report

Scans:
- output/MSVLPatchTST/test_results/*/summary*.txt
- output/Original/test_results/*/summary*.txt

Extracts Overall MSE, MAE and Huber Loss and writes `output/summary_comparison.md`.

Usage:
    python compare_summaries.py [--msvl-dir PATH] [--orig-dir PATH] [--output-dir PATH]
"""
import re
import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Repository root (two levels up from this script)
ROOT = Path(__file__).resolve().parents[2]

def convert_windows_path(path_str):
    """Convert Windows path to WSL path if running in WSL."""
    if path_str is None:
        return None
    # Check if it looks like a Windows path (e.g., C:\Users\...)
    win_match = re.match(r'^([A-Za-z]):[\\\/](.*)$', path_str)
    if win_match:
        drive = win_match.group(1).lower()
        rest = win_match.group(2).replace('\\', '/')
        return f'/mnt/{drive}/{rest}'
    return path_str

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Compare MSVLPatchTST and Original test results')
parser.add_argument('--msvl-dir', type=str, default=None,
                    help='Path to MSVL test_results directory (default: output/MSVLPatchTST/test_results)')
parser.add_argument('--orig-dir', type=str, default=None,
                    help='Path to Original test_results directory (default: output/Original/test_results)')
parser.add_argument('--output-dir', type=str, default=None,
                    help='Path to output directory for summary and plots (default: output)')
args = parser.parse_args()

# Convert Windows paths to WSL paths if needed
msvl_dir_path = convert_windows_path(args.msvl_dir)
orig_dir_path = convert_windows_path(args.orig_dir)
output_dir_path = convert_windows_path(args.output_dir)

# Set directories based on args or defaults
# Note: MSVL_TEST_DIR and ORIG_TEST_DIR default to repository's output folder, not the custom output dir
OUTPUT_DIR = Path(output_dir_path) if output_dir_path else ROOT / "output"
MSVL_TEST_DIR = Path(msvl_dir_path) if msvl_dir_path else ROOT / "output" / "MSVLPatchTST" / "test_results"
ORIG_TEST_DIR = Path(orig_dir_path) if orig_dir_path else ROOT / "output" / "Original" / "test_results"

# Add MSVLPatchTST to path to import config
sys.path.insert(0, str(ROOT / "MSVLPatchTST"))
from config import MSVLConfig

# Get channel groups from config
config = MSVLConfig()
LONG_CHANNEL_FEATURES = config.channel_groups['long_channel']['target_names']  # ['p (mbar)', 'T (degC)', 'wv (m/s)']
SHORT_CHANNEL_FEATURES = config.channel_groups['short_channel']['target_names']  # ['rain (mm)', 'max. wv (m/s)', 'raining (s)']

PATTERNS = [
    MSVL_TEST_DIR.glob("*/summary*.txt"),
    ORIG_TEST_DIR.glob("*/summary*.txt"),
]

metrics = {}

line_re = re.compile(r"^(MSE|MAE|Huber Loss):\s*([0-9.eE+-]+)")

for group in PATTERNS:
    for path in group:
        name = path.parent.name
        text = path.read_text()
        mse = mae = huber = None
        for m in line_re.finditer(text):
            key, val = m.group(1), m.group(2)
            if key == "MSE":
                mse = float(val)
            elif key == "MAE":
                mae = float(val)
            elif key == "Huber Loss":
                huber = float(val)
        if mse is None and mae is None and huber is None:
            # try alternate lines that may have slightly different labels
            mse_m = re.search(r"^MSE:\s*([0-9.eE+-]+)", text, re.MULTILINE)
            mae_m = re.search(r"^MAE:\s*([0-9.eE+-]+)", text, re.MULTILINE)
            hub_m = re.search(r"^Huber Loss:\s*([0-9.eE+-]+)", text, re.MULTILINE)
            if mse_m:
                mse = float(mse_m.group(1))
            if mae_m:
                mae = float(mae_m.group(1))
            if hub_m:
                huber = float(hub_m.group(1))
        metrics[name] = {"MSE": mse, "MAE": mae, "Huber": huber}

# Find original name (if any) and order results with Original first
ordered = []
orig_dir = ORIG_TEST_DIR
if orig_dir.exists():
    for p in orig_dir.iterdir():
        if p.is_dir():
            if p.name in metrics:
                ordered.append((p.name, metrics.pop(p.name)))

# then the remaining MSVL groups (sorted)
if MSVL_TEST_DIR.exists():
    for p in sorted(MSVL_TEST_DIR.iterdir()):
        if p.is_dir() and p.name in metrics:
            ordered.append((p.name, metrics.pop(p.name)))

# any leftover
for name in sorted(metrics.keys()):
    ordered.append((name, metrics[name]))

# Helpers

def safe_fname(s):
    return ''.join(c if c.isalnum() or c in '._-' else '_' for c in s)

# Write overall markdown
out = OUTPUT_DIR / "summary_comparison.md"
plots_dir = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

with out.open("w") as f:
    f.write("# Summary comparison: MSVLPatchTST vs Original\n\n")
    f.write("Comparison of Overall metrics extracted from experiment summary files. Only MSE, MAE and Huber Loss are shown.\n\n")
    f.write("| Experiment | MSE | MAE | Huber Loss |\n")
    f.write("|---|---:|---:|---:|\n")
    for name, vals in ordered:
        mse = f"{vals['MSE']:.6f}" if vals['MSE'] is not None else "N/A"
        mae = f"{vals['MAE']:.6f}" if vals['MAE'] is not None else "N/A"
        hub = f"{vals['Huber']:.6f}" if vals['Huber'] is not None else "N/A"
        f.write(f"| {name} | {mse} | {mae} | {hub} |\n")

    f.write("\n---\n\n")
    f.write("## Per-feature comparisons and plots\n\n")

    # Load original per-feature mapping (pick first original experiment)
    orig_dir = ORIG_TEST_DIR
    orig_mapping = {}  # feature_name -> index in original numpy arrays
    orig_metrics = {}  # feature_name -> metrics dict
    orig_pred = orig_true = None
    if orig_dir.exists():
        # pick first directory
        d = next(orig_dir.iterdir())
        per_feat = d / "per_feature_metrics.csv"
        if per_feat.exists():
            with per_feat.open() as pf:
                rdr = csv.DictReader(pf)
                for row in rdr:
                    name_feat = row['feature']
                    orig_mapping[name_feat] = int(row['index'])
                    orig_metrics[name_feat] = { 'mse': float(row['mse']), 'mae': float(row['mae']), 'huber': float(row['huber']) }
        p_pred = d / 'pred.npy'
        p_true = d / 'true.npy'
        if p_pred.exists():
            orig_pred = np.load(p_pred, allow_pickle=True)
        if p_true.exists():
            orig_true = np.load(p_true, allow_pickle=True)

    # For each MSVL experiment, create plots comparing to original
    if MSVL_TEST_DIR.exists():
        for mdir in sorted(MSVL_TEST_DIR.iterdir()):
            if not mdir.is_dir():
                continue
            # Parse parameters from the directory name for header
            seq_pred = re.search(r'_(\d+)_(\d+)', mdir.name)
            seq_pred_str = ''
            if seq_pred:
                seq_pred_str = f"Pred-Len ({seq_pred.group(1)}, {seq_pred.group(2)})"
            sp = re.search(r'sp(\d+)', mdir.name)
            ss = re.search(r'ss(\d+)', mdir.name)
            lp = re.search(r'lp(\d+)', mdir.name)
            ls = re.search(r'ls(\d+)', mdir.name)
            ch1 = f"Channel-1 ({lp.group(1)}, {ls.group(1)})" if lp and ls else ''
            ch2 = f"Channel-2 ({sp.group(1)}, {ss.group(1)})" if sp and ss else ''
            param_parts = [p for p in [seq_pred_str, ch1, ch2] if p]
            # join with commas to keep header on a single line (e.g. "Pred-Len (...), Channel-1 (...), Channel-2 (...)")
            param_str = ', '.join(param_parts)
            header = param_str if param_str else mdir.name
            f.write(f"### {header}\n\n")
            # read MSVL per-feature metrics
            m_metrics = {}  # feature_name -> metrics dict
            m_map = {}  # feature_name -> index in MSVL numpy arrays
            per_feat_m = mdir / next((p.name for p in mdir.iterdir() if p.name.startswith('per_feature_metrics')), 'per_feature_metrics.csv')
            if per_feat_m.exists():
                with per_feat_m.open() as pf:
                    rdr = csv.DictReader(pf)
                    for row in rdr:
                        name_feat = row['feature']
                        m_map[name_feat] = int(row['index'])
                        m_metrics[name_feat] = { 'mse': float(row['mse']), 'mae': float(row['mae']), 'huber': float(row['huber']) }

            # load predictions and truths
            # find pred file that starts with pred
            p_pred = next((p for p in mdir.iterdir() if p.name.startswith('pred') and p.suffix == '.npy'), None)
            p_true = next((p for p in mdir.iterdir() if p.name.startswith('true') and p.suffix == '.npy'), None)
            if p_pred is None or p_true is None or orig_pred is None or orig_true is None:
                f.write("Missing prediction/true files for plotting.\n\n")
                continue
            m_pred = np.load(p_pred, allow_pickle=True)
            m_true = np.load(p_true, allow_pickle=True)

            # reshape to (total_timesteps, n_features)
            def flatten(arr):
                if arr.ndim == 3:
                    return arr.reshape(-1, arr.shape[-1])
                elif arr.ndim == 2:
                    return arr
                else:
                    return arr

            orig_pred_flat = flatten(orig_pred)
            orig_true_flat = flatten(orig_true)
            m_pred_flat = flatten(m_pred)
            m_true_flat = flatten(m_true)

            # Write table header
            f.write("| feature | channel | original MSE | msvl MSE | original MAE | msvl MAE | original Huber | msvl Huber |\n")
            f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")

            # collect plot paths for Channel-1 and Channel-2 separately
            ch1_plot_paths = []
            ch2_plot_paths = []

            # Order features: Channel-1 (long_channel) first, then Channel-2 (short_channel)
            ordered_features = []
            for feat in LONG_CHANNEL_FEATURES:
                if feat in m_map and feat in orig_mapping:
                    ordered_features.append((feat, 'Channel-1'))
            for feat in SHORT_CHANNEL_FEATURES:
                if feat in m_map and feat in orig_mapping:
                    ordered_features.append((feat, 'Channel-2'))

            for feat, channel in ordered_features:
                # Get indices from CSV - these are exact array column positions
                col_orig = orig_mapping[feat]  # Index in original numpy arrays
                col_m = m_map[feat]  # Index in MSVL numpy arrays
                
                # ensure columns exist
                if col_orig >= orig_pred_flat.shape[1] or col_m >= m_pred_flat.shape[1]:
                    continue
                
                # Get data using indices from CSV
                t = np.arange(orig_pred_flat.shape[0])
                y_true = orig_true_flat[:, col_orig]
                y_orig = orig_pred_flat[:, col_orig]
                y_m = m_pred_flat[:, col_m]

                # Prepare metric text (use per-feature metrics from CSV)
                om = orig_metrics.get(feat, {'mse': None, 'mae': None, 'huber': None})
                mm = m_metrics.get(feat, {'mse': None, 'mae': None, 'huber': None})

                # Parse parameters from the directory name for titles
                seq_pred = re.search(r'_(\d+)_(\d+)', mdir.name)
                pred_a = pred_b = None
                if seq_pred:
                    pred_a, pred_b = seq_pred.group(1), seq_pred.group(2)
                sp = re.search(r'sp(\d+)', mdir.name)
                ss = re.search(r'ss(\d+)', mdir.name)
                lp = re.search(r'lp(\d+)', mdir.name)
                ls = re.search(r'ls(\d+)', mdir.name)

                # Build a concise feature label (e.g. 'p-mbar' from 'p (mbar)')
                base = feat.split('(')[0].strip()
                unit_match = re.search(r'\(([^)]+)\)', feat)
                unit = re.sub(r'[^A-Za-z0-9]', '', unit_match.group(1)) if unit_match else ''
                base_clean = re.sub(r'[^A-Za-z0-9_.-]', '-', base)
                feature_label = f"{base_clean}-{unit}" if unit else base_clean

                # Channel text based on actual channel assignment from config
                if channel == 'Channel-1':
                    ch_txt = f"Channel-1 ({lp.group(1)}, {ls.group(1)})" if lp and ls else 'Channel-1'
                else:
                    ch_txt = f"Channel-2 ({sp.group(1)}, {ss.group(1)})" if sp and ss else 'Channel-2'

                # Prepare metric lines for title
                om_m = om['mse'] if om['mse'] is not None else None
                mm_m = mm['mse'] if mm['mse'] is not None else None
                om_line = f"Original MSE={om['mse']:.6f}, MAE={om['mae']:.6f}, Huber={om['huber']:.6f}" if om_m is not None else "Original: N/A"
                mm_line = f"MSVL MSE={mm['mse']:.6f}, MAE={mm['mae']:.6f}, Huber={mm['huber']:.6f}" if mm_m is not None else "MSVL: N/A"

                # Build title
                top_parts = [feature_label]
                if pred_a and pred_b:
                    top_parts.append(f"Pred-Len ({pred_a}, {pred_b})")
                if ch_txt:
                    top_parts.append(ch_txt)
                top_line = ', '.join(top_parts)

                # Assemble title (three lines)
                title_lines = [top_line, om_line, mm_line]
                title_txt = "\n".join(title_lines)

                plt.figure(figsize=(10,3))
                # Colors: dark green for True, dark blue for Original, dark red for MSVL
                plt.plot(t, y_true, color='#006400', label='True', linewidth=2)
                plt.plot(t, y_orig, color='#0D47A1', label='Original pred', linewidth=1.5, linestyle='--')
                plt.plot(t, y_m, color='#8B0000', label='MSVL pred', linewidth=1.5)
                plt.title(title_txt)
                plt.xlabel('time index')
                plt.legend()
                plt.tight_layout()

                plot_name = plots_dir / f"{safe_fname(mdir.name)}__{safe_fname(feat)}.png"
                plt.savefig(plot_name)
                plt.close()

                # save relative path for the image grid
                plot_rel_path = str(plot_name.relative_to(OUTPUT_DIR).as_posix())
                if channel == 'Channel-1':
                    ch1_plot_paths.append(plot_rel_path)
                else:
                    ch2_plot_paths.append(plot_rel_path)

                # Write table row
                om_mse = f"{om['mse']:.6f}" if om['mse'] is not None else 'N/A'
                mm_mse = f"{mm['mse']:.6f}" if mm['mse'] is not None else 'N/A'
                om_mae = f"{om['mae']:.6f}" if om['mae'] is not None else 'N/A'
                mm_mae = f"{mm['mae']:.6f}" if mm['mae'] is not None else 'N/A'
                om_h = f"{om['huber']:.6f}" if om['huber'] is not None else 'N/A'
                mm_h = f"{mm['huber']:.6f}" if mm['huber'] is not None else 'N/A'

                f.write(f"| {feat} | {channel} | {om_mse} | {mm_mse} | {om_mae} | {mm_mae} | {om_h} | {mm_h} |\n")

            # Render plots as a 3x2 matrix: Channel-1 row 1, Channel-2 row 2
            f.write("\n**Plots**\n\n")
            f.write("| Channel-1: Long Channel (p, T, wv) | | |\n")
            f.write("|---|---|---|\n")
            # Row 1: Channel-1 plots
            f.write("|")
            for i in range(3):
                if i < len(ch1_plot_paths):
                    f.write(f" <img src=\"{ch1_plot_paths[i]}\" width=\"320\"> |")
                else:
                    f.write("  |")
            f.write("\n\n")
            
            f.write("| Channel-2: Short Channel (rain, max.wv, raining) | | |\n")
            f.write("|---|---|---|\n")
            # Row 2: Channel-2 plots
            f.write("|")
            for i in range(3):
                if i < len(ch2_plot_paths):
                    f.write(f" <img src=\"{ch2_plot_paths[i]}\" width=\"320\"> |")
                else:
                    f.write("  |")
            f.write("\n")

            f.write('\n')

print(f"Wrote comparison and plots to {out} and {plots_dir}")
