#!/bin/bash

# Inference-only script for Weather dataset using Original PatchTST
# - Uses checkpoint from $GIT_REPO_ROOT/output/Original/checkpoints/weather_336_96
# - Runs test/inference only (no training)
# - Outputs to $GIT_REPO_ROOT/output/Original/test_results/weather_336_96/

set -euo pipefail

# Set paths relative to GIT_REPO_ROOT
GIT_REPO_ROOT=$(git rev-parse --show-toplevel)

# Check if running in Google Colab (skip venv if so)
if [ -n "${COLAB_RELEASE_TAG:-}" ] || [ -d "/content" ]; then
    echo "Running in Google Colab, skipping venv activation"
else
    # Activate virtual environment (POSIX + Windows venv handling)
    if [ -f "$GIT_REPO_ROOT/.venv/bin/activate" ]; then
        echo "Activating virtualenv (bin/activate)"
        source "$GIT_REPO_ROOT/.venv/bin/activate"
    elif [ -f "$GIT_REPO_ROOT/.venv/Scripts/activate" ]; then
        echo "Activating virtualenv (Scripts/activate)"
        sed -i 's/\r$//' "$GIT_REPO_ROOT/.venv/Scripts/activate" 2>/dev/null || true
        source "$GIT_REPO_ROOT/.venv/Scripts/activate"
    else
        echo "No virtualenv activation found at $GIT_REPO_ROOT/.venv - creating one with python3..."
        python3 -m venv "$GIT_REPO_ROOT/.venv"
        source "$GIT_REPO_ROOT/.venv/bin/activate"
    fi
fi

# Original PatchTST experiment parameters (match training defaults)
seq_len=336
label_len=48
pred_len=96
model_name=PatchTST
root_path_name=$GIT_REPO_ROOT/datasets/weather/
data_path_name=weather.csv
model_id_name=weather
data_name=custom
random_seed=2021

echo "===== Original PatchTST inference (test only) ====="
echo "Project root: $GIT_REPO_ROOT"

# Checkpoint and output paths
CKPT_DIR="$GIT_REPO_ROOT/output/Original/checkpoints/weather_${seq_len}_${pred_len}"
OUTPUT_DIR="$GIT_REPO_ROOT/output/Original/test_results/weather_${seq_len}_${pred_len}"
model_id="${model_id_name}_${seq_len}_${pred_len}"

# The setting name matches training checkpoint naming
setting="${model_id}_PatchTST_custom_ftM_sl${seq_len}_ll${label_len}_pl${pred_len}_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0"

# Check checkpoint exists
if [ ! -f "$CKPT_DIR/checkpoint.pth" ]; then
    echo "Checkpoint not found at: $CKPT_DIR/checkpoint.pth" >&2
    echo "Please run training first or copy checkpoint to this location." >&2
    exit 1
fi

echo ""
echo "Using checkpoint: $CKPT_DIR/checkpoint.pth"
echo "Output directory: $OUTPUT_DIR"
echo "Model ID: $model_id"

# Locate run_longExp.py
run_script="$GIT_REPO_ROOT/PatchTST_supervised/run_longExp.py"
if [ ! -f "$run_script" ]; then
    echo "Could not find PatchTST_supervised/run_longExp.py in repository." >&2
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$GIT_REPO_ROOT/output/Original/logs"

# Run test-only inference with sliding window prediction
# 4 samples with stride=pred_len, giving 4 x 96 = 384 total predicted timesteps
# Each prediction uses 336 lookback to predict next 96 steps (non-overlapping)
# Sample 0: [0:336]   → [336:432]
# Sample 1: [96:432]  → [432:528]
# Sample 2: [192:528] → [528:624]
# Sample 3: [288:624] → [624:720]
echo ""
echo "Running Original PatchTST inference using test set..."
echo "Sliding window: stride=$pred_len, 4 samples x $pred_len steps = $((4 * pred_len)) total predicted timesteps"
(cd "$GIT_REPO_ROOT/PatchTST_supervised" && \
  python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 0 \
    --iterative \
    --num_iterations 1 \
    --max_samples 4 \
    --window_stride $pred_len \
    --root_path "$root_path_name" \
    --data_path "$data_path_name" \
    --model_id "$model_id" \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 21 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 16 \
    --stride 8 \
    --des 'Exp' \
    --checkpoint_path "$CKPT_DIR/checkpoint.pth" )

echo ""
echo "================================================================"
echo "Original PatchTST inference finished!"
echo "================================================================"

# Run plotting and metrics generation
PLOT_LOG="$GIT_REPO_ROOT/output/Original/logs/${model_name}_${model_id}_plot.log"
echo ""
echo "Running plotting and metrics generation (log: $PLOT_LOG)..."
if ! (cd "$GIT_REPO_ROOT" && PYTHONPATH="$GIT_REPO_ROOT:$GIT_REPO_ROOT/PatchTST_supervised:${PYTHONPATH:-}" python -u "$GIT_REPO_ROOT/main/plot_original.py" \
    --model_id_name "$model_id_name" \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --results_src "$GIT_REPO_ROOT/PatchTST_supervised/results" \
    --output_dir "$OUTPUT_DIR" \
    --data_file "$data_path_name" 2>&1 | tee -a "$PLOT_LOG"); then
    echo "Plotting failed. See $PLOT_LOG" >&2
else
    echo "Plotting finished successfully. See $PLOT_LOG for details."
fi

echo ""
echo "================================================================"
echo "All outputs saved to: $OUTPUT_DIR"
echo "  - per_feature_metrics.csv"
echo "  - test_data_statistics.csv"
echo "  - prediction_grid_sl${seq_len}_pl${pred_len}.png"
echo "  - summary.txt"
echo "================================================================"

exit 0
