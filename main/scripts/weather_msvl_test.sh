#!/bin/bash

# Inference-only script for Weather dataset using MSVLPatchTST
# - Uses checkpoint from $GIT_REPO_ROOT/output/MSVLPatchTST/checkpoints/weather_336_96_sp{}_ss{}_lp{}_ls{}
# - Runs test/inference only (no training)
# - Outputs to $GIT_REPO_ROOT/output/MSVLPatchTST/test_results/weather_336_96_sp{}_ss{}_lp{}_ls{}/

set -euo pipefail

# Parse command line arguments
# Usage option 1: Pass config file
#   ./weather_msvl_test.sh patch_len_configs/ps_config_1.txt
# Usage option 2: Pass individual parameters
#   ./weather_msvl_test.sh 8 4 16 8

if [ -f "$1" ]; then
    # Config file provided
    CONFIG_FILE="$1"
    PATCH_STRIDE_PARAMS=$(cat "$CONFIG_FILE")
    echo "Using config file: $CONFIG_FILE"
    echo "Parameters: $PATCH_STRIDE_PARAMS"
    
    # Extract values for folder naming
    patch_len_short=$(echo "$PATCH_STRIDE_PARAMS" | grep -oP '(?<=--patch_len_short )\d+')
    stride_short=$(echo "$PATCH_STRIDE_PARAMS" | grep -oP '(?<=--stride_short )\d+')
    patch_len_long=$(echo "$PATCH_STRIDE_PARAMS" | grep -oP '(?<=--patch_len_long )\d+')
    stride_long=$(echo "$PATCH_STRIDE_PARAMS" | grep -oP '(?<=--stride_long )\d+')
else
    # Individual parameters provided
    patch_len_short=${1:-16}
    stride_short=${2:-8}
    patch_len_long=${3:-16}
    stride_long=${4:-8}
    PATCH_STRIDE_PARAMS="--patch_len_short $patch_len_short --stride_short $stride_short --patch_len_long $patch_len_long --stride_long $stride_long"
    echo "Using individual parameters:"
fi

echo "Short channel: patch_len=$patch_len_short, stride=$stride_short"
echo "Long channel: patch_len=$patch_len_long, stride=$stride_long"

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

# MSVLPatchTST experiment parameters (match training defaults)
seq_len=336
label_len=48
pred_len=96
model_name=MSVLPatchTST
root_path_name=$GIT_REPO_ROOT/datasets/weather/
data_path_name=weather_with_hour.csv
model_id_name=weather
data_name=custom
random_seed=2021

echo "===== MSVLPatchTST inference (test only) ====="
echo "Project root: $GIT_REPO_ROOT"
echo "Short channel: patch_len=$patch_len_short, stride=$stride_short"
echo "Long channel: patch_len=$patch_len_long, stride=$stride_long"

# Checkpoint and output paths
CKPT_DIR="$GIT_REPO_ROOT/output/MSVLPatchTST/checkpoints/weather_${seq_len}_${pred_len}_sp${patch_len_short}_ss${stride_short}_lp${patch_len_long}_ls${stride_long}"
OUTPUT_DIR="$GIT_REPO_ROOT/output/MSVLPatchTST/test_results/weather_${seq_len}_${pred_len}_sp${patch_len_short}_ss${stride_short}_lp${patch_len_long}_ls${stride_long}"
model_id="${model_id_name}_${seq_len}_${pred_len}_sp${patch_len_short}_ss${stride_short}_lp${patch_len_long}_ls${stride_long}"

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

# Locate MSVLPatchTST run_longExp.py
run_script="$GIT_REPO_ROOT/MSVLPatchTST/run_longExp.py"
if [ ! -f "$run_script" ]; then
    echo "Could not find MSVLPatchTST/run_longExp.py in repository." >&2
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$GIT_REPO_ROOT/output/MSVLPatchTST/logs"

# Run test-only inference with sliding window prediction
# 4 samples with stride=pred_len, giving 4 x 96 = 384 total predicted timesteps
# Each prediction uses 336 lookback to predict next 96 steps (non-overlapping)
# Sample 0: [0:336]   → [336:432]
# Sample 1: [96:432]  → [432:528]
# Sample 2: [192:528] → [528:624]
# Sample 3: [288:624] → [624:720]
echo ""
echo "Running MSVLPatchTST inference using test set..."
echo "Sliding window: stride=$pred_len, 4 samples x $pred_len steps = $((4 * pred_len)) total predicted timesteps"
echo "Short channel: patch_len=$patch_len_short, stride=$stride_short"
echo "Long channel: patch_len=$patch_len_long, stride=$stride_long"
python -u "$run_script" \
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
    --enc_in 22 \
    --c_out 6 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    $PATCH_STRIDE_PARAMS \
    --des 'Exp' \
    --train_epochs 100 \
    --patience 20 \
    --itr 1 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --checkpoint_path "$CKPT_DIR/checkpoint.pth"

echo ""
echo "================================================================"
echo "MSVLPatchTST inference finished!"
echo "================================================================"

# Run plotting and metrics generation
PLOT_LOG="$GIT_REPO_ROOT/output/MSVLPatchTST/logs/${model_name}_${model_id}_plot.log"
echo ""
echo "Running plotting and metrics generation (log: $PLOT_LOG)..."
if ! (cd "$GIT_REPO_ROOT" && PYTHONPATH="$GIT_REPO_ROOT:$GIT_REPO_ROOT/MSVLPatchTST:${PYTHONPATH:-}" python -u "$GIT_REPO_ROOT/main/plot_msvl.py" \
    --model_id_name "$model_id_name" \
    --seq_len $seq_len \
    --pred_len $pred_len \
    $PATCH_STRIDE_PARAMS \
    --results_src "$GIT_REPO_ROOT/output/MSVLPatchTST/test_results" \
    --output_dir "$OUTPUT_DIR" \
    --data_file "$data_path_name" 2>&1 | tee -a "$PLOT_LOG"); then
    echo "Plotting failed. See $PLOT_LOG" >&2
else
    echo "Plotting finished successfully. See $PLOT_LOG for details."
fi

echo ""
echo "================================================================"
echo "All outputs saved to: $OUTPUT_DIR"
echo "  - per_feature_metrics_sl${seq_len}_pl${pred_len}_sp${patch_len_short}_ss${stride_short}_lp${patch_len_long}_ls${stride_long}.csv"
echo "  - test_data_statistics_sl${seq_len}_pl${pred_len}_sp${patch_len_short}_ss${stride_short}_lp${patch_len_long}_ls${stride_long}.csv"
echo "  - prediction_grid_sl${seq_len}_pl${pred_len}_sp${patch_len_short}_ss${stride_short}_lp${patch_len_long}_ls${stride_long}.png"
echo "  - summary_sl${seq_len}_pl${pred_len}_sp${patch_len_short}_ss${stride_short}_lp${patch_len_long}_ls${stride_long}.txt"
echo "================================================================"

exit 0
