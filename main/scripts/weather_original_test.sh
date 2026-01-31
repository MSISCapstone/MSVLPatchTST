#!/bin/bash

# Inference-only script for Weather dataset
# - Uses the most-recent checkpoint under $GIT_REPO_ROOT/output/checkpoints
# - Prints test-set statistics (using same split logic as Dataset_Custom)
# - Runs test/inference only (no training)

set -euo pipefail

# Set paths relative to GIT_REPO_ROOT
GIT_REPO_ROOT=$(git rev-parse --show-toplevel)

# Activate virtual environment (POSIX + Windows venv handling; strip CRLF if needed)
if [ -f "$GIT_REPO_ROOT/.venv/bin/activate" ]; then
    source "$GIT_REPO_ROOT/.venv/bin/activate"
elif [ -f "$GIT_REPO_ROOT/.venv/Scripts/activate" ]; then
    # Try to strip CRLF in case this is a Windows-created venv
    sed -i 's/\r$//' "$GIT_REPO_ROOT/.venv/Scripts/activate" 2>/dev/null || true
    source "$GIT_REPO_ROOT/.venv/Scripts/activate"
else
    echo "No virtualenv activation found at $GIT_REPO_ROOT/.venv - creating one with python3..."
    python3 -m venv "$GIT_REPO_ROOT/.venv"
    source "$GIT_REPO_ROOT/.venv/bin/activate"
fi

# Default experiment parameters (match training defaults used by weather_original.sh)
seq_len=336
label_len=48
pred_len=96
model_name=PatchTST
root_path_name=$GIT_REPO_ROOT/datasets/weather/
data_path_name=weather.csv
model_id_name=weather
data_name=custom
random_seed=2021

echo "===== Weather inference (test only) ====="
echo "Project root: $GIT_REPO_ROOT"

# Construct checkpoint path directly (no search needed - we know the exact pattern)
CKPT_BASE="$GIT_REPO_ROOT/main/scripts/checkpoints"
if [ ! -d "$CKPT_BASE" ]; then
    echo "No checkpoints directory found at $CKPT_BASE" >&2
    exit 1
fi

# Construct the expected setting name based on training parameters
# Pattern: {model_id}_{seq_len}_{pred_len}_PatchTST_custom_ftM_sl{seq_len}_ll{label_len}_pl{pred_len}_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0
setting="${model_id_name}_${seq_len}_${pred_len}_PatchTST_custom_ftM_sl${seq_len}_ll${label_len}_pl${pred_len}_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0"
latest_dir="$CKPT_BASE/$setting"

if [ ! -d "$latest_dir" ]; then
    echo "Expected checkpoint not found at: $latest_dir" >&2
    echo "Available checkpoints:" >&2
    ls -1 "$CKPT_BASE" >&2
    exit 1
fi
setting=$(basename "$latest_dir")
ckpt_file="$latest_dir/checkpoint.pth"

if [ ! -f "$ckpt_file" ]; then
    echo "No checkpoint file found at $ckpt_file" >&2
    exit 1
fi

# Extract model_id from the setting name (assumes pattern: <model_id>_<ModelName>_...)
model_marker="_${model_name}_"
if [[ "$setting" == *"$model_marker"* ]]; then
    model_id_from_setting="${setting%%$model_marker*}"
else
    # fallback: take everything before the second underscore if possible
    model_id_from_setting="$(echo "$setting" | awk -F'_' '{print $1}')"
fi

echo "\nUsing checkpoint setting: $setting"
echo "Inferred model_id: $model_id_from_setting"
echo "Checkpoint file: $ckpt_file"

# Locate run_longExp.py
if [ -f "$GIT_REPO_ROOT/PatchTST_supervised/run_longExp.py" ]; then
    run_script="$GIT_REPO_ROOT/PatchTST_supervised/run_longExp.py"
else
    run_script=$(find "$GIT_REPO_ROOT" -name run_longExp.py | head -n1 || true)
fi
if [ -z "$run_script" ]; then
    echo "Could not find run_longExp.py in repository. Ensure PatchTST_supervised is present." >&2
    exit 1
fi

# Run test-only (uses the inferred model_id so the constructed 'setting' matches the checkpoint folder)
echo "\nRunning inference using test set..."
(cd "$GIT_REPO_ROOT/PatchTST_supervised" && \
  python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 0 \
    --root_path "$root_path_name" \
    --data_path "$data_path_name" \
    --model_id "$model_id_from_setting" \
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
    --checkpoints "$CKPT_BASE" )

# Run plotting and metrics generation using the predictions from this run
PLOT_LOG="$GIT_REPO_ROOT/logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}_plot.log"
echo "\nRunning plotting and metrics generation (log: $PLOT_LOG)..."
if ! (cd "$GIT_REPO_ROOT" && PYTHONPATH="$GIT_REPO_ROOT:$GIT_REPO_ROOT/PatchTST_supervised:${PYTHONPATH:-}" python -u "$GIT_REPO_ROOT/main/plot_original.py" \
    --model_id_name "$model_id_name" \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --results_src "$GIT_REPO_ROOT/PatchTST_supervised/results/${setting}" 2>&1 | tee -a "$PLOT_LOG"); then
    echo "Plotting failed. See $PLOT_LOG" >&2
else
    echo "Plotting finished successfully. See $PLOT_LOG for details."
fi

echo "\nInference finished. Results (npys, plots, etc.) are in $GIT_REPO_ROOT/PatchTST_supervised/results/${setting}/ and $GIT_REPO_ROOT/PatchTST_supervised/test_results/${setting}/; plots & summaries also in $GIT_REPO_ROOT/output/results/${model_id_name}_${seq_len}_${pred_len}/"

exit 0
