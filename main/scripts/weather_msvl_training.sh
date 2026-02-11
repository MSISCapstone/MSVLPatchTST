#!/bin/bash

# MSVLPatchTST for Weather Forecasting
# Multi-Scale Variable-Length PatchTST with exact parameters as baseline
# Uses new architecture from MSVLPatchTST module

# Parse command line arguments
# Usage option 1: Pass config file
#   ./weather_msvl_training.sh patch_len_configs/ps_config_1.txt
# Usage option 2: Pass individual parameters
#   ./weather_msvl_training.sh 8 4 16 8

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
    # Setup virtual environment
    if [ ! -d "$GIT_REPO_ROOT/.venv" ]; then
        echo "Creating virtual environment at $GIT_REPO_ROOT/.venv"
        python3 -m venv "$GIT_REPO_ROOT/.venv"
    fi

    # Activate virtual environment (POSIX and Windows venv compatibility)
    if [ -f "$GIT_REPO_ROOT/.venv/bin/activate" ]; then
        echo "Activating virtualenv (bin/activate)"
        source "$GIT_REPO_ROOT/.venv/bin/activate"
    elif [ -f "$GIT_REPO_ROOT/.venv/Scripts/activate" ]; then
        echo "Activating virtualenv (Scripts/activate)"
        source "$GIT_REPO_ROOT/.venv/Scripts/activate"
    else
        echo "No activation script found in $GIT_REPO_ROOT/.venv; attempting to create venv with python3"
        python3 -m venv "$GIT_REPO_ROOT/.venv"
        if [ -f "$GIT_REPO_ROOT/.venv/bin/activate" ]; then
            source "$GIT_REPO_ROOT/.venv/bin/activate"
        elif [ -f "$GIT_REPO_ROOT/.venv/Scripts/activate" ]; then
            source "$GIT_REPO_ROOT/.venv/Scripts/activate"
        else
            echo "Failed to create or locate activation script in $GIT_REPO_ROOT/.venv" >&2
            exit 1
        fi
    fi
fi

# Create necessary directories for MSVLPatchTST outputs
if [ ! -d "$GIT_REPO_ROOT/output/MSVLPatchTST" ]; then
    mkdir -p "$GIT_REPO_ROOT/output/MSVLPatchTST"
fi

if [ ! -d "$GIT_REPO_ROOT/output/MSVLPatchTST/logs" ]; then
    mkdir -p "$GIT_REPO_ROOT/output/MSVLPatchTST/logs"
fi

if [ ! -d "$GIT_REPO_ROOT/output/MSVLPatchTST/checkpoints" ]; then
    mkdir -p "$GIT_REPO_ROOT/output/MSVLPatchTST/checkpoints"
fi

if [ ! -d "$GIT_REPO_ROOT/output/MSVLPatchTST/results" ]; then
    mkdir -p "$GIT_REPO_ROOT/output/MSVLPatchTST/results"
fi

if [ ! -d "$GIT_REPO_ROOT/output/MSVLPatchTST/test_results" ]; then
    mkdir -p "$GIT_REPO_ROOT/output/MSVLPatchTST/test_results"
fi

# Exact same parameters as original
seq_len=336
model_name=MSVLPatchTST
DROPOUT=0.2

# Dataset configuration - using weather_with_hour.csv for MSVL
root_path_name=$GIT_REPO_ROOT/datasets/weather/
data_path_name=weather_with_hour.csv
model_id_name=weather
data_name=custom

random_seed=2021

echo "Running MSVLPatchTST (Multi-Scale Variable-Length PatchTST)"
echo "Parameters: short[p${patch_len_short}_s${stride_short}] long[p${patch_len_long}_s${stride_long}]"
echo "================================================================"

for pred_len in 96
do
    log_file="$GIT_REPO_ROOT/output/MSVLPatchTST/logs/${model_name}_${model_id_name}_${seq_len}_${pred_len}_sp${patch_len_short}_ss${stride_short}_lp${patch_len_long}_ls${stride_long}.log"
    
    echo "========================================" | tee "$log_file"
    echo "MSVLPatchTST" | tee -a "$log_file"
    echo "seq_len=${seq_len}, pred_len=${pred_len}" | tee -a "$log_file"
    echo "Short channel: patch_len=${patch_len_short}, stride=${stride_short}" | tee -a "$log_file"
    echo "Long channel: patch_len=${patch_len_long}, stride=${stride_long}" | tee -a "$log_file"
    echo "========================================" | tee -a "$log_file"
    
    # Run MSVLPatchTST with exact same parameters as original
    python -u "$GIT_REPO_ROOT/MSVLPatchTST/run_longExp.py" \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path "$root_path_name" \
      --data_path "$data_path_name" \
      --model_id "${model_id_name}_${seq_len}_${pred_len}_sp${patch_len_short}_ss${stride_short}_lp${patch_len_long}_ls${stride_long}" \
      --model MSVLPatchTST \
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
      --dropout $DROPOUT \
      --fc_dropout $DROPOUT  \
      --head_dropout 0 \
      $PATCH_STRIDE_PARAMS \
      --des 'Exp' \
      --train_epochs 100 \
      --patience 3 \
      --checkpoints "$GIT_REPO_ROOT/output/MSVLPatchTST/checkpoints" \
      --itr 1 --batch_size 128 --learning_rate 0.0001 >> "$log_file" 2>&1
    
    echo "========================================" | tee -a "$log_file"
    echo "Training completed for pred_len=${pred_len}" | tee -a "$log_file"
    echo "========================================" | tee -a "$log_file"
    
done

echo ""
echo "================================================================"
echo "All experiments completed! Check $GIT_REPO_ROOT/output/MSVLPatchTST/ for results"
echo "================================================================"
