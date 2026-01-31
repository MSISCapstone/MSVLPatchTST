#!/bin/bash

# MSVLPatchTST for Weather Forecasting
# Multi-Scale Variable-Length PatchTST with exact parameters as baseline
# Uses new architecture from MSVLPatchTST module

# Set paths relative to GIT_REPO_ROOT
GIT_REPO_ROOT=$(git rev-parse --show-toplevel)

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

# Dataset configuration - using weather_with_hour.csv for MSVL
root_path_name=$GIT_REPO_ROOT/datasets/weather/
data_path_name=weather_with_hour.csv
model_id_name=weather
data_name=custom

random_seed=2021

echo "Running MSVLPatchTST (Multi-Scale Variable-Length PatchTST)"
echo "================================================================"

for pred_len in 96
do
    log_file="$GIT_REPO_ROOT/output/MSVLPatchTST/logs/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log"
    
    echo "========================================" | tee "$log_file"
    echo "MSVLPatchTST" | tee -a "$log_file"
    echo "seq_len=${seq_len}, pred_len=${pred_len}" | tee -a "$log_file"
    echo "========================================" | tee -a "$log_file"
    
    # Run MSVLPatchTST with exact same parameters as original
    python -u "$GIT_REPO_ROOT/MSVLPatchTST/run_longExp.py" \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path "$root_path_name" \
      --data_path "$data_path_name" \
      --model_id "${model_id_name}_${seq_len}_${pred_len}" \
      --model MSVLPatchTST \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 22 \
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
      --train_epochs 100 \
      --patience 20 \
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
