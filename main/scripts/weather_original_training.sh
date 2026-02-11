#!/bin/bash

# Baseline PatchTST for Weather Forecasting (Channel-Independent, Single-Scale)
# Standard configuration for comparison with enhanced versions

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

# Create necessary directories for Original PatchTST outputs
if [ ! -d "$GIT_REPO_ROOT/output/Original" ]; then
    mkdir -p "$GIT_REPO_ROOT/output/Original"
fi

if [ ! -d "$GIT_REPO_ROOT/output/Original/logs" ]; then
    mkdir -p "$GIT_REPO_ROOT/output/Original/logs"
fi

if [ ! -d "$GIT_REPO_ROOT/output/Original/checkpoints" ]; then
    mkdir -p "$GIT_REPO_ROOT/output/Original/checkpoints"
fi

if [ ! -d "$GIT_REPO_ROOT/output/Original/results" ]; then
    mkdir -p "$GIT_REPO_ROOT/output/Original/results"
fi

if [ ! -d "$GIT_REPO_ROOT/output/Original/test_results" ]; then
    mkdir -p "$GIT_REPO_ROOT/output/Original/test_results"
fi

seq_len=336
model_name=PatchTST

# Dataset configuration
root_path_name=$GIT_REPO_ROOT/datasets/weather/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

random_seed=2021

echo "Running Baseline PatchTST (Channel-Independent, Single-Scale)"
echo "================================================================"

for pred_len in 96
do
    log_file="$GIT_REPO_ROOT/output/Original/logs/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log"
    
    echo "========================================" | tee "$log_file"
    echo "Baseline PatchTST" | tee -a "$log_file"
    echo "seq_len=${seq_len}, pred_len=${pred_len}" | tee -a "$log_file"
    echo "========================================" | tee -a "$log_file"
    
    python -u "$GIT_REPO_ROOT/PatchTST_supervised/run_longExp.py" \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path "$root_path_name" \
      --data_path "$data_path_name" \
      --model_id "${model_id_name}_${seq_len}_${pred_len}" \
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
      --train_epochs 100 \
      --patience 20 \
      --checkpoints "$GIT_REPO_ROOT/output/Original/checkpoints" \
      --itr 1 --batch_size 128 --learning_rate 0.0001 >> "$log_file" 2>&1
    
    echo "========================================" | tee -a "$log_file"
    echo "Training completed for pred_len=${pred_len}" | tee -a "$log_file"
    echo "========================================" | tee -a "$log_file"
    
done

echo ""
echo "================================================================"
echo "All experiments completed! Check $GIT_REPO_ROOT/output/Original/ for results"
echo "================================================================"