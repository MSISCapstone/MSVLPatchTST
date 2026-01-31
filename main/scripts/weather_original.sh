#!/bin/bash

# Baseline PatchTST for Weather Forecasting (Channel-Independent, Single-Scale)
# Standard configuration for comparison with enhanced versions

# Set paths relative to GIT_REPO_ROOT
GIT_REPO_ROOT=$(git rev-parse --show-toplevel)

# Setup virtual environment
if [ ! -d "$GIT_REPO_ROOT/.venv" ]; then
    echo "Creating virtual environment at $GIT_REPO_ROOT/.venv"
    python3 -m venv "$GIT_REPO_ROOT/.venv"
fi

# Activate virtual environment
source "$GIT_REPO_ROOT/.venv/bin/activate"

if [ ! -d "$GIT_REPO_ROOT/output" ]; then
    mkdir "$GIT_REPO_ROOT/output"
fi

if [ ! -d "$GIT_REPO_ROOT/output/LongForecasting" ]; then
    mkdir "$GIT_REPO_ROOT/output/LongForecasting"
fi

if [ ! -d "$GIT_REPO_ROOT/output/checkpoints" ]; then
    mkdir "$GIT_REPO_ROOT/output/checkpoints"
fi

seq_len=96
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
    log_file="$GIT_REPO_ROOT/output/LongForecasting/${model_name}_Baseline_${model_id_name}_${seq_len}_${pred_len}.log"
    
    echo "========================================" | tee "$log_file"
    echo "Baseline PatchTST" | tee -a "$log_file"
    echo "seq_len=${seq_len}, pred_len=${pred_len}" | tee -a "$log_file"
    echo "========================================" | tee -a "$log_file"
    
    python -u $GIT_REPO_ROOT/PatchTST_supervised/run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}'_baseline_'${seq_len}'_'${pred_len} \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --e_layers 3 \
      --n_heads 8 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --fc_dropout 0.2 \
      --head_dropout 0.0 \
      --patch_len 16 \
      --stride 8 \
      --des 'Baseline_Exp' \
      --checkpoints $GIT_REPO_ROOT/output/checkpoints/ \
      --train_epochs 100 \
      --patience 20 \
      --itr 1 --batch_size 32 --learning_rate 0.0001 2>&1 | tee -a "$log_file"
    
    echo "========================================" | tee -a "$log_file"
    echo "Training completed for pred_len=${pred_len}" | tee -a "$log_file"
    echo "========================================" | tee -a "$log_file"
    
done

echo ""
echo "================================================================"
echo "All experiments completed! Check $GIT_REPO_ROOT/output/LongForecasting/ for results"
echo "================================================================"