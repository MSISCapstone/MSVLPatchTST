#!/bin/bash

# Cross-Channel Weather Forecasting with PatchTST
# This script demonstrates how to use the cross-channel enhancement
# Enables variable interaction (temperature, humidity, wind, etc. interact)

if [ ! -d "/content/logs" ]; then
    mkdir /content/logs
fi

if [ ! -d "/content/logs/LongForecasting" ]; then
    mkdir /content/logs/LongForecasting
fi

seq_len=336
model_name=PatchTST

# Dataset configuration
root_path_name=/content/PatchTST/datasets/weather/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

random_seed=2021

echo "Running PatchTST with Cross-Channel Interaction for Weather Forecasting"
echo "========================================================================="

for pred_len in 336 720
do
    log_file="/content/logs/LongForecasting/${model_name}_CrossChannel_${model_id_name}_${seq_len}_${pred_len}.log"
    
    echo "========================================" | tee "$log_file"
    echo "Cross-Channel PatchTST (Single-Scale)" | tee -a "$log_file"
    echo "seq_len=${seq_len}, pred_len=${pred_len}" | tee -a "$log_file"
    echo "========================================" | tee -a "$log_file"
    
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}'_crosschannel_'${seq_len}'_'${pred_len} \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --channel_independent 0 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --fc_dropout 0.2 \
      --head_dropout 0 \
      --patch_len 16 \
      --stride 8 \
      --des 'CrossChannel_Exp' \
      --train_epochs 100 \
      --patience 20 \
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.0001 2>&1 | tee -a "$log_file"
    
    echo "========================================" | tee -a "$log_file"
    echo "Training completed for pred_len=${pred_len}" | tee -a "$log_file"
    echo "========================================" | tee -a "$log_file"
    
done

echo ""
echo "========================================================================="
echo "All experiments completed! Check /content/logs/LongForecasting/ for results"
echo "========================================================================="
