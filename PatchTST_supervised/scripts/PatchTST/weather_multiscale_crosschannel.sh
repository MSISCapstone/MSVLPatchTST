#!/bin/bash

# Multi-Scale + Cross-Channel PatchTST for Weather Forecasting
# Combines both enhancements for maximum performance:
#   - Multi-scale: Captures temporal patterns at 6hr, 12hr, 24hr scales
#   - Cross-channel: Enables interaction between weather variables

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

# Enable BOTH enhancements
multi_scale=1
channel_independent=0        # 0 = cross-channel (variables interact)

# Multi-scale configuration
patch_lengths="6,12,24"      # Three temporal scales
patch_strides="3,6,12"       # 50% overlap for each
patch_weights="0.2,0.5,0.3"  # Balanced fusion

for pred_len in 336 720
do
    log_file="/content/logs/LongForecasting/${model_name}_MultiCross_${model_id_name}_${seq_len}_${pred_len}.log"
    
    echo "========================================" | tee "$log_file"
    echo "Multi-Scale + Cross-Channel PatchTST" | tee -a "$log_file"
    echo "seq_len=${seq_len}, pred_len=${pred_len}" | tee -a "$log_file"
    echo "========================================" | tee -a "$log_file"
    
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}'_multiCross_'${seq_len}'_'${pred_len} \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --channel_independent $channel_independent \
      --multi_scale $multi_scale \
      --patch_lengths $patch_lengths \
      --patch_strides $patch_strides \
      --patch_weights $patch_weights \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --fc_dropout 0.2 \
      --head_dropout 0 \
      --patch_len 16 \
      --stride 8 \
      --des 'MultiScale_CrossChannel_Exp' \
      --train_epochs 100 \
      --patience 20 \
      --itr 1 --batch_size 128 --learning_rate 0.0001 2>&1 | tee -a "$log_file"
    
    echo "========================================" | tee -a "$log_file"
    echo "Training completed for pred_len=${pred_len}" | tee -a "$log_file"
    echo "========================================" | tee -a "$log_file"
    
done
