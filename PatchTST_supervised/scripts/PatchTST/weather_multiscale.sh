#!/bin/bash

# Multi-Scale PatchTST for Weather Forecasting
# Uses variable-length patches (6hr, 12hr, 24hr) to capture:
#   - 6hr patches: Rapid weather changes, frontal passages, convection
#   - 12hr patches: Weather events, semi-diurnal patterns
#   - 24hr patches: Diurnal cycles, daily temperature range

model_name=PatchTST
random_seed=2021

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

# seq_len is context window (input sequence length)
seq_len=336

# Enable multi-scale patching
multi_scale=1
patch_lengths="6,12,24"      # Three scales for weather (hourly data)
patch_strides="3,6,12"       # 50% overlap for each scale
patch_weights="0.2,0.5,0.3"  # Medium scale (12hr) weighted highest

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_multiscale_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --multi_scale $multi_scale \
      --patch_lengths $patch_lengths \
      --patch_strides $patch_strides \
      --patch_weights $patch_weights \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'MultiScale_Exp' \
      --train_epochs 100\
      --patience 20\
      --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_MultiScale_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
