#!/bin/bash

# Comprehensive Testing Script for PatchTST Enhancements
# Tests all configurations: baseline, cross-channel, multi-scale, and both

echo "=========================================="
echo "PatchTST Enhancement Testing Suite"
echo "=========================================="
echo ""

# Configuration
model_name=PatchTST
random_seed=2021
root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom
seq_len=336
pred_len=96  # Short test - use 96 for faster validation

# Model parameters (smaller for faster testing)
e_layers=2
n_heads=8
d_model=64
d_ff=128
batch_size=64
train_epochs=5  # Short for testing - increase to 100 for production

echo "Test Configuration:"
echo "  Data: $data_path_name"
echo "  Sequence Length: $seq_len"
echo "  Prediction Length: $pred_len"
echo "  Training Epochs: $train_epochs (short for testing)"
echo "  Batch Size: $batch_size"
echo ""

# Create logs directory
mkdir -p logs/LongForecasting

# Test 1: Baseline (no enhancements)
echo "=========================================="
echo "Test 1/4: Baseline (No Enhancements)"
echo "=========================================="
python -u run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_baseline_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 21 \
  --channel_independent 1 \
  --multi_scale 0 \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --dropout 0.2 \
  --fc_dropout 0.2 \
  --head_dropout 0 \
  --patch_len 16 \
  --stride 8 \
  --des 'Test_Baseline' \
  --train_epochs $train_epochs \
  --patience 10 \
  --itr 1 \
  --batch_size $batch_size \
  --learning_rate 0.0001 \
  >logs/LongForecasting/$model_name'_Test1_Baseline_'$model_id_name'_'$seq_len'_'$pred_len.log 

echo "✓ Test 1 complete - Check logs/LongForecasting/ for results"
echo ""

# Test 2: Cross-Channel Only
echo "=========================================="
echo "Test 2/4: Cross-Channel Only"
echo "=========================================="
python -u run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_crosschannel_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 21 \
  --channel_independent 0 \
  --multi_scale 0 \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --dropout 0.2 \
  --fc_dropout 0.2 \
  --head_dropout 0 \
  --patch_len 16 \
  --stride 8 \
  --des 'Test_CrossChannel' \
  --train_epochs $train_epochs \
  --patience 10 \
  --itr 1 \
  --batch_size $batch_size \
  --learning_rate 0.0001 \
  >logs/LongForecasting/$model_name'_Test2_CrossChannel_'$model_id_name'_'$seq_len'_'$pred_len.log 

echo "✓ Test 2 complete - Check logs/LongForecasting/ for results"
echo ""

# Test 3: Multi-Scale Only
echo "=========================================="
echo "Test 3/4: Multi-Scale Only (3 scales)"
echo "=========================================="
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
  --channel_independent 1 \
  --multi_scale 1 \
  --patch_lengths "6,12,24" \
  --patch_strides "3,6,12" \
  --patch_weights "0.2,0.5,0.3" \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --dropout 0.2 \
  --fc_dropout 0.2 \
  --head_dropout 0 \
  --patch_len 16 \
  --stride 8 \
  --des 'Test_MultiScale' \
  --train_epochs $train_epochs \
  --patience 10 \
  --itr 1 \
  --batch_size $batch_size \
  --learning_rate 0.0001 \
  >logs/LongForecasting/$model_name'_Test3_MultiScale_'$model_id_name'_'$seq_len'_'$pred_len.log 

echo "✓ Test 3 complete - Check logs/LongForecasting/ for results"
echo ""

# Test 4: Both Enhancements
echo "=========================================="
echo "Test 4/4: Cross-Channel + Multi-Scale"
echo "=========================================="
python -u run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_both_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 21 \
  --channel_independent 0 \
  --multi_scale 1 \
  --patch_lengths "6,12,24" \
  --patch_strides "3,6,12" \
  --patch_weights "0.2,0.5,0.3" \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --dropout 0.2 \
  --fc_dropout 0.2 \
  --head_dropout 0 \
  --patch_len 16 \
  --stride 8 \
  --des 'Test_Both' \
  --train_epochs $train_epochs \
  --patience 10 \
  --itr 1 \
  --batch_size $batch_size \
  --learning_rate 0.0001 \
  >logs/LongForecasting/$model_name'_Test4_Both_'$model_id_name'_'$seq_len'_'$pred_len.log 

echo "✓ Test 4 complete - Check logs/LongForecasting/ for results"
echo ""

# Summary
echo "=========================================="
echo "All Tests Complete!"
echo "=========================================="
echo ""
echo "Results saved to logs/LongForecasting/"
echo ""
echo "Log files:"
echo "  1. Baseline:           ${model_name}_Test1_Baseline_${model_id_name}_${seq_len}_${pred_len}.log"
echo "  2. Cross-Channel:      ${model_name}_Test2_CrossChannel_${model_id_name}_${seq_len}_${pred_len}.log"
echo "  3. Multi-Scale:        ${model_name}_Test3_MultiScale_${model_id_name}_${seq_len}_${pred_len}.log"
echo "  4. Both Enhancements:  ${model_name}_Test4_Both_${model_id_name}_${seq_len}_${pred_len}.log"
echo ""
echo "To view results:"
echo "  grep 'mse\\|mae\\|rmse' logs/LongForecasting/${model_name}_Test*.log"
echo ""
echo "To compare all tests:"
echo "  bash scripts/compare_results.sh"
echo ""
