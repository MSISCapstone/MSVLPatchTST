#!/bin/bash

# Compare Results Script
# Extracts and compares metrics from all test runs

echo "=========================================="
echo "PatchTST Enhancement Results Comparison"
echo "=========================================="
echo ""

# Find all test log files
logs_dir="logs/LongForecasting"

if [ ! -d "$logs_dir" ]; then
    echo "Error: $logs_dir directory not found"
    exit 1
fi

# Extract metrics function
extract_metrics() {
    log_file=$1
    config_name=$2
    
    if [ ! -f "$log_file" ]; then
        echo "  $config_name: Log file not found"
        return
    fi
    
    # Extract test metrics (usually at the end of log)
    mse=$(grep -i "mse:" "$log_file" | tail -1 | grep -oP "mse:\s*\K[0-9.]+")
    mae=$(grep -i "mae:" "$log_file" | tail -1 | grep -oP "mae:\s*\K[0-9.]+")
    
    if [ -z "$mse" ]; then
        echo "  $config_name: Training incomplete or metrics not found"
    else
        printf "  %-25s MSE: %-10s MAE: %-10s\n" "$config_name" "$mse" "$mae"
    fi
}

# Compare all configurations
echo "Results Summary:"
echo ""

extract_metrics "$logs_dir/PatchTST_Test1_Baseline_weather_336_96.log" "Baseline"
extract_metrics "$logs_dir/PatchTST_Test2_CrossChannel_weather_336_96.log" "Cross-Channel"
extract_metrics "$logs_dir/PatchTST_Test3_MultiScale_weather_336_96.log" "Multi-Scale"
extract_metrics "$logs_dir/PatchTST_Test4_Both_weather_336_96.log" "Both Enhancements"

echo ""
echo "=========================================="
echo ""
echo "To see full logs:"
echo "  cat $logs_dir/PatchTST_Test1_Baseline_weather_336_96.log"
echo "  cat $logs_dir/PatchTST_Test2_CrossChannel_weather_336_96.log"
echo "  cat $logs_dir/PatchTST_Test3_MultiScale_weather_336_96.log"
echo "  cat $logs_dir/PatchTST_Test4_Both_weather_336_96.log"
echo ""
