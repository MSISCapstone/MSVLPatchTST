# MSVLPatchTST - Multi-Scale Variable-Length PatchTST

This guide explains how to set up, train, and evaluate MSVLPatchTST compared to the Original PatchTST on weather data.

## Project Structure

```
PatchTST/
├── MSVLPatchTST/              # MSVLPatchTST model implementation
│   ├── run_longExp.py         # Training/inference entry point
│   ├── models.py              # MSVLPatchTST model architecture
│   ├── config.py              # MSVLConfig configuration
│   ├── trainer.py             # Training loop
│   └── evaluation.py          # Evaluation functions
├── PatchTST_supervised/       # Original PatchTST implementation
│   └── run_longExp.py         # Training/inference entry point
├── main/scripts/              # Shell scripts for training/testing
│   ├── setup.sh               # Environment setup
│   ├── weather_msvl_training.sh
│   ├── weather_msvl_test.sh
│   ├── weather_original_training.sh
│   └── weather_original_test.sh
├── main/                      # Plotting scripts
│   ├── plot_msvl.py
│   └── plot_original.py
├── datasets/weather/          # Weather dataset
│   ├── weather.csv            # Original (21 features)
│   └── weather_with_hour.csv  # With hour features (22 features)
└── output/                    # All generated artifacts
    ├── MSVLPatchTST/
    │   ├── logs/
    │   ├── checkpoints/
    │   ├── results/
    │   └── test_results/
    └── Original/
        ├── logs/
        ├── checkpoints/
        ├── results/
        └── test_results/
```

## 1. Setup

Run the setup script to create a virtual environment and install dependencies:

```bash
# From the project root directory
./main/scripts/setup.sh
```

This will:
- Create a virtual environment at `.venv/`
- Install all required packages from `PatchTST_supervised/requirements.txt`

To activate the environment manually in future sessions:
```bash
source .venv/bin/activate
```

## 2. Training

### Train MSVLPatchTST (Multi-Scale Variable-Length)

```bash
./main/scripts/weather_msvl_training.sh
```

**Configuration:**
- Dataset: `weather_with_hour.csv` (20 weather + 2 hour features)
- Input channels: 22
- Sequence length: 336
- Prediction length: 96

### Train Original PatchTST (Baseline)

```bash
./main/scripts/weather_original_training.sh
```

**Configuration:**
- Dataset: `weather.csv` (21 features, no hour embedding)
- Input channels: 21
- Sequence length: 336
- Prediction length: 96

## 3. Testing / Inference

After training, run inference to generate predictions and metrics:

### Test MSVLPatchTST

```bash
./main/scripts/weather_msvl_test.sh
```

### Test Original PatchTST

```bash
./main/scripts/weather_original_test.sh
```

Both test scripts:
- Load the trained checkpoint
- Run sliding window prediction (4 samples × 96 steps = 384 timesteps)
- Generate plots and metrics for **target features only**

## 4. Output Locations

All generated artifacts are organized under `output/`:

### MSVLPatchTST Outputs

| Artifact | Location |
|----------|----------|
| Training logs | `output/MSVLPatchTST/logs/` |
| Checkpoints | `output/MSVLPatchTST/checkpoints/weather_336_96_sp{X}_ss{X}_lp{X}_ls{X}/` |
| Test results | `output/MSVLPatchTST/test_results/weather_336_96_sp{X}_ss{X}_lp{X}_ls{X}/` |

Where `sp`=short patch_len, `ss`=short stride, `lp`=long patch_len, `ls`=long stride.

### Original PatchTST Outputs

| Artifact | Location |
|----------|----------|
| Training logs | `output/Original/logs/` |
| Checkpoints | `output/Original/checkpoints/weather_336_96/` |
| Test results | `output/Original/test_results/weather_336_96/` |

### Generated Files in Test Results

After running test scripts, you'll find:

**MSVLPatchTST** (example with short 16/8, long 32/16):
```
output/MSVLPatchTST/test_results/weather_336_96_sp16_ss8_lp32_ls16/
├── pred_sl336_pl96_sp16_ss8_lp32_ls16.npy                   # Predictions array
├── true_sl336_pl96_sp16_ss8_lp32_ls16.npy                   # Ground truth array
├── predictions_sl336_pl96_sp16_ss8_lp32_ls16.csv            # Combined predictions CSV
├── per_feature_metrics_sl336_pl96_sp16_ss8_lp32_ls16.csv    # MAE, MSE, RMSE, RSE per target feature
├── test_data_statistics_sl336_pl96_sp16_ss8_lp32_ls16.csv   # Test data statistics
├── prediction_grid_sl336_pl96_sp16_ss8_lp32_ls16.png        # 2x3 plot grid
├── summary_sl336_pl96_sp16_ss8_lp32_ls16.txt                # Overall metrics summary
└── results_sl336_pl96_sp16_ss8_lp32_ls16.txt                # Basic results from inference
```

**Original PatchTST**:
```
output/Original/test_results/weather_336_96/
├── pred.npy                              # Predictions array
├── true.npy                              # Ground truth array
├── predictions.csv                       # Combined predictions CSV
├── per_feature_metrics.csv               # MAE, MSE, RMSE, RSE per target feature
├── test_data_statistics.csv              # Test data statistics
├── prediction_grid_sl336_pl96.png        # 2x3 plot grid for target features
└── summary.txt                           # Overall metrics summary
```

## 5. Target Features

Both models are evaluated on these 6 target features:

| Feature | Description |
|---------|-------------|
| `p (mbar)` | Pressure |
| `T (degC)` | Temperature |
| `wv (m/s)` | Wind velocity |
| `max. wv (m/s)` | Maximum wind velocity |
| `rain (mm)` | Rainfall |
| `raining (s)` | Raining duration |

## 6. Training & evaluation (technical details)

- **Loss used** ✅
  - Training uses the criterion selected via `--loss` in `MSVLPatchTST/run_longExp.py` (options: `huber` → `torch.nn.HuberLoss`, `mse` → `MSELoss`, `mae` → `L1Loss`).
  - The training loop (`MSVLPatchTST/trainer.py`) computes a **group‑weighted** loss (each channel group multiplied by its `patch_configs[<group>]['weight']`) and backpropagates on the summed loss.

- **Target features** 🎯
  - The model predicts **6 weather targets** (see section above). Group mapping (from `MSVLPatchTST/config.py`):
    - `Channel Group - 1` → `p (mbar)` (idx 0), `T (degC)` (idx 1), `rain (mm)` (idx 14)
    - `Channel Group - 2` → `wv (m/s)` (idx 11), `max. wv (m/s)` (idx 12), `raining (s)` (idx 15)
  - Evaluation and loss are computed only on these 6 target channels.

- **Learning rate & scheduler** 🔧
  - Default LR: `1e-4` (`config.learning_rate`, overridable with `--learning_rate`).
  - Optimizer: `AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)`.
  - Scheduler: default `OneCycleLR(max_lr=config.learning_rate)`; also supports epochwise policies via `--lradj` (e.g. `type3`, `constant`, `TST`) handled by `adjust_learning_rate`.

- **Other critical items** ⚠️
  - **RevIN normalization**: inputs (all 22 channels) are normalized/denormalized in `models.py`; denormalization is applied to the 6 target outputs using stored RevIN stats.
  - **Early stopping & checkpoints**: EarlyStopping saves the best model to `checkpoint.pth` and `trainer.py` loads this best checkpoint after training.
  - **Training loader detail**: `train_model` currently concatenates train/val/test datasets into a single `combined_loader` for training iterations — verify this behavior if you expect strict train/val/test separation.
  - **Evaluation metrics**: `MSVLPatchTST/evaluation.py` reports MAE, MSE, RMSE, MAPE, MSPE, RSE and Pearson correlation; per‑channel metrics are available via `evaluate_per_channel`.
  - **Sliding-window evaluation** helper (`evaluate_model_sliding_window`) is provided for multi-window inference experiments.

## 7. Model Architectures

### Original PatchTST Architecture

```
Input: [batch, seq_len=336, channels=21]
    │
    ▼
┌─────────────────────────────────────────┐
│  Patching (patch_len=16, stride=8)      │
│  → 42 patches per channel               │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Patch Embedding (Linear: 16 → 128)     │
│  + Positional Encoding                  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Transformer Encoder (3 layers)         │
│  - Multi-head Attention (16 heads)      │
│  - Feed-forward (d_ff=256)              │
│  - Channel-Independent Processing       │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Flatten + Linear Head                  │
│  → pred_len=96 per channel              │
└─────────────────────────────────────────┘
    │
    ▼
Output: [batch, pred_len=96, channels=21]
```

**Key characteristics:**
- Channel-independent: Each channel processed separately
- Single-scale patching: Fixed patch length of 16
- No temporal encoding: No hour-of-day information
- Parameters: ~2.5M

### MSVLPatchTST Architecture

```
Input: [batch, seq_len=336, channels=22]
       (20 weather + 2 hour features: sin/cos)
    │
    ▼
┌─────────────────────────────────────────┐
│  RevIN Normalization                    │
└─────────────────────────────────────────┘
    │
    ├──────────────────┬──────────────────┐
    ▼                  ▼                  │
┌──────────────┐ ┌──────────────┐         │
│ Long-Scale   │ │ Short-Scale  │         │
│ Encoder      │ │ Encoder      │         │
│              │ │              │         │
│ Targets:     │ │ Targets:     │         │
│ p, T, rain   │ │ wv, max.wv,  │         │
│              │ │ raining      │         │
│ patch_len=16 │ │ patch_len=12 │         │
│ stride=8     │ │ stride=4     │         │
│              │ │              │         │
│ ┌──────────┐ │ │ ┌──────────┐ │         │
│ │Attention │ │ │ │Attention │ │         │
│ │(22 heads)│ │ │ │(22 heads)│ │         │
│ └────┬─────┘ │ │ └────┬─────┘ │         │
│      ▼       │ │      ▼       │         │
│ ┌──────────┐ │ │ ┌──────────┐ │         │
│ │   FFN    │ │ │ │   FFN    │ │         │
│ │ d→4d→d  │ │ │ │ d→4d→d  │ │         │
│ │  GELU    │ │ │ │  GELU    │ │         │
│ └────┬─────┘ │ │ └────┬─────┘ │         │
│      ▼       │ │      ▼       │         │
│ ┌──────────┐ │ │ ┌──────────┐ │         │
│ │  Head    │ │ │ │  Head    │ │         │
│ └──────────┘ │ │ └──────────┘ │         │
└──────────────┘ └──────────────┘         │
    │                  │                  │
    │  out=[0,1,2]     │  out=[3,4,5]     │
    ▼                  ▼                  │
┌─────────────────────────────────────────┐
│  Concatenate (non-overlapping outputs)  │
│  → [batch, pred_len, 6]                 │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Cross-Group Attention                  │
│  - Learn inter-variable dependencies    │
│  - p, T, rain ↔ wv, max.wv, raining   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  FFN (Cross-Group Refinement)           │
│  - d_model → 4*d_model → d_model        │
│  - GELU + Dropout + Residual            │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  RevIN Denormalization                  │
└─────────────────────────────────────────┘
    │
    ▼
Output: [batch, pred_len=96, channels=6]
        (6 target features: p, T, rain, wv, max.wv, raining)
```

**Key characteristics:**
- Multi-scale patching: Long (16) for slow dynamics, Short (12) for fast dynamics
- Grouping: 
  - Channel Group - 1: p, T, rain (slow dynamics)
  - Channel Group - 2: wv, max.wv, raining (fast dynamics)
- Hour integration: sin/cos hour features in all encoders
- Cross-group attention: Learns physical couplings between groups
- MSE optimized only on 6 target features
- Parameters: ~21M

## 8. Model Comparison

| Aspect | Original PatchTST | MSVLPatchTST |
|--------|-------------------|--------------|
| Input channels | 21 | 22 (20 weather + 2 hour) |
| Output channels | 21 | 6 (target features only) |
| Patching | Single-scale (16) | Multi-scale (12, 16) |
| Hour embedding | None | Integrated (sin/cos) |
| Channel grouping | Independent | Groups |
| Cross-group attention | No | Yes |
| MSE optimized on | All 21 channels | 6 target features |

## Quick Start

### Basic Usage (Default Configuration)

```bash
# 1. Setup environment
./main/scripts/setup.sh

# 2. Train both models (default config)
./main/scripts/weather_msvl_training.sh
./main/scripts/weather_original_training.sh

# 3. Test and compare
./main/scripts/weather_msvl_test.sh
./main/scripts/weather_original_test.sh

# 4. Check results
cat output/MSVLPatchTST/test_results/weather_336_96_sp16_ss8_lp16_ls8/summary_sl336_pl96_sp16_ss8_lp16_ls8.txt
cat output/Original/test_results/weather_336_96/summary.txt
```

### Using Pre-configured Patch/Stride Settings

MSVLPatchTST supports multiple patch/stride configurations via config files in `main/scripts/patch_len_configs/`:

```bash
# Config: short 8/4, long 16/8 (fine short, medium long)
./main/scripts/weather_msvl_training.sh main/scripts/patch_len_configs/ps_config_08-04-16-08.txt
./main/scripts/weather_msvl_test.sh main/scripts/patch_len_configs/ps_config_08-04-16-08.txt

# Config: short 16/8, long 8/4 (medium short, fine long)
./main/scripts/weather_msvl_training.sh main/scripts/patch_len_configs/ps_config_16-08-08-04.txt
./main/scripts/weather_msvl_test.sh main/scripts/patch_len_configs/ps_config_16-08-08-04.txt

# Config: short 16/8, long 32/16 (medium short, coarse long)
./main/scripts/weather_msvl_training.sh main/scripts/patch_len_configs/ps_config_16-08-32-16.txt
./main/scripts/weather_msvl_test.sh main/scripts/patch_len_configs/ps_config_16-08-32-16.txt

# Config: short 32/16, long 16/8 (coarse short, medium long)
./main/scripts/weather_msvl_training.sh main/scripts/patch_len_configs/ps_config_32-16-16-08.txt
./main/scripts/weather_msvl_test.sh main/scripts/patch_len_configs/ps_config_32-16-16-08.txt
```

### Using Custom Parameters

You can also pass parameters directly:

```bash
# Format: <patch_len_short> <stride_short> <patch_len_long> <stride_long>
./main/scripts/weather_msvl_training.sh 12 6 24 12
./main/scripts/weather_msvl_test.sh 12 6 24 12
```

### Batch Experiments

Run all configurations in sequence:

```bash
# Train all configurations
for config in main/scripts/patch_len_configs/ps_config_*.txt; do
    echo "Training with $config"
    ./main/scripts/weather_msvl_training.sh "$config"
done

# Test all configurations
for config in main/scripts/patch_len_configs/ps_config_*.txt; do
    echo "Testing with $config"
    ./main/scripts/weather_msvl_test.sh "$config"
done
```
