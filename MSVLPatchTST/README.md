# MSVLPatchTST (Multi-Scale Variable-Length PatchTST)

Python package for MSVLPatchTST with variable-length patching, physics-based grouping, and hour-of-day integration.

## Installation

Add the MSVLPatchTST directory to your Python path:

```python
import sys
sys.path.append('./MSVLPatchTST')
```

## Quick Start

```python
import torch
from MSVLPatchTST.config import MSVLConfig
from MSVLPatchTST.models import MSVLPatchTST
from MSVLPatchTST.training_utils import set_seed, get_target_indices, get_scheduler
from MSVLPatchTST.trainer import train_model
from MSVLPatchTST.evaluation import evaluate_model

# Set seed for reproducibility
args = MSVLConfig()
set_seed(args.random_seed)

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MSVLPatchTST(args).float().to(device)

# Get target indices
target_indices, target_names = get_target_indices(args.channel_groups)

# Create optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
scheduler = get_scheduler(optimizer, num_warmup_steps=500, num_training_steps=len(train_loader) * args.train_epochs)

# Train model
criterion = torch.nn.MSELoss()
history = train_model(
    model, train_loader, val_loader, test_loader,
    optimizer, scheduler, criterion, args, device,
    target_indices, checkpoint_path
)

# Evaluate model
results = evaluate_model(model, test_loader, device, args)
```

## Module Structure

- `config.py` - Configuration class with physics-based channel grouping
- `models.py` - Physics-Integrated PatchTST model architecture
- `utils.py` - Utility functions (seed setting, validation, scheduling)
- `trainer.py` - Training loop
- `evaluation.py` - Evaluation metrics and functions
- `data_preprocessing.py` - Data preprocessing utilities

## Features

- **Predictor-Based Grouping**: Variables grouped by prediction targets (rain, temperature, wind)
- **Variable-Length Patching**: Different patch lengths for each group based on physical time scales
- **Hour-of-Day Integration**: Cyclical hour features integrated into all predictor groups
- **Cross-Group Attention**: Learn inter-variable dependencies
- **RevIN Normalization**: Per-channel instance normalization

## Target Variables

The model predicts 7 key weather parameters:
- Rain: `rain (mm)`, `raining (s)`
- Temperature: `T (degC)`, `Tpot (K)`, `Tdew (degC)`
- Wind: `wv (m/s)`, `max. wv (m/s)`

## Configuration

Customize your configuration by modifying `MSVLConfig`:

```python
args = MSVLConfig()
args.seq_len = 512  # Input sequence length
args.pred_len = 336  # Prediction length
args.d_model = 128  # Model dimension
args.n_heads = 8    # Number of attention heads
args.batch_size = 32
args.learning_rate = 0.0001
```

## Requirements

- PyTorch >= 1.9
- NumPy
- Pandas

## License

See LICENSE file for details.
