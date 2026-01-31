# Weather Forecasting Model Comparison: DLinear vs Physics-Integrated PatchTST

**Date:** January 26, 2026  
**Dataset:** Weather Dataset (420,551 hourly records, 21 features)  
**Task:** 336-hour (14-day) ahead weather forecasting  
**Target Variables:** 6 key weather features (p, T, wv, max. wv, rain, raining)

---

## Executive Summary

This document compares two approaches for weather forecasting:
1. **Baseline (DLinear)**: Simple linear model from PatchTST framework
2. **Physics-Integrated PatchTST**: Custom architecture with domain-specific channel grouping and cross-group attention

### Key Findings

| Metric | DLinear (Baseline) | Physics-Integrated | Difference | Improvement |
|--------|-------------------|-------------------|------------|-------------|
| **MSE (Target Features)** | 0.4687 | 0.5181 | +0.0494 | -10.5% |
| **MAE (Target Features)** | 0.3518 | 0.4269 | +0.0751 | -21.3% |
| **Test Samples** | 10,192 | 10,176 | -16 | - |

**Conclusion:** The baseline DLinear model currently outperforms the Physics-Integrated PatchTST model on both MSE and MAE metrics for the 6 target weather variables.

---

## 1. Model Architectures

### 1.1 DLinear (Baseline)

**Architecture Type:** Simple linear decomposition model

**Key Characteristics:**
- Direct linear mapping from input sequence to output prediction
- Decomposes time series into trend and seasonal components
- Minimal parameters (~500K-1M parameters typically)
- No explicit weather physics modeling
- Uses standard time features (hour, day, month, etc.)

**Training Configuration:**
- Sequence length: 336 hours (14 days)
- Prediction length: 336 hours (14 days)
- Label length: 48 hours
- Features: Multivariate (21 features → 21 features)
- Optimizer: AdamW
- Learning rate: Default from PatchTST framework
- Data split: 70% train / 10% val / 20% test

**Strengths:**
- Simple, efficient architecture
- Fast training and inference
- Proven baseline for time series forecasting
- Less prone to overfitting

**Limitations:**
- No domain knowledge incorporation
- Treats all features equally
- No explicit modeling of feature interactions
- Limited capacity for complex patterns

---

### 1.2 Physics-Integrated PatchTST

**Architecture Type:** Custom patch-based transformer with physics-inspired design

**Key Characteristics:**
- Channel grouping based on weather physics:
  - **Long-term channels**: Slow-changing atmospheric variables (pressure, temperature, wind)
  - **Short-term channels**: Fast-changing precipitation variables (rain, raining duration)
  - **Hour features**: Temporal context (sin/cos encodings)
- Cross-group attention mechanism for inter-group dependencies
- Separate patch embeddings per channel group
- Physics-aware feature engineering

**Training Configuration:**
- Sequence length: 336 hours (14 days)
- Prediction length: 336 hours (14 days)
- Batch size: 16
- Learning rate: 0.001
- Optimizer: AdamW (weight_decay=1e-4)
- Scheduler: OneCycleLR
- Patience: 3 epochs (early stopping)
- Data split: **100% for training** (train+val+test combined), subset for validation
- Random seed: 2021

**Channel Groups:**

| Group | Source Features | Target Features | Purpose |
|-------|----------------|-----------------|---------|
| **Long-term** | p, T, Tdew, rh, wv, max. wv, wd, Tlog, hour_sin, hour_cos | p, T, wv, max. wv | Atmospheric dynamics |
| **Short-term** | rain, raining, hour_sin, hour_cos | rain, raining | Precipitation events |

**Strengths:**
- Incorporates domain knowledge
- Explicit modeling of feature groups with different temporal dynamics
- Cross-group attention captures dependencies between atmospheric and precipitation variables
- Hour-of-day features for diurnal patterns

**Limitations:**
- More complex architecture → higher risk of overfitting
- Requires careful hyperparameter tuning
- Channel grouping assumptions may not always hold
- Longer training time

---

## 2. Experimental Results

### 2.1 Test Set Performance (Target Variables Only)

Both models were evaluated on **6 target weather variables**:
1. **p (mbar)** - Air pressure
2. **T (degC)** - Temperature
3. **wv (m/s)** - Wind speed
4. **max. wv (m/s)** - Maximum wind speed
5. **rain (mm)** - Rainfall amount
6. **raining (s)** - Rainfall duration

#### Detailed Metrics Comparison

| Method | MSE | MAE | RMSE | Test Samples |
|--------|-----|-----|------|--------------|
| **DLinear (Baseline)** | **0.4687** | **0.3518** | **0.6846** | 10,192 |
| **Physics-Integrated PatchTST** | 0.5181 | 0.4269 | 0.7198 | 10,176 |
| **Δ (Absolute)** | +0.0494 | +0.0751 | +0.0352 | -16 |
| **Δ (Relative %)** | +10.5% | +21.3% | +5.1% | -0.16% |

**Notes:**
- Lower values are better for all metrics
- Physics-Integrated model shows **higher error** on both MSE and MAE
- Test sample difference (16 samples) is negligible and likely due to data loading configuration

---

### 2.2 All Features Performance (DLinear Only)

DLinear was also evaluated on **all 21 features** (not just targets):

| Metric | Value |
|--------|-------|
| MSE | 0.2645 |
| MAE | 0.3177 |

This shows that when evaluating on all features, the overall error is lower because:
- The 15 non-target features may have smaller prediction errors
- Averaging across more features dilutes the impact of harder-to-predict target variables

---

## 3. Analysis & Interpretation

### 3.1 Why is DLinear Outperforming Physics-Integrated PatchTST?

#### **1. Training Data Configuration**
- **DLinear**: Uses standard 70/10/20 split (train on 70% only)
- **Physics-Integrated**: Uses **100% of data for training** (train+val+test combined)
  
**Impact:** Despite using more training data, Physics-Integrated shows worse generalization. This suggests:
- Potential **overfitting** to the training distribution
- The model may be memorizing patterns rather than learning generalizable relationships
- Validation on a subset doesn't reflect true held-out performance

**Recommendation:** Revert to standard 70/10/20 split for fair comparison.

---

#### **2. Model Complexity vs Data Efficiency**
- **DLinear**: Simple linear model → harder to overfit, efficient learning
- **Physics-Integrated**: Complex transformer with attention → requires more data and careful regularization

**Impact:**
- More parameters don't always mean better performance
- Transformer architectures need extensive tuning (dropout, attention dropout, layer norm, etc.)
- Linear models can be surprisingly effective for time series with clear trends

**Recommendation:** 
- Add dropout layers (fc_dropout, head_dropout, attention_dropout)
- Experiment with smaller model capacity
- Consider gradual unfreezing or pre-training strategies

---

#### **3. Channel Grouping Assumptions**
The Physics-Integrated model assumes:
- Pressure, temperature, wind → "long-term" dynamics
- Rain, raining → "short-term" dynamics

**Potential Issues:**
- Real weather phenomena don't always follow these rigid boundaries
- Wind speed (wv, max. wv) can change rapidly during storms
- Temperature can shift quickly with frontal systems
- Separating channel groups may **lose important cross-correlations**

**Example:** 
- Sudden pressure drops strongly correlate with precipitation
- If these are in different groups with limited interaction, the model misses this crucial relationship

**Recommendation:**
- Experiment with different grouping strategies
- Try unified encoding (no groups) with cross-channel attention
- Use attention analysis to validate if cross-group attention is learning useful patterns

---

#### **4. Loss Function and Target Selection**
Both models compute MSE loss on the **same 6 target variables** during training.

**Current Setup:**
```python
# Physics-Integrated loss calculation
loss = criterion(pred_subset, true_subset)  # Only on target_indices
```

**Consideration:**
- Only supervising on 6 variables may under-utilize the 21 input features
- The model learns to predict 21 features but only receives feedback on 6

**Recommendation:**
- Try multi-task loss: supervise all 21 features + weighted loss on targets
- Add auxiliary losses (e.g., trend consistency, physical constraints)

---

#### **5. Hyperparameter Tuning**
The Physics-Integrated model uses:
- Learning rate: 0.001
- Batch size: 16
- Patience: 3 (very aggressive early stopping)

**Impact:**
- Patience=3 may stop training too early
- Model might not converge to optimal solution
- Complex models typically need more epochs

**Recommendation:**
- Increase patience to 5-10 epochs
- Try learning rate scheduling (ReduceLROnPlateau)
- Experiment with batch sizes (32, 64)
- Monitor training curves to detect underfitting

---

### 3.2 What's Working Well?

Despite underperforming the baseline, the Physics-Integrated approach has promising aspects:

1. **Domain Knowledge Integration**: The channel grouping concept is theoretically sound
2. **Modular Design**: Separate encoders for different feature groups enable specialized processing
3. **Cross-Group Attention**: Mechanism exists to capture inter-group dependencies
4. **Temporal Features**: Hour-of-day encoding adds useful inductive bias

---

### 3.3 Prediction Quality Analysis

Based on the error magnitudes:

| Model | MSE | Interpretation |
|-------|-----|----------------|
| DLinear | 0.4687 | Average squared error of ~0.68σ per timestep per feature |
| Physics-Int | 0.5181 | Average squared error of ~0.72σ per timestep per feature |

**Context:**
- Predictions are in **normalized scale** (mean=0, std=1)
- MSE ~0.47-0.52 means typical errors are 0.68-0.72 standard deviations
- For 336-hour horizon, this represents reasonable but not exceptional performance
- The 10-21% MAE increase in Physics-Integrated is significant and actionable

---

## 4. Recommendations for Improvement

### 4.1 Immediate Actions (High Priority)

1. **Fix Data Leakage**
   - Revert to 70/10/20 split (don't train on test data)
   - Use proper held-out test set for final evaluation
   - Current results may be artificially inflated

2. **Increase Early Stopping Patience**
   - Change patience from 3 to 10 epochs
   - Add learning rate reduction on plateau
   - Monitor validation loss trends more carefully

3. **Add Regularization**
   ```python
   args.dropout = 0.2
   args.fc_dropout = 0.2
   args.head_dropout = 0.1
   args.attn_dropout = 0.1
   ```

4. **Baseline Experiments**
   - Test Physics-Integrated with **no grouping** (unified encoder)
   - Compare against simple cross-channel attention without groups
   - Isolate the impact of channel grouping vs attention mechanisms

---

### 4.2 Medium-Term Experiments

1. **Architecture Ablation Studies**
   - Test without cross-group attention
   - Test with cross-channel attention after flattening (as discussed)
   - Compare per-channel vs cross-channel encoders

2. **Hyperparameter Optimization**
   - Grid search: learning_rate × batch_size × d_model
   - Try different patch lengths (16, 32, 64)
   - Experiment with number of encoder layers

3. **Feature Engineering**
   - Add lagged features (t-1, t-2, t-3)
   - Include derived features (pressure tendency, temperature rate of change)
   - Experiment with different hour encodings (cyclic vs learned embeddings)

4. **Loss Function Variations**
   - Weighted MSE (higher weights on harder-to-predict features like rain)
   - Huber loss (robust to outliers in precipitation)
   - Multi-horizon loss (different weights for near vs far predictions)

---

### 4.3 Long-Term Research Directions

1. **Physics-Informed Constraints**
   - Add soft constraints: pressure-wind relationships, thermodynamic bounds
   - Hybrid model: neural network + differential equations
   - Learn residuals from physics-based baseline

2. **Attention Analysis**
   - Visualize attention patterns to validate physical interpretability
   - Check if cross-group attention learns expected relationships
   - Use attention rollout to understand prediction reasoning

3. **Ensemble Methods**
   - Combine DLinear + Physics-Integrated predictions
   - Train multiple Physics-Integrated models with different groupings
   - Stacking: use DLinear predictions as input features

4. **Transfer Learning**
   - Pre-train on related weather datasets
   - Fine-tune on specific regions or seasons
   - Multi-task learning across different forecast horizons

---

## 5. Detailed Comparison Table

### Model Specifications

| Aspect | DLinear (Baseline) | Physics-Integrated PatchTST |
|--------|-------------------|----------------------------|
| **Architecture** | Linear decomposition | Patch-based Transformer |
| **Channel Handling** | Independent channels | Grouped channels (long/short) |
| **Attention Mechanism** | None | Cross-group attention |
| **Parameters** | ~500K-1M (estimated) | Not specified (likely >5M) |
| **Input Features** | 21 features | 21 features + 2 hour features |
| **Output Features** | 21 features | 21 features |
| **Training Focus** | All features | 6 target features |
| **Sequence Length** | 336 hours | 336 hours |
| **Prediction Horizon** | 336 hours | 336 hours |
| **Batch Size** | Default (~32-64) | 16 |
| **Learning Rate** | Default | 0.001 |
| **Optimizer** | AdamW | AdamW (weight_decay=1e-4) |
| **Scheduler** | OneCycleLR | OneCycleLR |
| **Data Split** | 70/10/20 | 100% train (problematic) |
| **Early Stopping** | Not specified | Patience=3 |
| **Regularization** | Minimal | Minimal (needs improvement) |

---

### Performance Comparison

| Metric Category | DLinear | Physics-Int | Winner | Gap |
|----------------|---------|-------------|--------|-----|
| **MSE (Target)** | 0.4687 | 0.5181 | DLinear | 10.5% better |
| **MAE (Target)** | 0.3518 | 0.4269 | DLinear | 21.3% better |
| **RMSE (Target)** | 0.6846 | 0.7198 | DLinear | 5.1% better |
| **Training Speed** | Fast | Slower | DLinear | - |
| **Inference Speed** | Fast | Slower | DLinear | - |
| **Interpretability** | Medium | Higher | Physics-Int | Domain structure |
| **Flexibility** | Low | High | Physics-Int | Configurable groups |

---

## 6. Statistical Significance

### Error Magnitude Context

**Mean Absolute Error (MAE) Interpretation:**
- DLinear: 0.352σ per prediction
- Physics-Int: 0.427σ per prediction

For a typical weather variable (e.g., temperature with σ=8°C):
- DLinear: ±2.8°C average error
- Physics-Int: ±3.4°C average error

**Difference:** 0.6°C average error increase → **practically significant** for weather forecasting

---

### Prediction Horizon Impact

Both models predict 336 hours (14 days) ahead:
- **Days 1-3**: Both models likely perform well
- **Days 4-7**: Error accumulation begins
- **Days 8-14**: Chaotic weather dynamics → increasing uncertainty

**Recommendation:** Analyze per-timestep errors to identify where Physics-Integrated diverges from DLinear.

---

## 7. Visualization Insights

Both notebooks include prediction visualizations for the 6 target features. Key observations:

### Sample Prediction (seed=2021, sample from test set)

**Features Visualized:**
1. p (mbar) - Pressure
2. T (degC) - Temperature
3. wv (m/s) - Wind speed
4. max. wv (m/s) - Max wind speed
5. rain (mm) - Rainfall
6. raining (s) - Rain duration

**Common Patterns:**
- Both models capture **smooth trends** well (pressure, temperature)
- Both struggle with **sudden events** (rain spikes, wind gusts)
- Ground truth shows clear diurnal cycles (captured better by models with hour features)

**Differences:**
- DLinear: Smoother predictions, conservative estimates
- Physics-Int: More variable predictions, attempts to capture finer details (but introduces more error)

---

## 8. Conclusions

### Current State

1. **DLinear (baseline) is the better model** given current configurations
2. Physics-Integrated PatchTST shows promise but requires significant tuning
3. The gap is **substantial** (10-21% worse) but not insurmountable

### Root Causes of Underperformance

1. **Data leakage** (training on 100% of data)
2. **Insufficient regularization** (high-capacity model without dropout)
3. **Early stopping too aggressive** (patience=3)
4. **Channel grouping may be suboptimal** (unclear if it helps or hurts)
5. **Hyperparameters not tuned** (learning rate, batch size, model size)

### Path Forward

**Short-term** (1-2 weeks):
- Fix data split (70/10/20)
- Add dropout regularization
- Increase patience
- Run ablation: no grouping vs grouping

**Medium-term** (1-2 months):
- Hyperparameter search
- Architecture variants
- Attention analysis
- Per-feature and per-timestep error analysis

**Long-term** (3+ months):
- Physics-informed constraints
- Ensemble methods
- Transfer learning
- Production deployment

---

## 9. Next Steps

### Experiment Queue (Prioritized)

1. **[CRITICAL] Fix data split** → Re-run Physics-Int with 70/10/20 split
2. **[HIGH] Add regularization** → dropout=0.2, fc_dropout=0.2, attn_dropout=0.1
3. **[HIGH] Increase patience** → Change from 3 to 10 epochs
4. **[MEDIUM] Ablation study** → Test unified cross-channel encoder (no grouping)
5. **[MEDIUM] Hyperparameter tuning** → Grid search learning rate and batch size
6. **[LOW] Architecture exploration** → Move cross-channel attention after flattening
7. **[LOW] Feature engineering** → Add more temporal features (day-of-week, month)
8. **[LOW] Analysis** → Per-feature error breakdown, attention visualization

---

## Appendix A: Raw Experimental Data

### DLinear Test Results

```
TEST SET EVALUATION RESULTS (Target Features Only)
============================================================
MSE:  0.468708
MAE:  0.351838
============================================================
Predictions shape: (10192, 336, 6)
Ground truth shape: (10192, 336, 6)
```

### Physics-Integrated PatchTST Test Results

```
TEST SET EVALUATION RESULTS (Target Variables Only)
============================================================
MSE:  0.518092
MAE:  0.426897
============================================================
Predictions shape: (10176, 336, 6)
Ground truth shape: (10176, 336, 6)
Target variables: p (mbar), rain (mm), T (degC), max. wv (m/s), raining (s), wv (m/s)
```

### Configuration Details

**DLinear:**
- Model: DLinear
- Sequence length: 336
- Prediction length: 336
- Label length: 48
- Features: Multivariate (M)
- Data: weather.csv

**Physics-Integrated:**
- Model: PhysicsIntegratedPatchTST
- Sequence length: 336
- Prediction length: 336
- Batch size: 16
- Learning rate: 0.001
- Channel groups: long_term (10 features → 4 targets), short_term (4 features → 2 targets)
- Cross-group attention: Enabled
- Cross-channel encoder: Disabled
- Hour features: Added (sin/cos encoding)
- Data: weather_with_hour.csv

---

## Appendix B: Target Variable Definitions

| Variable | Unit | Description | Expected Range |
|----------|------|-------------|----------------|
| **p** | mbar | Atmospheric pressure | 980-1040 mbar |
| **T** | °C | Air temperature | -25 to +40°C |
| **wv** | m/s | Wind velocity | 0-15 m/s |
| **max. wv** | m/s | Maximum wind velocity | 0-25 m/s |
| **rain** | mm | Rainfall amount | 0-50 mm/hour |
| **raining** | s | Duration of rainfall in hour | 0-3600 s |

---

**Document Version:** 1.0  
**Last Updated:** January 26, 2026  
**Author:** AI Analysis System  
**Status:** Draft for Review
