# Proposal Defence Round 1

## 1) Benefits of the New Architecture for Weather Data
- **Channel grouping by temporal behavior:** Separates long-horizon variables (e.g., temperature, pressure) from rapid-change variables (e.g., rain bursts, gusts), letting each path learn dynamics on the scale that matters.
- **Variable-length patching:** Longer patches (stride/overlap) capture smooth diurnal trends; shorter patches capture spikes and on/off events. Reduces aliasing and preserves sharp transitions.
- **Per-channel attention:** Each variable gets its own attention stack, avoiding cross-variable interference and letting the model specialize per sensor/variable.
- **Cross-group fusion:** A dedicated cross-group attention merges slow and fast signals, improving phenomena where fast events modulate slow trends (e.g., rain cooling temperature, gusts following pressure drops).
- **Hour-of-day features integrated:** Embeds daily periodicity explicitly, improving phase alignment for diurnal cycles across variables.
- **Target-selective heads:** Heads exist only for weather variables (hour features are input-only), focusing capacity on forecasted outputs.
- **RevIN for stability:** Instance-wise normalization reduces distribution shift across time and stations, improving robustness.
- **Physics-aware design:** The grouping and fusion reflect physical couplings (e.g., humidity → rain; temperature ↔ pressure ↔ wind), leading to more interpretable inductive biases.

## 2) Potential Q&A (20+)
1. **Why group channels into long vs short?** To match model receptive fields to each variable’s characteristic timescale, reducing over/under-smoothing.
2. **How were patch lengths chosen?** Long path uses larger, overlapping patches for smooth trends; short path uses shorter, non-overlapping patches to preserve spikes.
3. **Why per-channel attention instead of shared?** Avoids cross-variable interference and lets each variable learn its own temporal patterns.
4. **How does cross-group attention help?** It exchanges information between slow and fast variables, capturing causal links (e.g., rapid rain onset cooling temperature).
5. **Do hour-of-day features matter if the model is powerful?** Yes—explicit cyclical encoding stabilizes learning and improves phase alignment for diurnal patterns.
6. **What targets are optimized?** Only the weather outputs (not the hour features); heads are aligned to the target indices per group.
7. **How does this differ from plain PatchTST?** Adds physics-based grouping, variable patch sizes, per-channel attention, and cross-group fusion; PatchTST treats all channels uniformly.
8. **Why keep RevIN?** It reduces non-stationarity and covariate shift across time/locations, improving generalization.
9. **Could cross-group attention introduce noise?** It can; we blend with a learnable weight (sigmoid of a parameter) so the model can down-weight if harmful.
10. **How do we prevent overfitting with more modules?** Dropout, weight decay (AdamW), OneCycleLR, and early stopping; per-channel stacks are lightweight.
11. **What about computational cost?** Cost grows linearly with channel count per-group; splitting into two groups keeps attention manageable and parallelizable.
12. **How are targets mapped when channels are reordered?** The config maps names → indices; target_indices are derived from channel_groups to keep alignment consistent.
13. **Does grouping hurt variables that mix behaviors?** Potentially; cross-group attention mitigates by letting mixed-behavior variables receive fused context.
14. **How is robustness to outliers handled?** RevIN plus per-channel processing limits cross-variable contamination; optional clipping/constraints can be applied at evaluation.
15. **Do hour features leak future info?** No—hour_sin/hour_cos are deterministic from timestamps available at prediction time.
16. **Can we extend to spatial data?** Yes, by adding a spatial encoder or graph module before channel grouping.
17. **How to handle missing channels?** Masking at the patch level or simple imputation; per-channel encoders reduce cross-channel dependency on missing values.
18. **What loss is used?** MSE on target channels; can be extended with MAE/Huber for robustness.
19. **How does the model behave under covariate shift (seasonal change)?** RevIN plus hour features help; retraining or fine-tuning on recent data is recommended.
20. **Why not a single multi-scale encoder instead of two paths?** Two explicit paths simplify inductive bias, make tuning patch sizes easier, and isolate computation per regime.
21. **How do we validate gains?** Compare against baseline PatchTST on standard splits; ablate grouping, cross-group attention, and hour features.

## 3) Possible Limitations / Pitfalls
- **Group assignment sensitivity:** Mis-grouped variables (e.g., a mid-variability channel) may underperform; requires domain-informed grouping.
- **Extra complexity:** More components (two encoders + cross-group) increase tuning surface and potential for configuration errors.
- **Cross-group noise:** If correlations are weak or non-stationary, cross-group attention may add noise; blending mitigates but not eliminates.
- **Compute/memory overhead:** Two parallel paths and per-channel attention cost more than a single shared encoder.
- **RevIN dependence:** If normalization stats drift or are poorly estimated, denorm can mis-scale outputs.
- **Limited spatial awareness:** Architecture is temporal-only; spatial correlations (stations, grids) are not modeled.
- **Data quality:** Sensor drift, missingness, and outliers still need upstream handling.
- **Evaluation mismatch:** If quantization or clipping is applied only at evaluation, training/eval distributions can diverge; keep preprocessing consistent.

## 4) Why Encoder-Only (vs. Decoder-Only like GPT)
- **Forecasting setup:** We predict future values from known past; an encoder-only temporal model naturally ingests the full context and outputs all horizons without autoregressive token-by-token decoding.
- **Parallel prediction:** Encoder heads produce the entire prediction window in parallel, unlike decoder-only autoregression which is sequential and slower.
- **Reduced exposure bias:** Non-autoregressive output avoids error accumulation step-to-step common in decoder-only generation.
- **Attention focus:** Encoder attention is bidirectional over the input window (past), which is all that’s needed for forecasting; decoder-only causal masks are unnecessary overhead here.
- **Inductive bias:** PatchTST-style encoders with patch embeddings and per-channel attention are tailored to time-series structure; GPT-style tokenization and causal masks are less aligned with continuous sensor data.
- **Efficiency:** No need for a separate encoder-decoder stack; simpler training and inference, better latency for rolling forecasts.
- **Determinism:** Weather forecasting values are continuous and often evaluated with regression losses; encoder-only models fit naturally without language-style sampling.
