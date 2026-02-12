# FEMTO Feature Extraction (CNN / Dilated CNN AE) + Regression Prompt

## Dataset Context
- FEMTO / PRONOSTIA bearing dataset
- Natural degradation with very limited run-to-failure samples
- 17 runs total (11 train, 6 test)
- 3 operating settings (400–500 N, 1200–1800 rpm)
- Signals: vib_h, vib_v sampled at 25.6 kHz
- One cycle = 2560 samples (0.1 s), cycle interval = 10 s

## Hard Constraints (must not be violated)
- Feature extraction method: CNN-based autoencoder
  - standard CNN AE or dilated CNN AE
- Sensor columns: vib_h, vib_v
- Target column: rul
- RUL clipping: 200
- Resampling:
  - group_by: unit
  - keep_healthy_fraction ≈ 0.001
- Normalization:
  - normalize strictly per operational setting (per-os)
- Windowing:
  - by_cycle = true
  - fast = true
  - no window size or stride configuration
  - encode_os = false
- Use TSMA denoising with m = 2
- Use multi-windowing:
  - stack multiple latent windows before regression (search space between 10 and 30)
  - stack multipe windows for seq2val models (search space between 10 and 30)
- target_scaling: minmax or standard (target only)
- Drop os after preprocessing

## Key Challenges
- Extremely few degradation trajectories
- Strong operating-setting dependency
- High stochasticity in degradation patterns
- Tsfresh with XGBoost tends to create constant predictions

## Feature Extraction Strategy
- Dilated convolutions are well suited to increase receptive field
- Train AE only to reconstruction stability (not overfitting)

## Regression Strategy
- Input: stacked latent vectors from multi-windowing
- Preferred regressors:

  1) XGBoost
  2) TCN (seq2val or regression)
  3) Small MLP

## Hints
- Dialated cnn ae for feature extraction with regression with 
  xgboost (m=20) works  very well
- The window length is large (2560 samples) * m (multi window)
- Prefer depth over width (small channel counts, more layers if needed).

## Objective
- Preserve degradation variance across operating settings
- Avoid constant prediction behavior
- Prefer early predictions over late predictions
