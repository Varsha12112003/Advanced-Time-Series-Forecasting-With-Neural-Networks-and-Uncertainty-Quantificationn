# Technical Report — Advanced Time Series Forecasting with Neural Networks and Uncertainty Quantification

## Objective
Implement, train, and evaluate a deep learning model (e.g., LSTM / TCN / Transformer) for multi-step time series forecasting on complex datasets exhibiting non-stationarity, seasonality and noise. Emphasize probabilistic forecasting and uncertainty quantification (UQ).

## Dataset
- Use either a real-world dataset (financial, energy, or industrial monitoring) or a synthetic dataset that demonstrates non-stationarity and seasonality.
- Preprocessing steps: missing value handling, normalization (fit on train), rolling-window dataset creation for multi-step forecasting, train/validation/test splits with contiguous blocks to preserve temporal order.

## Model choices
- Baseline: ARIMA / Exponential Smoothing (use as benchmark with statsmodels).
- Neural model: LSTM encoder-decoder or a Transformer/TCN. This repo includes an LSTM + Monte Carlo Dropout skeleton (see `model.py`).

## Uncertainty Quantification (UQ)
Recommended UQ methods to implement and compare:
1. **Monte Carlo Dropout**: enable dropout at inference and sample multiple forward passes to approximate predictive distribution (epistemic + some aleatoric).
2. **Neural Quantile Regression**: output quantiles (e.g., 0.05, 0.5, 0.95) using pinball loss.
3. **Direct Heteroscedastic Output**: model predicts mean and variance and use a Gaussian negative log-likelihood loss.
4. **Ensembling**: train multiple models with different seeds/hyperparameters and combine predictions.

## Losses and Metrics
- Point forecast: MAE, RMSE.
- Probabilistic calibration: Prediction Interval Coverage Probability (PICP), Interval Width, Continuous Ranked Probability Score (CRPS).
- Sharpness and calibration plots; reliability diagrams.
- Evaluate across multiple horizons (1-step, multi-step up to H).

## Experimental protocol
- Hyperparameter search (learning rate, hidden size, dropout, sequence length).
- Walk-forward validation for time series.
- Compare against simple statistical baselines and report relative improvements.

## Implementation notes (in this repo)
- `model.py` contains an LSTM model with dropout layers that can be used with MC Dropout and an optional quantile head.
- `train_sample.py` demonstrates preparing a toy dataset, training the model, and producing multiple MC samples for intervals.

## Deliverables
- Code and notebooks to reproduce experiments.
- A textual report comparing models, UQ methods, and calibration results.
- Figures: forecast plots with intervals, calibration plots, error tables across horizons.

