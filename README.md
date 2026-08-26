# Liquidity Stress Prediction

Reproducible competition pipeline for predicting liquidity stress in the 30 days after each customer snapshot.

## Setup

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv sync
```

## Data audit

Keep the supplied files in `Data/`, then run:

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv run python src/audit_data.py
```

Audit results are written to `artifacts/data_audit.json`. Competition data and generated submissions should not be committed or shared outside the registered team.

Run the target-signal and train/test-shift analysis with:

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv run python src/analyze_signals.py
```

## Baseline model

Run the fixed-seed, three-fold CatBoost baseline with:

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv run python src/train_baseline.py
```

The script records out-of-fold Log Loss and ROC-AUC in `artifacts/baseline_metrics.json` and writes the validated test predictions to `submissions/catboost_baseline.csv`.
