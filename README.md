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

