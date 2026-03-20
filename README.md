# Predicting Residential EV Charging Loads

Predict electric vehicle charging energy (**kWh** per session) from Norwegian residential charging data, traffic counts, and optional hourly aggregate features. The project ships as an installable Python package with a **CLI**, **tests**, **CI**, optional **FastAPI** and **Streamlit** entry points, and YAML **configuration**.

**Supported Python:** 3.10–3.12 (see `.python-version` for the recommended 3.11).

## Repository layout

```
.
├── configs/default.yaml          # Paths, training, MLP, CV, outputs
├── datasets/                     # CSV inputs (; delimiter, European decimals)
├── docs/
│   ├── DATASETS.md               # What each file is for
│   └── FEATURES_AND_IMPROVEMENTS_PLAN.md
├── models/                       # Generated: mlp.pt, preprocessor.joblib, baselines.joblib (see .gitignore)
├── src/ev_charging/              # Library + CLI
├── tests/
├── predict_electric_vehicle_charging_loads.ipynb   # Original exploratory notebook
├── streamlit_app.py
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,api]"  # + streamlit already in dev extras
# CPU-only PyTorch (optional, recommended on Linux CI):
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Train (baselines + MLP + metrics + plots)

Put `--root` **after** the subcommand (`train`, `predict`, …). Example:

```bash
python -m ev_charging train --config configs/default.yaml --root .
```

Writes (by default):

- `models/mlp.pt` — PyTorch weights + architecture metadata  
- `models/preprocessor.joblib` — fitted encoders (train-only categories)  
- `models/baselines.joblib` — sklearn baselines (dummy mean/median, linear regression, histogram GB)  
- `artifacts/plots/` — predicted vs actual, residuals  
- `runs/metrics.csv` — appended run metrics when `outputs.log_runs` is true  

Training includes **k-fold CV** on sklearn models (configurable), optional **hyperparameter search** for the MLP on a quick validation schedule, **permutation importance** for gradient boosting, and **residual quantile intervals** on the test set when a validation split is enabled.

## Batch prediction

You need **trained artifacts** first (`python -m ev_charging train ...`), then a CSV in the **same shape as the training feature table** (after EV↔traffic merge, before one-hot encoding).

### Easiest: build input from the bundled datasets

This uses `configs/default.yaml` paths (`datasets/EV charging reports.csv` + traffic, and optional hourly merge if enabled in config):

```bash
python -m ev_charging build-predict-input --root . --output rows_to_score.csv --max-rows 100
python -m ev_charging predict --root . --input rows_to_score.csv --output scored.csv
```

- Include **`El_kWh`** in the input if you want actuals in the file for side‑by‑side comparison; the model still predicts from the other columns.
- Default delimiter for `predict` is **`;`** (matches the project CSVs). Use `--sep ,` for comma-separated files.

### Manual CSV

Required columns (when `use_traffic: true`): `User_type`, `Duration_hours`, `month_plugin`, `weekdays_plugin`, the five traffic count columns (`KROPPAN BRU`, …), plus any `hourly_*` columns your preprocessor was fit with. Optional: `El_kWh` or `target` (for evaluation only).

## Other CLI commands

```bash
python -m ev_charging build-predict-input --root . --output sample.csv --max-rows 50
python -m ev_charging cv --config configs/default.yaml          # sklearn CV only
python -m ev_charging ablation --compare-traffic                # traffic on vs off (sklearn)
python -m ev_charging datasets-info                             # optional CSV descriptions
```

## API (FastAPI)

```bash
export EV_CHARGING_ROOT="$(pwd)"
uvicorn ev_charging.api:app --reload --port 8000
```

- `GET /health`  
- `POST /predict/csv` — multipart file upload; `sep` query param (default `;`)

## Streamlit

```bash
streamlit run streamlit_app.py
```

Set `EV_CHARGING_ROOT` if not running from the repo root.

## Docker

Build and run the API (train artifacts must exist or be bind-mounted into `/app/models`):

```bash
docker build -t ev-charging-api .
docker run -p 8000:8000 -v "$(pwd)/models:/app/models" ev-charging-api
```

## Configuration highlights

| Key | Purpose |
|-----|---------|
| `training.split_method` | `random` or `group_user` (GroupShuffleSplit on `User_ID`) |
| `training.target` | `El_kWh` or `El_kWh_per_hour` |
| `training.use_traffic` | Ablation: drop traffic count columns |
| `training.use_hourly_private_features` | Merge hourly private aggregates (set `data.hourly_private_csv`) |
| `mlp.val_fraction` | Holdout from train for early stopping / residual intervals |
| `hyperparam_search` | Small grid over learning rate and first hidden width |

## Development

```bash
python3 -m ruff check src tests
python3 -m pytest
```

## License

MIT — see `LICENSE`. **Datasets:** confirm you may redistribute or document external provenance (see `docs/DATASETS.md`).

## Original notebook results

| Model | Test MSE | √MSE (≈ avg error, kWh) |
|-------|----------|--------------------------|
| Linear regression | ~121 | ~11 |
| Neural network (notebook) | ~118 | ~10.9 |

The packaged pipeline reproduces the same modeling idea with stricter preprocessing (encoder fit on train only), extra baselines, and evaluation tooling.
