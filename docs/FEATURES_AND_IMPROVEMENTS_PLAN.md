# Features and improvements plan

This document outlines possible next steps for the **Predicting Residential EV Charging Loads** repository. It is ordered roughly by impact vs. effort: hygiene and reproducibility first, then modeling rigor, then productization and research extensions.

---

## Current state (baseline)

- **Primary artifact:** `predict_electric_vehicle_charging_loads.ipynb` — loads EV session data and traffic counts, merges on plug-in hour, encodes categoricals, trains sklearn linear regression and a small PyTorch MLP, saves `models/model.pth`.
- **Dependencies:** `requirements.txt` lists packages without pinned versions.
- **Data in repo:** `EV charging reports.csv` and `Local traffic distribution.csv` are used in the notebook; additional CSVs (`AMS data from garage Bl2.csv`, `Hourly EV loads - *.csv`) are present but not wired into the pipeline.
- **Docs vs. tree:** `README.md` describes a nested `predict_electric_vehicle_charging_loads/` layout; the actual layout is flatter — worth aligning to avoid confusion.

---

## 1. Repository hygiene and reproducibility

| Item | Why it matters |
|------|----------------|
| **Pin dependencies** | Same `pip install` → same behavior across machines and CI. Use `requirements.txt` with `==` versions or migrate to `pyproject.toml` + lockfile (e.g. uv/poetry). |
| **Align README with reality** | Document actual paths, optional datasets, and how to obtain/regenerate data if anything is ever split out. |
| **Model artifacts policy** | Decide whether `models/model.pth` is committed for demos or gitignored and produced by a training script; document the choice. |
| **Python version** | State a single supported minor (e.g. 3.11) in README and optionally `.python-version` for local tools. |
| **LICENSE** | Add if the project is public or shared; clarify dataset redistribution if applicable. |

---

## 2. Data pipeline and documentation

| Feature / improvement | Notes |
|----------------------|--------|
| **Incorporate or document unused datasets** | Either add notebooks or sections for AMS / hourly aggregate loads, or explain why they are out of scope (privacy, different task, future work). |
| **Shared loading utilities** | Extract `read_csv` paths, Norwegian decimal comma handling, and merge logic into importable Python modules to avoid duplication if you add scripts or tests. |
| **Data validation** | Light checks (expected columns, dtypes, missingness, date ranges) at load time; fail fast with clear messages. |
| **Reproducible splits** | Fix `random_state` everywhere; consider stratifying or grouping (e.g. by `User_ID` or `Garage_ID`) if leakage across sessions is a concern. |

---

## 3. Modeling and evaluation

| Feature / improvement | Notes |
|----------------------|--------|
| **Stronger baselines** | Mean predictor, median, or dummy regressors; optionally gradient boosting (e.g. HistGradientBoosting) as a non-neural ceiling before investing in deep models. |
| **Cross-validation** | Replace or complement a single 80/20 split with k-fold or time-based CV if temporal ordering matters. |
| **Metrics suite** | MAE, MAPE (where defined), R², and calibration-style plots (predicted vs. actual); residuals by hour/month/user type. |
| **Hyperparameters** | Small search (learning rate, width/depth, weight decay) with a validation fold; log runs (CSV, MLflow, or Weights & Biases) if you scale up experiments. |
| **Uncertainty** | Quantile regression, ensembles, or MC dropout for prediction intervals — useful for grid/operations use cases. |

---

## 4. Engineering: from notebook to maintainable code

| Feature / improvement | Notes |
|----------------------|--------|
| **`src/` package** | `data.py`, `features.py`, `train.py`, `evaluate.py` — notebook becomes a thin narrative or EDA-only layer. |
| **CLI entry points** | e.g. `python -m ev_charging train --config configs/default.yaml` for batch training on a server. |
| **Tests** | Unit tests for parsing, encoding, and merge logic; smoke test that trains 1–2 steps on a tiny fixture CSV. |
| **CI** | GitHub Actions: lint (ruff), format (optional), run tests on push/PR; optional scheduled job if training is lightweight. |
| **Configuration** | YAML or TOML for paths, hyperparameters, and seeds instead of hardcoding in cells. |

---

## 5. Inference and “product” surfaces

| Feature / improvement | Notes |
|----------------------|--------|
| **Batch inference script** | Load `model.pth` + preprocessor, score a CSV of sessions, write predictions — minimal path to operational use. |
| **REST API (optional)** | FastAPI + Pydantic schema for one-session prediction; containerize for deployment. |
| **Streamlit / Gradio (optional)** | Quick UI for stakeholders to paste or upload a few rows and see predictions and feature values. |

---

## 6. Research and interpretability

| Feature / improvement | Notes |
|----------------------|--------|
| **Ablation studies** | Train with vs. without traffic features to quantify their contribution (README already hints at this). |
| **Feature importance** | Permutation importance for tree models; or gradient × input for the MLP on selected batches. |
| **Alternative targets** | e.g. duration-normalized energy, or classification buckets (low/medium/high) if that matches downstream decisions. |
| **Temporal models** | If hourly load series become first-class, consider sequence models or simple lag features from `Hourly EV loads` files. |

---

## Suggested sequencing

1. **Quick wins:** Fix README structure, pin requirements, fix random seeds and document split strategy.  
2. **Trust in numbers:** Add MAE/R², residual plots, and a stronger tree baseline.  
3. **Maintainability:** Extract Python modules + minimal tests + CI.  
4. **Stretch:** Use extra datasets or ship inference CLI/API depending on audience (research vs. operations).

This plan can be trimmed or reprioritized based on whether the goal is **publication-quality experiments**, **internal forecasting**, or **teaching/demos**.
