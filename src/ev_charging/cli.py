from __future__ import annotations

import argparse
import copy
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from ev_charging.baselines import fit_baselines, permutation_importance_for_model
from ev_charging.config import load_config
from ev_charging.evaluate import (
    append_metrics_csv,
    plot_pred_vs_actual,
    plot_residual_hist,
    regression_metrics,
)
from ev_charging.extra_data import describe_extra_datasets
from ev_charging.features import FeaturePreprocessor
from ev_charging.pipeline import build_train_test_arrays
from ev_charging.train_mlp import (
    load_mlp_bundle,
    mlp_test_metrics,
    predict_mlp,
    save_mlp_bundle,
    train_mlp,
)
from ev_charging.uncertainty import residual_quantile_intervals


def _project_root(args: argparse.Namespace) -> Path:
    return Path(args.root).resolve()


def cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    root = _project_root(args)
    out = cfg["outputs"]

    (
        pre,
        X_train_full,
        y_train_full,
        X_test,
        y_test,
        X_mlp_train,
        y_mlp_train,
        X_val,
        y_val,
        _test_df,
    ) = build_train_test_arrays(cfg, root)

    tr_state = int(cfg["training"]["random_state"])
    metrics_rows: list[dict] = []

    # --- sklearn CV (train matrix only) ---
    cv_cfg = cfg.get("cross_validation", {})
    if cv_cfg.get("enabled", False):
        folds = int(cv_cfg.get("folds", 5))
        kf = KFold(n_splits=folds, shuffle=True, random_state=tr_state)
        for name, est in [
            ("cv_linear_regression", LinearRegression()),
            (
                "cv_hist_gradient_boosting",
                HistGradientBoostingRegressor(
                    max_depth=6,
                    learning_rate=0.06,
                    max_iter=200,
                    random_state=tr_state,
                ),
            ),
        ]:
            scores = cross_val_score(
                est,
                X_train_full,
                y_train_full,
                cv=kf,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )
            rmse = float(np.sqrt(-scores.mean()))
            metrics_rows.append(
                {
                    "model": name,
                    "cv_rmse_mean": rmse,
                    "cv_rmse_std": float(np.sqrt(scores.std() if scores.std() > 0 else 0)),
                    "folds": folds,
                }
            )
            print(f"{name}: CV RMSE mean={rmse:.4f} (neg_mse scores: {-scores.mean():.2f} ± {scores.std():.2f})")

    # --- baselines (full train) ---
    bundle = fit_baselines(
        X_train_full,
        y_train_full,
        X_test,
        y_test,
        random_state=tr_state,
    )
    for name, m in bundle.metrics_test.items():
        print(f"{name} test: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))
        row = {"model": name, **m}
        metrics_rows.append(row)

    mlp_cfg = dict(cfg["mlp"])
    hs_cfg = cfg.get("hyperparam_search", {})

    if hs_cfg.get("enabled"):
        grid_lr = hs_cfg.get("grid", {}).get("lr", [mlp_cfg["lr"]])
        grid_h1 = hs_cfg.get("grid", {}).get("hidden1", [mlp_cfg["hidden1"]])
        quick = int(hs_cfg.get("quick_epochs", 400))
        vf = float(hs_cfg.get("val_fraction", 0.2))
        sub_tr, sub_va = train_test_split(
            np.arange(len(X_mlp_train)),
            test_size=vf,
            random_state=tr_state,
        )
        Xq_tr, yq_tr = X_mlp_train[sub_tr], y_mlp_train[sub_tr]
        Xq_va, yq_va = X_mlp_train[sub_va], y_mlp_train[sub_va]
        best_mse = float("inf")
        best_combo = (mlp_cfg["lr"], mlp_cfg["hidden1"])
        for lr, h1 in itertools.product(grid_lr, grid_h1):
            model, _ = train_mlp(
                Xq_tr,
                yq_tr,
                Xq_va,
                yq_va,
                epochs=quick,
                lr=float(lr),
                hidden1=int(h1),
                hidden2=int(mlp_cfg["hidden2"]),
                weight_decay=float(mlp_cfg.get("weight_decay", 0.0)),
                torch_seed=int(mlp_cfg.get("torch_seed", 42)),
                early_stopping_patience=max(50, quick // 5),
            )
            pred_va = predict_mlp(model, Xq_va)
            mse = regression_metrics(yq_va, pred_va)["mse"]
            print(f"hyperparam try lr={lr} hidden1={h1} quick_val_mse={mse:.4f}")
            if mse < best_mse:
                best_mse = mse
                best_combo = (float(lr), int(h1))
        mlp_cfg["lr"], mlp_cfg["hidden1"] = best_combo
        print(f"selected hyperparams: lr={best_combo[0]} hidden1={best_combo[1]}")

    patience = int(mlp_cfg.get("early_stopping_patience", 0))
    model, train_meta = train_mlp(
        X_mlp_train,
        y_mlp_train,
        X_val,
        y_val,
        epochs=int(mlp_cfg["epochs"]),
        lr=float(mlp_cfg["lr"]),
        hidden1=int(mlp_cfg["hidden1"]),
        hidden2=int(mlp_cfg["hidden2"]),
        weight_decay=float(mlp_cfg.get("weight_decay", 0.0)),
        torch_seed=int(mlp_cfg.get("torch_seed", 42)),
        early_stopping_patience=patience,
    )

    mlp_metrics = mlp_test_metrics(model, X_test, y_test)
    print("mlp test: " + ", ".join(f"{k}={v:.4f}" for k, v in mlp_metrics.items()))
    metrics_rows.append({"model": "mlp", **mlp_metrics})

    # Residual-based intervals (validation set)
    eval_cfg = cfg.get("evaluation", {})
    ri_cfg = eval_cfg.get("residual_intervals", {})
    if ri_cfg.get("enabled", False) and X_val is not None and y_val is not None:
        val_pred = predict_mlp(model, X_val)
        test_pred = predict_mlp(model, X_test)
        lo, hi = residual_quantile_intervals(
            y_val,
            val_pred,
            test_pred,
            q_low=float(ri_cfg.get("q_low", 0.05)),
            q_high=float(ri_cfg.get("q_high", 0.95)),
        )
        coverage = float(np.mean((y_test >= lo) & (y_test <= hi)))
        print(f"residual interval empirical test coverage (nominal ~90%): {coverage:.3f}")
        metrics_rows.append({"model": "mlp_interval_coverage", "coverage": coverage})

    plot_dir = Path(out["plots_dir"])
    if eval_cfg.get("plot", True):
        plot_pred_vs_actual(y_test, predict_mlp(model, X_test), plot_dir / "mlp_pred_vs_actual.png", title="MLP")
        plot_residual_hist(y_test, predict_mlp(model, X_test), plot_dir / "mlp_residuals.png")
        hgb = bundle.models["hist_gradient_boosting"]
        plot_pred_vs_actual(
            y_test,
            hgb.predict(X_test),
            plot_dir / "hgb_pred_vs_actual.png",
            title="HistGradientBoosting",
        )

    n_rep = int(eval_cfg.get("permutation_importance_repeats", 0))
    if n_rep > 0 and pre.feature_names_:
        hgb = bundle.models["hist_gradient_boosting"]
        names = pre.feature_names_
        imp = permutation_importance_for_model(
            hgb,
            X_test,
            y_test,
            names,
            n_repeats=n_rep,
            random_state=tr_state,
        )
        print("permutation importance (hist_gradient_boosting, top 15):")
        for n, v in imp[:15]:
            print(f"  {n}: {v:.5f}")

    Path(out["model_path"]).parent.mkdir(parents=True, exist_ok=True)
    save_mlp_bundle(
        root / out["model_path"],
        model,
        hidden1=int(mlp_cfg["hidden1"]),
        hidden2=int(mlp_cfg["hidden2"]),
        n_features=X_train_full.shape[1],
        torch_seed=int(mlp_cfg.get("torch_seed", 42)),
        train_meta=train_meta,
    )
    pre.save(root / out["preprocessor_path"])
    bundle.save(root / out["baselines_path"])

    if out.get("log_runs") and out.get("metrics_csv"):
        for row in metrics_rows:
            append_metrics_csv(root / out["metrics_csv"], row)

    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    root = _project_root(args)
    out = cfg["outputs"]
    pre = FeaturePreprocessor.load(root / out["preprocessor_path"])
    model, _payload = load_mlp_bundle(root / out["model_path"])

    sep = args.sep if args.sep is not None else ";"

    df = pd.read_csv(args.input, sep=sep)
    if "target" not in df.columns and "El_kWh" in df.columns:
        tcfg = cfg["training"]["target"]
        if tcfg == "El_kWh":
            df = df.copy()
            df["target"] = pd.to_numeric(
                df["El_kWh"].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
        else:
            dur = pd.to_numeric(
                df["Duration_hours"].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
            el = pd.to_numeric(
                df["El_kWh"].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
            df = df.copy()
            df["target"] = el / dur.replace(0, np.nan)

    X, _y = pre.transform(df)
    pred = predict_mlp(model, X)
    out_df = df.copy()
    out_df["predicted_kWh"] = pred
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(pred)} rows)")
    return 0


def cmd_ablation(args: argparse.Namespace) -> int:
    """Compare sklearn models with traffic features on vs off (same split, refit preprocessor)."""
    cfg = load_config(args.config)
    root = _project_root(args)
    base = cfg["training"].get("use_traffic", True)

    results = []
    for use_traffic in [True, False] if args.compare_traffic else [base]:
        cfg_ab = copy.deepcopy(cfg)
        cfg_ab["training"]["use_traffic"] = use_traffic
        (
            pre,
            X_train_full,
            y_train_full,
            X_test,
            y_test,
            _xm,
            _ym,
            _xv,
            _yv,
            _,
        ) = build_train_test_arrays(cfg_ab, root)
        bundle = fit_baselines(
            X_train_full,
            y_train_full,
            X_test,
            y_test,
            random_state=int(cfg_ab["training"]["random_state"]),
        )
        row = {
            "use_traffic": use_traffic,
            "hgb_rmse": bundle.metrics_test["hist_gradient_boosting"]["rmse"],
            "lr_rmse": bundle.metrics_test["linear_regression"]["rmse"],
        }
        results.append(row)
        print(row)

    if len(results) == 2:
        on = next(r for r in results if r["use_traffic"])
        off = next(r for r in results if not r["use_traffic"])
        print(
            f"HistGB RMSE: with traffic={on['hgb_rmse']:.4f}, "
            f"without={off['hgb_rmse']:.4f} (lower is better)"
        )
    return 0


def cmd_datasets_info(_args: argparse.Namespace) -> int:
    print(describe_extra_datasets())
    return 0


def cmd_cv_only(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    root = _project_root(args)
    (
        _pre,
        X_train_full,
        y_train_full,
        _xt,
        _yt,
        _xm,
        _ym,
        _xv,
        _yv,
        _,
    ) = build_train_test_arrays(cfg, root)
    tr_state = int(cfg["training"]["random_state"])
    folds = int(args.folds or cfg.get("cross_validation", {}).get("folds", 5))
    kf = KFold(n_splits=folds, shuffle=True, random_state=tr_state)
    for name, est in [
        ("linear_regression", LinearRegression()),
        (
            "hist_gradient_boosting",
            HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.06,
                max_iter=200,
                random_state=tr_state,
            ),
        ),
    ]:
        scores = cross_val_score(
            est,
            X_train_full,
            y_train_full,
            cv=kf,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        print(f"{name}: RMSE={float(np.sqrt(-scores.mean())):.4f} ± {float(scores.std()):.4f}")
    return 0


def _add_root(ap: argparse.ArgumentParser) -> None:
    """Per-subcommand so `train ... --root .` works (global flags before subcommands are easy to get wrong)."""
    ap.add_argument(
        "--root",
        default=".",
        help="Project root directory (contains datasets/, configs/, etc.)",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ev-charging", description="EV charging load prediction pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Train baselines + MLP and write artifacts")
    _add_root(t)
    t.add_argument("--config", default="configs/default.yaml")
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="Batch predict from a feature CSV")
    _add_root(pr)
    pr.add_argument("--config", default="configs/default.yaml")
    pr.add_argument("--input", required=True)
    pr.add_argument("--output", required=True)
    pr.add_argument("--sep", default=None, help="CSV delimiter (default: comma, or ; for .csv heuristic)")
    pr.set_defaults(func=cmd_predict)

    ab = sub.add_parser("ablation", help="Traffic on/off comparison for sklearn baselines")
    _add_root(ab)
    ab.add_argument("--config", default="configs/default.yaml")
    ab.add_argument(
        "--compare-traffic",
        action="store_true",
        help="Run both use_traffic true and false",
    )
    ab.set_defaults(func=cmd_ablation)

    ds = sub.add_parser("datasets-info", help="Describe optional datasets")
    _add_root(ds)
    ds.set_defaults(func=cmd_datasets_info)

    cv = sub.add_parser("cv", help="Run cross-validation only (sklearn)")
    _add_root(cv)
    cv.add_argument("--config", default="configs/default.yaml")
    cv.add_argument("--folds", type=int, default=None)
    cv.set_defaults(func=cmd_cv_only)

    return p


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
