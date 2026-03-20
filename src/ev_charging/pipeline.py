from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from ev_charging.data import (
    add_target_column,
    drop_incomplete_rows,
    load_ev_charging_reports,
    load_traffic_reports,
    merge_ev_traffic,
    prepare_merged_features_frame,
    validate_feature_frame,
)
from ev_charging.extra_data import hourly_private_for_merge, merge_hourly_private_features
from ev_charging.features import FeaturePreprocessor


def load_merged_dataframe(cfg: dict[str, Any], project_root: Path) -> pd.DataFrame:
    data = cfg["data"]
    ev_path = project_root / data["ev_charging_csv"]
    traffic_path = project_root / data["traffic_csv"]
    ev = load_ev_charging_reports(ev_path)
    traffic = load_traffic_reports(traffic_path)
    merged = merge_ev_traffic(ev, traffic)
    if cfg["training"].get("use_hourly_private_features"):
        hp = data.get("hourly_private_csv")
        if not hp:
            raise ValueError("use_hourly_private_features requires data.hourly_private_csv")
        hourly = hourly_private_for_merge(project_root / hp)
        merged = merge_hourly_private_features(merged, hourly)
    return merged


def prepared_frame(cfg: dict[str, Any], merged: pd.DataFrame) -> pd.DataFrame:
    tr = cfg["training"]
    df = prepare_merged_features_frame(
        merged,
        use_traffic=tr["use_traffic"],
        keep_split_meta=True,
    )
    df = add_target_column(df, tr["target"])
    df = drop_incomplete_rows(df)
    validate_feature_frame(df)
    return df


def split_indices(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    tr = cfg["training"]
    rs = int(tr["random_state"])
    test_size = float(tr["test_size"])
    method = tr.get("split_method", "random")
    idx = np.arange(len(df))
    if method == "group_user":
        if "User_ID" not in df.columns:
            raise ValueError("group_user split requires User_ID column")
        groups = df["User_ID"].values
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        train_idx, test_idx = next(gss.split(idx, groups=groups))
        return train_idx, test_idx
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=rs)
    return train_idx, test_idx


def train_val_split_indices(
    n: int,
    val_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    if val_fraction <= 0 or val_fraction >= 1:
        return idx, np.array([], dtype=int)
    sub_tr, sub_val = train_test_split(idx, test_size=val_fraction, random_state=random_state)
    return sub_tr, sub_val


def build_train_test_arrays(
    cfg: dict[str, Any],
    project_root: Path,
) -> tuple[
    FeaturePreprocessor,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    pd.DataFrame,
]:
    """Arrays for sklearn use full train; MLP uses train minus optional validation holdout."""
    merged = load_merged_dataframe(cfg, project_root)
    df = prepared_frame(cfg, merged)
    train_idx, test_idx = split_indices(df, cfg)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    tr = cfg["training"]
    pre = FeaturePreprocessor(use_traffic=tr["use_traffic"])
    pre.fit(train_df)
    X_train_full, y_train_full = pre.transform(train_df)
    X_test, y_test = pre.transform(test_df)
    assert y_train_full is not None and y_test is not None

    mlp_cfg = cfg.get("mlp", {})
    vf = float(mlp_cfg.get("val_fraction", 0.0))
    sub_tr, sub_val = train_val_split_indices(len(train_df), vf, int(tr["random_state"]))
    X_val = y_val = None
    if len(sub_val) > 0:
        X_val = X_train_full[sub_val]
        y_val = y_train_full[sub_val]

    X_mlp_train = X_train_full[sub_tr]
    y_mlp_train = y_train_full[sub_tr]

    return (
        pre,
        X_train_full,
        y_train_full,
        X_test,
        y_test,
        X_mlp_train,
        y_mlp_train,
        X_val,
        y_val,
        test_df,
    )
