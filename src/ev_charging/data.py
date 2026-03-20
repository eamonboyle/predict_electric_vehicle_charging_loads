from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ev_charging.constants import (
    DROP_FOR_FEATURES,
    EV_REQUIRED_COLUMNS,
    META_COLUMNS_FOR_SPLIT,
    TRAFFIC_COUNT_COLUMNS,
    TRAFFIC_REQUIRED_COLUMNS,
)


def load_ev_charging_reports(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, sep=";", header=0, na_values=["NA"])
    _validate_columns(df, EV_REQUIRED_COLUMNS, "EV charging reports", path)
    return df


def load_traffic_reports(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, sep=";", header=0)
    _validate_columns(df, TRAFFIC_REQUIRED_COLUMNS, "Traffic", path)
    return df


def _validate_columns(df: pd.DataFrame, required: list[str], name: str, path: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} ({path}): missing columns {missing}. Found: {list(df.columns)}")


def merge_ev_traffic(ev: pd.DataFrame, traffic: pd.DataFrame) -> pd.DataFrame:
    """Align EV sessions with hourly traffic rows (same convention as the original notebook)."""
    ev = ev.copy()
    ev["Start_plugin_hour"] = ev["Start_plugin"].astype(str).str[:-2] + "00"
    merged = pd.merge(ev, traffic, left_on="Start_plugin_hour", right_on="Date_from", how="inner")
    if merged.empty:
        raise ValueError("Merge produced no rows; check date/hour alignment between EV and traffic.")
    return merged


def clean_european_decimals(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(",", ".", regex=False)


def coerce_numeric_traffic(df: pd.DataFrame) -> None:
    for col in TRAFFIC_COUNT_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")


def prepare_merged_features_frame(
    merged: pd.DataFrame,
    *,
    use_traffic: bool = True,
    keep_split_meta: bool = False,
) -> pd.DataFrame:
    """Drop ID/timestamp columns used only for joining; fix decimals; coerce traffic to float."""
    drop = list(DROP_FOR_FEATURES)
    if keep_split_meta:
        drop = [c for c in drop if c not in META_COLUMNS_FOR_SPLIT]
    df = merged.drop(columns=[c for c in drop if c in merged.columns], axis=1)
    if not use_traffic:
        df = df.drop(columns=[c for c in TRAFFIC_COUNT_COLUMNS if c in df.columns], axis=1)

    df = df.copy()
    df["El_kWh"] = clean_european_decimals(df["El_kWh"])
    if "Duration_hours" in df.columns:
        df["Duration_hours"] = clean_european_decimals(df["Duration_hours"])

    coerce_numeric_traffic(df)
    return df


def add_target_column(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = df.copy()
    out["El_kWh"] = pd.to_numeric(out["El_kWh"], errors="coerce")
    out["Duration_hours"] = pd.to_numeric(out["Duration_hours"], errors="coerce")
    if target == "El_kWh":
        out["target"] = out["El_kWh"]
    elif target == "El_kWh_per_hour":
        dur = out["Duration_hours"].replace(0, np.nan)
        out["target"] = out["El_kWh"] / dur
    else:
        raise ValueError(f"Unknown target {target!r}")
    return out


def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing target or duration (when duration is a feature)."""
    subset = ["target"]
    if "Duration_hours" in df.columns:
        subset.append("Duration_hours")
    out = df.dropna(subset=subset)
    return out


def validate_feature_frame(df: pd.DataFrame, *, stage: str = "features") -> None:
    if df.empty:
        raise ValueError(f"{stage}: dataframe is empty after cleaning.")
    na_target = df["target"].isna().sum() if "target" in df.columns else 0
    if na_target:
        raise ValueError(f"{stage}: {na_target} rows still have NaN target.")
