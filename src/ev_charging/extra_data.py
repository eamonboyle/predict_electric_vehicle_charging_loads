"""Optional datasets: AMS garage metering and hourly synthetic EV load aggregates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_ams_garage(path: str | Path) -> pd.DataFrame:
    """Hourly AMS kWh and synthetic simultaneous-charging scenarios for garage Bl2."""
    path = Path(path)
    return pd.read_csv(path, sep=";", na_values=["NA"])


def load_hourly_ev_private(path: str | Path) -> pd.DataFrame:
    """Hourly private-EV synthetic/flex load aggregates and user counts."""
    path = Path(path)
    return pd.read_csv(path, sep=";", na_values=["NA"])


def hourly_private_for_merge(path: str | Path) -> pd.DataFrame:
    """Normalize hourly private CSV: parse European decimals; keep `date_from` as merge key."""
    df = load_hourly_ev_private(path)
    if "date_from" not in df.columns:
        raise ValueError("Expected column date_from in hourly private CSV")
    skip = {"date_from", "weekday", "month"}
    for c in df.columns:
        if c in skip or c == "daily_hour":
            continue
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace(",", ".", regex=False)
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def merge_hourly_private_features(merged_ev_traffic: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """Prefix hourly numeric columns and left-merge on floored session start hour."""
    out = merged_ev_traffic.copy()
    plug = pd.to_datetime(out["Start_plugin"], dayfirst=True, errors="coerce")
    out["_hourly_join"] = plug.dt.floor("h").dt.strftime("%d.%m.%Y %H:%M")

    h = hourly.copy()
    key = "date_from"
    feat_cols = [c for c in h.columns if c != key]
    ren = {c: f"hourly_{c}" for c in feat_cols}
    h = h.rename(columns=ren)
    merged = pd.merge(out, h, left_on="_hourly_join", right_on=key, how="left")
    merged = merged.drop(columns=["_hourly_join", key], errors="ignore")
    return merged


def describe_extra_datasets() -> str:
    return (
        "Optional CSVs under `datasets/`:\n"
        "- `AMS data from garage Bl2.csv`: hourly AMS energy and synthetic charger scenarios.\n"
        "- `Hourly EV loads - Aggregated private.csv` (and shared/per-user variants): synthetic "
        "flex/load shapes; merge on floored session `Start_plugin` hour when "
        "`use_hourly_private_features: true` in config.\n"
        "- `Hourly EV loads - Per user.csv`: user-level hourly series (exploration / future models).\n"
    )
