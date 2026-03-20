from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from ev_charging.constants import META_COLUMNS_FOR_SPLIT, TRAFFIC_COUNT_COLUMNS


@dataclass
class FeaturePreprocessor:
    """Fit on training rows only; encodes categoricals and assembles the numeric design matrix."""

    use_traffic: bool = True
    _extra_numeric_cols: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._month_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self._day_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.feature_names_: list[str] | None = None

    def _drop_meta(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df
        to_drop = [c for c in META_COLUMNS_FOR_SPLIT if c in d.columns]
        if to_drop:
            d = d.drop(columns=to_drop)
        return d

    def _traffic_cols(self, df: pd.DataFrame) -> list[str]:
        if not self.use_traffic:
            return []
        return [c for c in TRAFFIC_COUNT_COLUMNS if c in df.columns]

    @staticmethod
    def _hourly_cols(df: pd.DataFrame) -> list[str]:
        return sorted(c for c in df.columns if c.startswith("hourly_"))

    def fit(self, df: pd.DataFrame) -> FeaturePreprocessor:
        d = self._drop_meta(df.copy())
        self._month_ohe.fit(d[["month_plugin"]].astype(str))
        self._day_ohe.fit(d[["weekdays_plugin"]].astype(str))
        self._extra_numeric_cols = self._hourly_cols(d)
        m_cats = self._month_ohe.categories_[0]
        w_cats = self._day_ohe.categories_[0]
        self.feature_names_ = (
            ["User_type", "Duration_hours"]
            + [f"month_{c}" for c in m_cats]
            + [f"wd_{c}" for c in w_cats]
            + self._traffic_cols(d)
            + self._extra_numeric_cols
        )
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        d = self._drop_meta(df.copy())
        y = None
        if "target" in d.columns:
            y = pd.to_numeric(d["target"], errors="coerce").to_numpy(dtype=np.float64)
        for c in ("target", "El_kWh"):
            if c in d.columns:
                d = d.drop(columns=[c])

        d["User_type"] = (d["User_type"].astype(str) == "Private").astype(np.float64)
        d["Duration_hours"] = pd.to_numeric(d["Duration_hours"], errors="coerce")

        Xm = self._month_ohe.transform(d[["month_plugin"]].astype(str))
        Xw = self._day_ohe.transform(d[["weekdays_plugin"]].astype(str))

        blocks: list[np.ndarray] = [
            d[["User_type", "Duration_hours"]].to_numpy(dtype=np.float64),
            Xm,
            Xw,
        ]
        tcols = self._traffic_cols(d)
        if tcols:
            t = d[tcols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            blocks.append(t.to_numpy(dtype=np.float64))

        if self._extra_numeric_cols:
            parts = []
            for c in self._extra_numeric_cols:
                if c in d.columns:
                    parts.append(pd.to_numeric(d[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64))
                else:
                    parts.append(np.zeros(len(d), dtype=np.float64))
            blocks.append(np.column_stack(parts))

        X = np.hstack(blocks)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, y

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        self.fit(df)
        return self.transform(df)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> FeaturePreprocessor:
        return joblib.load(path)
