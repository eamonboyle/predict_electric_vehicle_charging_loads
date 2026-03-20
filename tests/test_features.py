from pathlib import Path

import numpy as np
import pytest

from ev_charging.config import load_config
from ev_charging.features import FeaturePreprocessor
from ev_charging.pipeline import build_train_test_arrays


@pytest.fixture(scope="module")
def root() -> Path:
    r = Path(__file__).resolve().parents[1]
    if not (r / "datasets").exists():
        pytest.skip("datasets/ not available")
    return r


def test_preprocessor_deterministic_shape(root: Path) -> None:
    cfg = load_config(root / "configs/default.yaml")
    cfg["training"]["test_size"] = 0.3
    (
        pre,
        X_train_full,
        y_train_full,
        _xte,
        _yte,
        _xm,
        _ym,
        _xv,
        _yv,
        _,
    ) = build_train_test_arrays(cfg, root)
    assert X_train_full.shape[0] == len(y_train_full)
    assert X_train_full.shape[1] == len(pre.feature_names_ or [])
    assert pre.feature_names_ is not None


def test_preprocessor_save_roundtrip(tmp_path: Path) -> None:
    import pandas as pd

    pre = FeaturePreprocessor(use_traffic=True)
    df = pd.DataFrame(
        {
            "User_type": ["Private", "Shared"],
            "Duration_hours": [1.0, 2.0],
            "month_plugin": ["Dec", "Jan"],
            "weekdays_plugin": ["Friday", "Monday"],
            "KROPPAN BRU": [1.0, 2.0],
            "MOHOLTLIA": [1.0, 2.0],
            "SELSBAKK": [1.0, 2.0],
            "MOHOLT RAMPE 2": [1, 2],
            "Jonsvannsveien vest for Steinanvegen": [3, 4],
            "target": [0.5, 1.5],
        }
    )
    pre.fit(df)
    X1, y1 = pre.transform(df)
    path = tmp_path / "p.joblib"
    pre.save(path)
    pre2 = FeaturePreprocessor.load(path)
    X2, y2 = pre2.transform(df)
    np.testing.assert_allclose(X1, X2)
    np.testing.assert_allclose(y1, y2)
