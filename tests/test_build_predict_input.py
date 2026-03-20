from pathlib import Path

import pytest

from ev_charging.config import load_config
from ev_charging.pipeline import build_prediction_table


@pytest.fixture(scope="module")
def root() -> Path:
    r = Path(__file__).resolve().parents[1]
    if not (r / "datasets").exists():
        pytest.skip("datasets/ not available")
    return r


def test_build_prediction_table_smoke(root: Path) -> None:
    cfg = load_config(root / "configs/default.yaml")
    df = build_prediction_table(cfg, root, max_rows=30)
    assert len(df) > 0
    assert "User_type" in df.columns
    assert "Duration_hours" in df.columns
    assert "month_plugin" in df.columns
    if cfg["training"]["use_traffic"]:
        assert "KROPPAN BRU" in df.columns
