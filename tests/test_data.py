from pathlib import Path

import pytest

from ev_charging.data import (
    load_ev_charging_reports,
    load_traffic_reports,
    merge_ev_traffic,
)
from ev_charging.extra_data import hourly_private_for_merge, merge_hourly_private_features


@pytest.fixture(scope="module")
def root() -> Path:
    r = Path(__file__).resolve().parents[1]
    if not (r / "datasets").exists():
        pytest.skip("datasets/ not available")
    return r


def test_load_and_merge(root: Path) -> None:
    ev = load_ev_charging_reports(root / "datasets/EV charging reports.csv")
    tr = load_traffic_reports(root / "datasets/Local traffic distribution.csv")
    m = merge_ev_traffic(ev.head(200), tr)
    assert len(m) > 0
    assert "KROPPAN BRU" in m.columns


def test_prepare_and_hourly_merge(root: Path) -> None:
    ev = load_ev_charging_reports(root / "datasets/EV charging reports.csv")
    tr = load_traffic_reports(root / "datasets/Local traffic distribution.csv")
    m = merge_ev_traffic(ev.head(500), tr)
    h = hourly_private_for_merge(root / "datasets/Hourly EV loads - Aggregated private.csv")
    m2 = merge_hourly_private_features(m, h)
    hourly_cols = [c for c in m2.columns if c.startswith("hourly_")]
    assert hourly_cols
    assert m2[hourly_cols[0]].notna().any()


def test_validation_missing_column(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("a;b\n1;2\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_ev_charging_reports(p)
