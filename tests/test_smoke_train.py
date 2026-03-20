from pathlib import Path

import pytest
import yaml

from ev_charging.cli import cmd_train


@pytest.fixture(scope="module")
def root() -> Path:
    r = Path(__file__).resolve().parents[1]
    if not (r / "datasets").exists():
        pytest.skip("datasets/ not available")
    return r


def test_train_smoke(tmp_path: Path, root: Path) -> None:
    cfg = yaml.safe_load((root / "configs/default.yaml").read_text())
    cfg["mlp"]["epochs"] = 8
    cfg["mlp"]["early_stopping_patience"] = 0
    cfg["mlp"]["val_fraction"] = 0.0
    cfg["cross_validation"]["enabled"] = False
    cfg["hyperparam_search"]["enabled"] = False
    cfg["evaluation"]["plot"] = False
    cfg["evaluation"]["permutation_importance_repeats"] = 0
    cfg["evaluation"]["residual_intervals"]["enabled"] = False
    cfg["outputs"]["model_path"] = "models/mlp.pt"
    cfg["outputs"]["preprocessor_path"] = "models/preprocessor.joblib"
    cfg["outputs"]["baselines_path"] = "models/baselines.joblib"
    cfg["outputs"]["plots_dir"] = "artifacts/plots"
    cfg["outputs"]["metrics_csv"] = "runs/metrics.csv"
    cfg["data"]["ev_charging_csv"] = "datasets/EV charging reports.csv"
    cfg["data"]["traffic_csv"] = "datasets/Local traffic distribution.csv"

    root_tmp = tmp_path / "proj"
    root_tmp.mkdir()
    (root_tmp / "configs").mkdir()
    (root_tmp / "datasets").mkdir()
    (root_tmp / "models").mkdir()
    (root_tmp / "artifacts" / "plots").mkdir(parents=True)
    (root_tmp / "runs").mkdir()

    import shutil

    shutil.copy(
        root / "datasets/EV charging reports.csv",
        root_tmp / "datasets/EV charging reports.csv",
    )
    shutil.copy(
        root / "datasets/Local traffic distribution.csv",
        root_tmp / "datasets/Local traffic distribution.csv",
    )

    cfg_path = root_tmp / "configs/smoke.yaml"
    cfg_path.write_text(yaml.dump(cfg))

    class NS:
        root = str(root_tmp)
        config = str(cfg_path)

    assert cmd_train(NS()) == 0
    assert (root_tmp / "models/mlp.pt").is_file()
    assert (root_tmp / "models/preprocessor.joblib").is_file()
