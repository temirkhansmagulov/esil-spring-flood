"""End-to-end smoke test on synthetic inputs (about 15 s)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("rasterio")
pytest.importorskip("xgboost")
pytest.importorskip("shap")


def test_demo_pipeline(tmp_path):
    from esilflood.config import load_config
    from esilflood.pipeline import DEFAULT_STEPS, STEPS
    from esilflood.synthetic import make_demo_inputs

    make_demo_inputs(tmp_path)
    cfg = load_config(root=tmp_path)
    cfg.raw["static"]["target_resolution_m"] = 500
    cfg.raw["project"]["bootstrap_iterations"] = 50
    cfg.raw["project"]["random_seeds"] = [0, 1]
    cfg.raw["rq3"]["xgboost"]["n_estimators"] = 200
    cfg.ensure_dirs()
    for step in DEFAULT_STEPS:
        STEPS[step](cfg)
    for name in ("table2_final_dataset", "table3_runoff_coefficients", "table4_ks_sensitivity",
                 "table5_scenarios", "table6_model_comparison"):
        assert (cfg.tables / f"{name}.csv").exists(), name
    t6 = pd.read_csv(cfg.tables / "table6_model_comparison.csv")
    assert "XGBoost (proposed)" in t6["Model"].values
    t2 = pd.read_csv(cfg.tables / "table2_final_dataset.csv")
    assert "8,401 days" in t2.loc[0, "Records"]
