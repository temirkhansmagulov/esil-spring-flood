"""Run the analysis steps and write Tables 2-6 and figures."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import report as rp
from .config import Config
from .discharge import prepare_discharge, to_mm_per_day
from .timeutils import add_time_columns

log = logging.getLogger("esilflood")


# ------------------------------------------------------------------ step: data
def step_download(cfg: Config):
    from .era5 import download_era5_land

    files = download_era5_land(cfg)
    log.info("ERA5-Land: %d monthly files in %s", len(files), cfg["era5"]["raw_dir"])


def step_prepare(cfg: Config, force: bool = False):
    """Basin polygon, daily basin-mean forcing, screened discharge, static layers."""
    from .basin import load_or_delineate
    from .static_layers import build_static_layers, class_statistics

    cfg.ensure_dirs()
    polygon, area = load_or_delineate(cfg)
    log.info("Basin area %.0f km2 (published: %s km2)", area, cfg["basin"]["area_km2"])
    if abs(area - cfg["basin"]["area_km2"]) / cfg["basin"]["area_km2"] > 0.10:
        log.warning("Basin area differs from the published value by more than 10%%; "
                    "check the gauge location or polygon.")

    forcing_path = cfg.processed / "forcing_daily.csv"
    n_cells = None
    if forcing_path.exists() and not force:
        log.info("Using existing %s", forcing_path)
    else:
        from .era5 import process_era5_land

        forcing, n_cells = process_era5_land(cfg, polygon)
        forcing.to_csv(forcing_path)

    discharge, report = prepare_discharge(cfg)
    discharge.to_csv(cfg.processed / "discharge_clean.csv")

    layers = build_static_layers(cfg, polygon)
    stats = class_statistics(layers, cfg)
    stats.to_csv(cfg.processed / "class_statistics.csv")

    if n_cells is None:
        from .era5 import basin_weights
        lat = np.arange(cfg["basin"]["era5_area"][2], cfg["basin"]["era5_area"][0] + 1e-9, 0.1)
        lon = np.arange(cfg["basin"]["era5_area"][1], cfg["basin"]["era5_area"][3] + 1e-9, 0.1)
        n_cells = int((basin_weights(lat, lon, polygon, supersample=5) > 0).sum())
    meta = {"basin_area_km2": area, "era5_cells": n_cells, "discharge_report": report}
    with open(cfg.processed / "dataset_report.json", "w") as f:
        json.dump(meta, f, indent=2)
    return layers


# ------------------------------------------------------------------ loaders
def load_forcing(cfg) -> pd.DataFrame:
    df = pd.read_csv(cfg.processed / "forcing_daily.csv", parse_dates=["date"],
                     index_col="date")
    return add_time_columns(df, cfg)


def load_discharge(cfg) -> pd.DataFrame:
    return pd.read_csv(cfg.processed / "discharge_clean.csv", parse_dates=["date"],
                       index_col="date")


def load_layers(cfg) -> dict:
    import rasterio

    d = cfg.processed / "static"
    out = {}
    for name, f in (("landcover", "landcover.tif"), ("hsg", "hsg.tif"),
                    ("ks", "ks_cm_h.tif"), ("cn2", "cn2.tif")):
        with rasterio.open(d / f) as src:
            out[name] = src.read(1)
            out["transform"], out["crs"] = src.transform, src.crs
    out["mask"] = out["landcover"] > 0
    out["cn2"] = out["cn2"].astype(float)
    return out


# ------------------------------------------------------------------ table 2
def step_table2(cfg):
    forcing, discharge = load_forcing(cfg), load_discharge(cfg)
    meta = json.load(open(cfg.processed / "dataset_report.json"))
    df, notes = rp.table2(forcing, discharge, meta["discharge_report"], meta["era5_cells"], cfg)
    rp.save_table(df, cfg.tables, "table2_final_dataset", "Table 2: Final dataset.", notes)


# ------------------------------------------------------------------ RQ1
def step_rq1(cfg):
    from .rq1 import amc_comparison, daily_amc, ks_sensitivity, runoff_coefficients

    forcing, layers = load_forcing(cfg), load_layers(cfg)
    stats = pd.read_csv(cfg.processed / "class_statistics.csv", index_col="code")
    spring = forcing[forcing["spring"]]
    amc = daily_amc(spring, cfg)
    summary, yearly = runoff_coefficients(forcing, layers["cn2"], layers, cfg)
    yearly.to_csv(cfg.tables / "rq1_runoff_coefficient_per_year.csv")
    cmp_ = amc_comparison(forcing, layers, cfg, amc)
    t3, notes = rp.table3(stats, summary, cmp_, cfg)
    rp.save_table(t3, cfg.tables, "table3_runoff_coefficients",
                  "Table 3: Spring-event runoff coefficients by land-cover class.", notes)
    sens = ks_sensitivity(forcing, layers, cfg, summary)
    rp.save_table(rp.table4(sens, cfg), cfg.tables, "table4_ks_sensitivity",
                  "Table 4: Sensitivity of runoff coefficients to land-cover-adjusted Ks.")
    log.info("RQ1 done")


# ------------------------------------------------------------------ RQ2
def step_scenarios(cfg):
    from .rq2 import make_scenario_rasters

    shares = make_scenario_rasters(load_layers(cfg), cfg)
    shares.to_csv(cfg.tables / "rq2_scenario_landcover_shares.csv")
    log.info("Scenario land-cover maps written to %s", cfg.processed / "scenarios")


def load_swat(cfg) -> dict[str, pd.Series]:
    from .rq2 import read_swat_channel

    out = {}
    for name in ("baseline", "degradation", "recovery"):
        p = cfg.path(cfg["swat"][name])
        if p and p.exists():
            out[name] = read_swat_channel(p, cfg["swat"].get("outlet_gis_id"))
    return out


def step_rq2(cfg):
    from .rq2 import scenario_effects, scenario_effects_by_magnitude, swat_performance

    swat = load_swat(cfg)
    if "baseline" not in swat:
        log.warning("No SWAT+ baseline output found; skipping RQ2 effects (Table 5).")
        return
    obs = load_discharge(cfg)
    perf = swat_performance(swat["baseline"], obs, cfg)
    perf.to_csv(cfg.tables / "rq2_swat_performance.csv")
    area = cfg["basin"]["area_km2"]
    series = {k: to_mm_per_day(v, area) for k, v in swat.items()}
    eff = scenario_effects(series, cfg)
    rp.save_table(rp.table5(eff), cfg.tables, "table5_scenarios",
                  "Table 5: Change in spring flood characteristics at Esil–Turgen under "
                  "land-cover scenarios (median % change over years [95% CI]).",
                  "p (Wilcoxon) given for peak / volume / baseflow. SWAT+ baseline "
                  "performance per subset: see rq2_swat_performance.csv.")
    by_mag = scenario_effects_by_magnitude(series, cfg)
    by_mag.to_csv(cfg.tables / "rq2_scenarios_by_peak_magnitude.csv")
    log.info("RQ2 done")


# ------------------------------------------------------------------ RQ3
def step_rq3(cfg):
    from .rq3 import agreement_with_scenarios, run_rq3

    forcing, discharge = load_forcing(cfg), load_discharge(cfg)
    swat = load_swat(cfg)
    swat_mm = to_mm_per_day(swat["baseline"], cfg["basin"]["area_km2"]) if "baseline" in swat else None
    res = run_rq3(forcing, discharge, cfg, swat_mm)
    agree = agreement_with_scenarios(res["shap"], cfg)
    t6, notes = rp.table6(res, agree)
    rp.save_table(t6, cfg.tables, "table6_model_comparison",
                  "Table 6: Comparison of the proposed model with baselines on the test "
                  "years (spring window) at Esil–Turgen.", notes)
    shap_tbl = pd.DataFrame({"variable": res["shap"].index,
                             "mean_abs_shap": res["shap"].values.astype(float).round(4),
                             "sd_over_seeds": res["shap_by_seed"][res["shap"].index]
                             .std(ddof=1).values.astype(float).round(4)})
    shap_tbl.insert(0, "rank", range(1, len(shap_tbl) + 1))
    rp.save_table(shap_tbl, cfg.tables, "rq3_shap_ranking",
                  "SHAP ranking of drivers of observed spring discharge (test years).")
    res["predictions"].to_csv(cfg.tables / "rq3_test_predictions.csv")
    rp.plot_test_hydrographs(res["predictions"], cfg.figures / "rq3_test_hydrographs.png")
    rp.plot_shap(res["shap"], res["shap_by_seed"], cfg.figures / "rq3_shap_ranking.png")
    log.info("RQ3 done (train %d, validation %d, test %d spring days)",
             res["n_train"], res["n_val"], res["n_test"])


STEPS = {"download": step_download, "prepare": step_prepare, "table2": step_table2,
         "rq1": step_rq1, "scenarios": step_scenarios, "rq2": step_rq2, "rq3": step_rq3}
DEFAULT_STEPS = ["prepare", "table2", "rq1", "scenarios", "rq2", "rq3"]
