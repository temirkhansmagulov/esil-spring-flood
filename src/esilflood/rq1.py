"""RQ1: spring runoff coefficients of soil-land-cover classes (Tables 3 and 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import bootstrap_ci_values, wilcoxon_paired
from .static_layers import cn_amc, cn_grid, hsg_from_ks, scs_runoff


def daily_amc(forcing: pd.DataFrame, cfg) -> pd.Series:
    """AMC class (1 dry, 2 average, 3 wet) for each spring day from ERA5-Land layer-1
    soil moisture percentiles; frozen topsoil is treated as wet (AMC III)."""
    a = cfg["amc"]
    sm = forcing["swvl1"]
    lo, hi = np.nanpercentile(sm, [a["dry_percentile"], a["wet_percentile"]])
    amc = pd.Series(2, index=forcing.index)
    amc[sm <= lo] = 1
    amc[sm >= hi] = 3
    if a.get("frozen_soil_is_wet", True) and "stl1_c" in forcing:
        amc[forcing["stl1_c"] < 0] = 3
    return amc


def _cn_histogram(cn: np.ndarray, sel: np.ndarray):
    vals = cn[sel & np.isfinite(cn)]
    if len(vals) == 0:
        return np.array([]), np.array([])
    u, c = np.unique(np.round(vals, 1), return_counts=True)
    return u, c / c.sum()


def class_daily_runoff(p_mm: np.ndarray, cn_values, weights, amc=None) -> np.ndarray:
    """Area-weighted daily runoff (mm) of a class with a distribution of CN_II values."""
    if len(cn_values) == 0:
        return np.full(len(p_mm), np.nan)
    cn = np.broadcast_to(cn_values, (len(p_mm), len(cn_values)))
    if amc is not None:
        cn = cn_amc(cn, np.asarray(amc)[:, None])
    q = scs_runoff(np.asarray(p_mm)[:, None], cn)
    return q @ weights


def _pooled_c(df: pd.DataFrame) -> float:
    return df["Q"].sum() / df["P"].sum() if df["P"].sum() > 0 else np.nan


def runoff_coefficients(forcing: pd.DataFrame, cn: np.ndarray, layers: dict, cfg,
                        amc: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pooled spring C = sum(Q)/sum(P) per class with a block-bootstrap CI over years.

    Returns (summary per class, per-year C per class)."""
    spring = forcing[forcing["spring"]]
    p = (spring["tp_mm"] + spring["smlt_mm"]).values        # P = precipitation + snowmelt
    lc, mask = layers["landcover"], layers["mask"]
    n_boot = cfg["project"]["bootstrap_iterations"]
    rows, yearly = [], {}
    for code in cfg["table3_classes"]:
        u, w = _cn_histogram(cn, mask & (lc == int(code)))
        q = class_daily_runoff(p, u, w, None if amc is None else amc.loc[spring.index].values)
        d = pd.DataFrame({"P": p, "Q": q, "year": spring["hydro_year"].values})
        per_year = d.groupby("year").apply(_pooled_c, include_groups=False)
        yearly[int(code)] = per_year
        c = _pooled_c(d)
        lo, hi = _year_block_ci(d, n_boot)
        rows.append({"code": int(code), "C": c, "C_lo": lo, "C_hi": hi})
    return pd.DataFrame(rows).set_index("code"), pd.DataFrame(yearly)


def _year_block_ci(d: pd.DataFrame, n: int, seed: int = 0):
    if d["Q"].isna().all():
        return np.nan, np.nan
    sums = d.groupby("year")[["P", "Q"]].sum()
    rng = np.random.default_rng(seed)
    idx = np.arange(len(sums))
    boots = []
    for _ in range(n):
        s = sums.iloc[rng.choice(idx, len(idx), replace=True)].sum()
        if s["P"] > 0:
            boots.append(s["Q"] / s["P"])
    return tuple(np.percentile(boots, [2.5, 97.5])) if boots else (np.nan, np.nan)


def basin_c_per_year(forcing, cn, layers, amc=None) -> pd.Series:
    """Basin-wide spring C per hydrological year (all classes, area-weighted)."""
    spring = forcing[forcing["spring"]]
    p = (spring["tp_mm"] + spring["smlt_mm"]).values
    u, w = _cn_histogram(cn, layers["mask"] & (layers["landcover"] > 0))
    q = class_daily_runoff(p, u, w, None if amc is None else amc.loc[spring.index].values)
    d = pd.DataFrame({"P": p, "Q": q, "year": spring["hydro_year"].values})
    return d.groupby("year").apply(_pooled_c, include_groups=False)


def amc_comparison(forcing, layers, cfg, amc: pd.Series) -> dict:
    """Bottom row of Table 3: tabulated CN (AMC II) vs AMC-adjusted CN (Eq. 4)."""
    c2 = basin_c_per_year(forcing, layers["cn2"], layers)
    cadj = basin_c_per_year(forcing, layers["cn2"], layers, amc)
    diff = (cadj - c2).dropna()
    lo, hi = bootstrap_ci_values(diff.values, stat=np.mean,
                                 n=cfg["project"]["bootstrap_iterations"])
    return {"mean_dC": float(diff.mean()), "ci_lo": lo, "ci_hi": hi,
            "p_wilcoxon": wilcoxon_paired(cadj.loc[diff.index], c2.loc[diff.index]),
            "n_years": int(len(diff))}


def ks_sensitivity(forcing, layers, cfg, base_summary: pd.DataFrame) -> pd.DataFrame:
    """Table 4: multiply texture-based Ks by land-cover ratios [3], reassign HSG and CN,
    and report the change in C relative to the base run."""
    ks = layers["ks"]
    mult = {int(k): v for k, v in cfg["ks_landcover_multiplier"].items()}
    rows = []
    if not np.isfinite(ks).any():
        for code, m in mult.items():
            rows.append({"code": code, "multiplier": m, "dC": np.nan, "dC_lo": np.nan,
                         "dC_hi": np.nan, "note": "needs SoilGrids texture (Ks grid)"})
        return pd.DataFrame(rows).set_index("code")
    ks_adj = ks.copy()
    for code, m in mult.items():
        sel = layers["landcover"] == code
        ks_adj[sel] = ks[sel] * m
    hsg_adj = hsg_from_ks(ks_adj, cfg["hsg_ks_thresholds_cm_h"])
    hsg_adj[~layers["mask"]] = 0
    cn_adj = cn_grid(layers["landcover"], hsg_adj, cfg["curve_numbers"])
    cn_adj[~layers["mask"]] = np.nan

    spring = forcing[forcing["spring"]]
    p = (spring["tp_mm"] + spring["smlt_mm"]).values
    years = spring["hydro_year"].values
    for code, m in mult.items():
        sel = layers["mask"] & (layers["landcover"] == code)
        u0, w0 = _cn_histogram(layers["cn2"], sel)
        u1, w1 = _cn_histogram(cn_adj, sel)
        q0 = class_daily_runoff(p, u0, w0)
        q1 = class_daily_runoff(p, u1, w1)
        d = pd.DataFrame({"P": p, "Q0": q0, "Q1": q1, "year": years})
        sums = d.groupby("year")[["P", "Q0", "Q1"]].sum()
        dc = (sums["Q1"].sum() - sums["Q0"].sum()) / sums["P"].sum() if sums["P"].sum() else np.nan
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(cfg["project"]["bootstrap_iterations"]):
            s = sums.iloc[rng.choice(len(sums), len(sums), replace=True)].sum()
            if s["P"] > 0:
                boots.append((s["Q1"] - s["Q0"]) / s["P"])
        lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
        rows.append({"code": code, "multiplier": m, "dC": dc, "dC_lo": lo, "dC_hi": hi,
                     "note": ""})
    return pd.DataFrame(rows).set_index("code")
