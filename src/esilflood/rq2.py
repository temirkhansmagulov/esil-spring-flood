"""RQ2: land-cover scenarios for SWAT+ and their effect at the Turgen gauge (Table 5).

Workflow
1. `make_scenario_rasters` writes degradation/recovery land-cover maps.
2. Build one SWAT+ project per map in QSWAT+ (same DEM, soils, weather), calibrate the
   baseline and copy the calibrated parameters to the scenario projects.
3. Save each run's channel_sd_day.txt where config.yaml expects it.
4. `scenario_effects` computes the changes reported in Table 5.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import all_metrics, bootstrap_ci_values, wilcoxon_paired
from .static_layers import write_tif
from .timeutils import add_time_columns


# ------------------------------------------------------------ scenario rasters
def apply_conversions(lc: np.ndarray, steps: list[dict], rng) -> np.ndarray:
    """Convert random, non-overlapping fractions of a class to other classes.

    Fractions refer to the class area in the original map, so a recovery of
    30% cropland -> grassland plus 5% cropland -> trees converts 35% of cropland."""
    out = lc.copy()
    shuffled = {}
    used = {}
    for step in steps:
        src = step["from"]
        if src not in shuffled:
            shuffled[src] = rng.permutation(np.flatnonzero(lc == src))
            used[src] = 0
        n = int(round(step["fraction"] * len(shuffled[src])))
        pick = shuffled[src][used[src]:used[src] + n]
        used[src] += len(pick)
        out.flat[pick] = step["to"]
    return out


def make_scenarios(lc: np.ndarray, cfg) -> dict[str, np.ndarray]:
    sc = cfg["scenarios"]
    rng = np.random.default_rng(sc["seed"])
    degr = apply_conversions(lc, [sc["degradation"]], rng)
    rec = apply_conversions(lc, sc["recovery"], rng)
    return {"baseline": lc, "degradation": degr, "recovery": rec}


def make_scenario_rasters(layers: dict, cfg) -> pd.DataFrame:
    """Write scenario land-cover GeoTIFFs and return class shares for each scenario."""
    scen = make_scenarios(layers["landcover"], cfg)
    out_dir = cfg.processed / "scenarios"
    rows = []
    for name, arr in scen.items():
        write_tif(out_dir / f"landcover_{name}.tif", arr.astype("uint8"),
                  layers["transform"], layers["crs"], nodata=0)
        valid = arr[arr > 0]
        for code in np.unique(valid):
            rows.append({"scenario": name, "code": int(code),
                         "area_pct": 100.0 * np.count_nonzero(valid == code) / len(valid)})
    return pd.DataFrame(rows).pivot(index="code", columns="scenario", values="area_pct")


# ------------------------------------------------------------ SWAT+ output
def read_swat_channel(path: Path, gis_id=None) -> pd.Series:
    """Daily outflow (m3/s) at the outlet from SWAT+ channel_sd_day.txt, or from a CSV
    with columns date,q_m3s."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        return df["q_m3s"].astype(float)
    df = pd.read_csv(path, sep=r"\s+", skiprows=[0, 2], header=0, low_memory=False)
    if gis_id is not None:
        df = df[df["gis_id"] == gis_id]
    elif df["gis_id"].nunique() > 1:
        raise ValueError("Several channels in SWAT+ output; set swat.outlet_gis_id.")
    date = pd.to_datetime(dict(year=df["yr"], month=df["mon"], day=df["day"]))
    flow_col = "flo_out" if "flo_out" in df.columns else "flo"
    return pd.Series(df[flow_col].astype(float).values, index=date, name="q_m3s")


# ------------------------------------------------------------ effects (Table 5)
def yearly_characteristics(q_mm: pd.Series, cfg) -> pd.DataFrame:
    """Per hydrological year: spring peak (mm/d), spring volume (mm), JJA mean baseflow."""
    df = add_time_columns(pd.DataFrame({"q": q_mm}), cfg)
    spring = df[df["spring"]].groupby("hydro_year")["q"]
    months = cfg["period"]["baseflow_months"]
    summer = df[df.index.month.isin(months)].groupby("hydro_year")["q"].mean()
    return pd.DataFrame({"peak": spring.max(), "volume": spring.sum(), "baseflow": summer})


def scenario_effects(series: dict[str, pd.Series], cfg) -> pd.DataFrame:
    """Median % change per year vs baseline with bootstrap CI and paired Wilcoxon p."""
    base = yearly_characteristics(series["baseline"], cfg).dropna()
    n_boot = cfg["project"]["bootstrap_iterations"]
    rows = []
    for name in ("degradation", "recovery"):
        if name not in series:
            continue
        sc = yearly_characteristics(series[name], cfg).loc[base.index]
        row = {"scenario": name}
        pvals = []
        for var in ("peak", "volume", "baseflow"):
            ch = 100.0 * (sc[var] - base[var]) / base[var]
            row[f"d_{var}"] = float(ch.median())
            row[f"d_{var}_lo"], row[f"d_{var}_hi"] = bootstrap_ci_values(ch.values, n=n_boot)
            p = wilcoxon_paired(sc[var].values, base[var].values)
            row[f"p_{var}"] = p
            pvals.append(p)
        rows.append(row)
    return pd.DataFrame(rows).set_index("scenario") if rows else pd.DataFrame()


def scenario_effects_by_magnitude(series, cfg, k: int = 3) -> pd.DataFrame:
    """Table 5 repeated for the k largest and k smallest observed spring peak years."""
    base = yearly_characteristics(series["baseline"], cfg).dropna()
    order = base["peak"].sort_values()
    out = []
    for label, years in (("smallest", order.index[:k]), ("largest", order.index[-k:])):
        sub = {n: s[add_time_columns(pd.DataFrame(index=s.index), cfg)["hydro_year"]
                    .isin(years).values] for n, s in series.items()}
        eff = scenario_effects(sub, cfg)
        eff["subset"] = f"{k} {label} peak years"
        out.append(eff)
    return pd.concat(out)


def swat_performance(sim_m3s: pd.Series, obs: pd.DataFrame, cfg) -> pd.DataFrame:
    """SWAT+ baseline performance per subset (calibration/validation/test)."""
    df = obs[["q_m3s", "hydro_year", "split", "spring"]].join(sim_m3s.rename("sim"),
                                                               how="inner").dropna()
    rows = []
    for split, g in df.groupby("split"):
        m = all_metrics(g["q_m3s"], g["sim"], g["hydro_year"])
        m["split"] = split
        rows.append(m)
    return pd.DataFrame(rows).set_index("split")
