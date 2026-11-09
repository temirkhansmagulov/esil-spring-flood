"""Write Tables 2-6 as CSV and Markdown, plus figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def fmt(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "TBD"
    return f"{x:.{nd}f}"


def fmt_ci(v, lo, hi, nd=2) -> str:
    if v is None or not np.isfinite(v):
        return "TBD"
    if lo is None or not np.isfinite(lo):
        return fmt(v, nd)
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def fmt_p(p) -> str:
    if p is None or not np.isfinite(p):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, out_dir: Path, name: str, title: str, notes: str = ""):
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    with open(out_dir / f"{name}.md", "w", encoding="utf-8") as f:
        f.write(f"**{title}**\n\n")
        f.write(to_markdown(df))
        if notes:
            f.write(f"\n{notes}\n")


# ---------------------------------------------------------------- Table 2
def table2(forcing, discharge, report, n_cells, cfg) -> pd.DataFrame:
    s = cfg["period"]
    splits = []
    for name in ("train", "validation", "test"):
        sub = discharge[discharge["split"] == name]
        y0, y1 = s["split"][name]
        splits.append(f"{name} {y0}/{str(y0+1)[-2:]}–{y1}/{str(y1+1)[-2:]}: "
                      f"{len(sub):,} days ({int(sub['spring'].sum()):,} spring)")
    q_spring = discharge.loc[discharge["spring"], "q_m3s"].dropna()
    rows = [
        {"Variable": "Precipitation, snowmelt, SWE, 2-m temperature, soil moisture, "
                     "soil temperature",
         "Source": "ERA5-Land hourly, aggregated to daily",
         "Coverage": f"{forcing.index.min():%Y-%m-%d} – {forcing.index.max():%Y-%m-%d}",
         "Records": f"{len(forcing):,} days × {n_cells} cells",
         "Removed (reason)": f"{int(forcing.isna().any(axis=1).sum())} days with missing values"},
        {"Variable": "Daily discharge (Esil–Turgen)",
         "Source": "Kazhydromet open hydrological database",
         "Coverage": f"{s['start']} – {s['end']}",
         "Records": f"{report['valid_days']:,} valid of {report['days_in_period']:,} days",
         "Removed (reason)": (f"{report['removed_long_gap_days']} gap days (>"
                              f"{cfg['discharge']['max_gap_days']} d), "
                              f"{report['removed_ice_days']} ice days; "
                              f"{report['interpolated_days']} days interpolated")},
    ]
    df = pd.DataFrame(rows)
    q = q_spring.quantile([0.05, 0.5, 0.95, 0.99])
    notes = ("Split by hydrological years: " + "; ".join(splits) + ". Spring discharge "
             f"quantiles (m³/s): Q5 = {q.iloc[0]:.2f}, median = {q.iloc[1]:.2f}, "
             f"Q95 = {q.iloc[2]:.2f}, Q99 = {q.iloc[3]:.2f}; skewness = {q_spring.skew():.2f}.")
    return df, notes


# ---------------------------------------------------------------- Table 3 & 4
EXPECTED_RANK = {50: ("↑↑ highest", "[12, 11, 31]"), 60: ("↑ higher", "[12]"),
                 40: ("↑ higher", "[3, 4]"), 30: ("reference", "[3]"),
                 20: ("↓ lower", "[3]"), 10: ("↓↓ lowest", "[3, 10]")}


def table3(stats: pd.DataFrame, c: pd.DataFrame, amc_cmp: dict, cfg):
    rows = []
    for code in cfg["table3_classes"]:
        code = int(code)
        st = stats.loc[code] if code in stats.index else None
        cc = c.loc[code] if code in c.index else None
        exp, basis = EXPECTED_RANK.get(code, ("", ""))
        rows.append({
            "Land cover": cfg["curve_numbers"][code]["name"],
            "Area (%)": fmt(st["area_pct"], 1) if st is not None else "0.0",
            "HSG": st["dominant_hsg"] if st is not None else "–",
            "Ks (cm h⁻¹)": fmt(st["ks_cm_h"], 2) if st is not None else "–",
            "CN_II": fmt(st["cn2_mean"], 1) if st is not None else "–",
            "C [95% CI]": fmt_ci(cc["C"], cc["C_lo"], cc["C_hi"], 3) if cc is not None else "–",
            "Observed rank": "",
            "Expected rank": exp, "Basis": basis})
    df = pd.DataFrame(rows)
    order = df["C [95% CI]"].str.extract(r"^([0-9.]+)")[0].astype(float)
    df["Observed rank"] = order.rank(ascending=False, method="min").map(
        lambda r: "–" if not np.isfinite(r) else str(int(r)))
    notes = (f"Tabulated CN (AMC II) vs. AMC-adjusted CN (Eq. 4): mean ΔC = "
             f"{fmt_ci(amc_cmp['mean_dC'], amc_cmp['ci_lo'], amc_cmp['ci_hi'], 3)}; "
             f"Wilcoxon p = {fmt_p(amc_cmp['p_wilcoxon'])} (n = {amc_cmp['n_years']} years). "
             "C = ΣQ/ΣP over all spring days with P = precipitation + snowmelt; 95% CI from "
             "a block bootstrap over hydrological years.")
    return df, notes


def table4(sens: pd.DataFrame, cfg):
    expected = {40: "0 (reference)", 30: "↓", 10: "↓↓"}
    rows = []
    for code, r in sens.iterrows():
        rows.append({"Land cover": cfg["curve_numbers"][int(code)]["name"],
                     "Ks multiplier": fmt(r["multiplier"], 1),
                     "Expected ΔC": expected.get(int(code), ""),
                     "ΔC [95% CI]": ("0 (reference)" if r["multiplier"] == 1.0 and
                                     abs(r["dC"] if np.isfinite(r["dC"]) else 0) < 1e-12
                                     else fmt_ci(r["dC"], r["dC_lo"], r["dC_hi"], 4))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Table 5
def table5(eff: pd.DataFrame):
    rows = [{"Scenario": "Baseline / preservation (WorldCover 2021)", "Δ Peak (%)": "0",
             "Δ Volume (%)": "0", "Δ Baseflow (%)": "0", "p (Wilcoxon)": "–"}]
    labels = {"degradation": "Degradation: 30% of grassland → cropland",
              "recovery": "Recovery: 30% of cropland → grassland, +5% tree belts"}
    for name, r in eff.iterrows():
        rows.append({"Scenario": labels.get(name, name),
                     "Δ Peak (%)": fmt_ci(r["d_peak"], r["d_peak_lo"], r["d_peak_hi"], 1),
                     "Δ Volume (%)": fmt_ci(r["d_volume"], r["d_volume_lo"], r["d_volume_hi"], 1),
                     "Δ Baseflow (%)": fmt_ci(r["d_baseflow"], r["d_baseflow_lo"],
                                              r["d_baseflow_hi"], 1),
                     "p (Wilcoxon)": " / ".join(fmt_p(r[f"p_{v}"])
                                                for v in ("peak", "volume", "baseflow"))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Table 6
def table6(rq3: dict, agree: dict):
    rows = []
    order = ["Climatology (day-of-year mean)", "Linear regression", "SWAT+ (calibrated)",
             "XGBoost (proposed)"]
    for name in order:
        if name not in rq3["metrics"]:
            rows.append({"Model": name, "NSE (–)": "TBD", "KGE (–)": "TBD", "PBIAS (%)": "TBD",
                         "Ep (%)": "TBD", "Δtp (d)": "TBD", "Wilcoxon p vs XGBoost": "–"})
            continue
        m = rq3["metrics"][name]
        if name == "XGBoost (proposed)":
            mu, sd = m["seed_mean"], m["seed_sd"]
            cell = lambda k, nd: f"{mu[k]:.{nd}f} ± {sd[k]:.{nd}f}"
            row = {"Model": name, "NSE (–)": cell("NSE", 2), "KGE (–)": cell("KGE", 2),
                   "PBIAS (%)": cell("PBIAS", 1), "Ep (%)": cell("Ep", 1),
                   "Δtp (d)": cell("dtp", 1), "Wilcoxon p vs XGBoost": "–"}
        else:
            row = {"Model": name,
                   "NSE (–)": fmt_ci(m["NSE"], *m["NSE_ci"]),
                   "KGE (–)": fmt_ci(m["KGE"], *m["KGE_ci"]),
                   "PBIAS (%)": fmt_ci(m["PBIAS"], *m["PBIAS_ci"], nd=1),
                   "Ep (%)": fmt_ci(m["Ep"], *m["Ep_ci"], nd=1),
                   "Δtp (d)": fmt_ci(m["dtp"], *m["dtp_ci"], nd=1),
                   "Wilcoxon p vs XGBoost": fmt_p(rq3["wilcoxon"].get(name))}
        rows.append(row)
    rows.append({"Model": "Expected target [23]", "NSE (–)": "> 0.50", "KGE (–)": "–",
                 "PBIAS (%)": "±25", "Ep (%)": "–", "Δtp (d)": "–", "Wilcoxon p vs XGBoost": "–"})
    rows.append({"Model": "HBV-light, 2005–2010 (published) [21]", "NSE (–)": "0.85",
                 "KGE (–)": "–", "PBIAS (%)": "–", "Ep (%)": "–", "Δtp (d)": "–",
                 "Wilcoxon p vs XGBoost": "–"})
    xm = rq3["metrics"]["XGBoost (proposed)"]
    notes = (f"XGBoost: mean ± SD over {len(rq3['shap_by_seed'])} seeds; seed-averaged "
             f"prediction NSE 95% CI [{fmt(xm['NSE_ci'][0])}, {fmt(xm['NSE_ci'][1])}]. "
             "Baselines: value [95% block-bootstrap CI]. Wilcoxon: paired per-year RMSE on the "
             f"test years. Agreement of the SHAP ranking with the scenario sensitivity: "
             f"Spearman ρ = {fmt(agree.get('rho'))} (p = {fmt_p(agree.get('p'))}); "
             f"{agree.get('note', '')}.")
    return pd.DataFrame(rows), notes


# ---------------------------------------------------------------- figures
def plot_test_hydrographs(pred: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = sorted(set(pred.index.year))
    fig, axes = plt.subplots(len(years), 1, figsize=(8, 1.9 * len(years)), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, y in zip(axes, years):
        d = pred[pred.index.year == y]
        d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))  # show gaps
        ax.plot(d.index, d["observed"], color="black", lw=1.6, label="Observed")
        for col, c in (("XGBoost (proposed)", "#1f77b4"), ("Linear regression", "#999999"),
                       ("SWAT+ (calibrated)", "#d62728")):
            if col in d:
                ax.plot(d.index, d[col], lw=1.0, color=c, label=col)
        ax.set_ylabel("mm d⁻¹")
        ax.set_title(f"Spring {y}", fontsize=9, loc="left")
    axes[0].legend(fontsize=7, ncol=4, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_shap(shap_rank: pd.Series, shap_by_seed: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = shap_rank.sort_values()
    err = shap_by_seed[s.index].std(ddof=1)
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(s) + 1))
    ax.barh(s.index, s.values, xerr=err.values, color="#1f77b4")
    ax.set_xlabel("Mean |SHAP| (mm d⁻¹), summed over lags; error bar = SD over seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
