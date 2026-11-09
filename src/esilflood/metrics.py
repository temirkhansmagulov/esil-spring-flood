"""Evaluation metrics (Eqs. 5-8) and the statistical tests used in Section 6.4."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


def _clean(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    ok = np.isfinite(obs) & np.isfinite(sim)
    return obs[ok], sim[ok]


def nse(obs, sim) -> float:
    o, s = _clean(obs, sim)
    if len(o) < 2 or np.var(o) == 0:
        return np.nan
    return 1.0 - np.sum((o - s) ** 2) / np.sum((o - o.mean()) ** 2)


def rmse(obs, sim) -> float:
    o, s = _clean(obs, sim)
    return float(np.sqrt(np.mean((o - s) ** 2))) if len(o) else np.nan


def kge(obs, sim) -> float:
    """Kling-Gupta efficiency, Gupta et al. (2009), Eq. 6."""
    o, s = _clean(obs, sim)
    if len(o) < 2 or o.std() == 0 or o.mean() == 0:
        return np.nan
    r = np.corrcoef(o, s)[0, 1] if s.std() > 0 else 0.0
    alpha = s.std() / o.std()
    beta = s.mean() / o.mean()
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def pbias(obs, sim) -> float:
    """Percent bias, Moriasi et al. (2007), Eq. 7; positive = underestimation."""
    o, s = _clean(obs, sim)
    return float(100.0 * np.sum(o - s) / np.sum(o)) if np.sum(o) != 0 else np.nan


def peak_errors(obs: pd.Series, sim: pd.Series, years: pd.Series) -> pd.DataFrame:
    """Relative peak error Ep (%) and peak timing error dtp (days) per year, Eq. 8."""
    df = pd.DataFrame({"obs": obs, "sim": sim, "year": years}).dropna()
    rows = []
    for y, g in df.groupby("year"):
        to, ts = g["obs"].idxmax(), g["sim"].idxmax()
        qo, qs = g.loc[to, "obs"], g.loc[ts, "sim"]
        rows.append({"year": y,
                     "Ep": 100.0 * (qs - qo) / qo if qo != 0 else np.nan,
                     "dtp": (ts - to).days,
                     "peak_obs": qo, "peak_sim": qs})
    return pd.DataFrame(rows).set_index("year")


def all_metrics(obs: pd.Series, sim: pd.Series, years: pd.Series) -> dict:
    pe = peak_errors(obs, sim, years)
    return {"NSE": nse(obs, sim), "KGE": kge(obs, sim), "PBIAS": pbias(obs, sim),
            "RMSE": rmse(obs, sim),
            "Ep": float(pe["Ep"].mean()) if len(pe) else np.nan,
            "dtp": float(pe["dtp"].mean()) if len(pe) else np.nan}


def block_bootstrap_ci(df: pd.DataFrame, stat: Callable[[pd.DataFrame], float],
                       year_col: str = "year", n: int = 1000, seed: int = 0,
                       level: float = 0.95) -> tuple[float, float]:
    """95% CI by resampling whole hydrological years with replacement."""
    rng = np.random.default_rng(seed)
    groups = {y: g for y, g in df.groupby(year_col)}
    years = np.array(list(groups))
    vals = []
    for _ in range(n):
        pick = rng.choice(years, size=len(years), replace=True)
        sample = pd.concat([groups[y].assign(**{year_col: f"{y}_{i}"})
                            for i, y in enumerate(pick)])
        v = stat(sample)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def bootstrap_ci_values(values, stat=np.median, n: int = 1000, seed: int = 0,
                        level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic of one value per year."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = [stat(rng.choice(v, size=len(v), replace=True)) for _ in range(n)]
    lo, hi = np.percentile(boots, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def wilcoxon_paired(a, b) -> float:
    """Two-sided exact Wilcoxon signed-rank p-value for paired samples."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2 or np.allclose(a, b):
        return np.nan
    method = "exact" if len(a) <= 50 else "auto"
    return float(stats.wilcoxon(a, b, alternative="two-sided", method=method).pvalue)


def spearman(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return (np.nan, np.nan)
    res = stats.spearmanr(a, b)
    return float(res.statistic), float(res.pvalue)
