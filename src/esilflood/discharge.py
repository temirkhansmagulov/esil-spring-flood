"""Load and screen the Esil-Turgen daily discharge series (Section 7.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .timeutils import add_time_columns


def load_discharge_csv(path) -> pd.DataFrame:
    """Read a CSV with columns date, q_m3s and optionally ice (1 = ice-affected).

    Export the Kazhydromet series to this format first (see README, "Discharge data").
    """
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    if "date" not in cols or "q_m3s" not in cols:
        raise ValueError("Discharge CSV needs columns 'date' and 'q_m3s' (optional 'ice').")
    out = pd.DataFrame({
        "q_m3s": pd.to_numeric(df[cols["q_m3s"]], errors="coerce").values,
        "ice": (pd.to_numeric(df[cols["ice"]], errors="coerce").fillna(0).astype(int).values
                if "ice" in cols else 0),
    }, index=pd.to_datetime(df[cols["date"]], dayfirst=False).values)
    out.index.name = "date"
    out = out[~out.index.duplicated(keep="first")].sort_index()
    out.loc[out["q_m3s"] < 0, "q_m3s"] = np.nan
    return out


def screen_discharge(df: pd.DataFrame, start: str, end: str, max_gap_days: int = 7):
    """Apply the screening rules: interpolate gaps <= max_gap_days, remove longer gaps
    and ice-affected days. Returns the cleaned series and a removal report."""
    full = pd.date_range(start, end, freq="D")
    q = df["q_m3s"].reindex(full)
    ice = df["ice"].reindex(full).fillna(0).astype(int)
    report = {"days_in_period": len(full),
              "missing_raw": int(q.isna().sum()),
              "ice_days": int((ice == 1).sum())}

    q = q.mask(ice == 1)                      # ice days are treated as missing and removed
    isna = q.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    short_gap = isna & (run_len <= max_gap_days)
    long_gap = isna & (run_len > max_gap_days)
    interp = q.interpolate(method="time", limit_area="inside")
    q_clean = q.copy()
    q_clean[short_gap] = interp[short_gap]
    q_clean[ice == 1] = np.nan                # never fill ice days
    report["interpolated_days"] = int((short_gap & (ice == 0) & q_clean.notna()).sum())
    report["removed_long_gap_days"] = int((long_gap & (ice == 0)).sum())
    report["removed_ice_days"] = int((ice == 1).sum())
    report["valid_days"] = int(q_clean.notna().sum())
    out = pd.DataFrame({"q_m3s": q_clean, "ice": ice})
    out.index.name = "date"
    return out, report


def to_mm_per_day(q_m3s: pd.Series, area_km2: float) -> pd.Series:
    """Convert discharge (m3/s) to runoff depth (mm/day) over the basin area."""
    return q_m3s * 86400.0 / (area_km2 * 1e6) * 1000.0


def prepare_discharge(cfg) -> tuple[pd.DataFrame, dict]:
    p = cfg["period"]
    raw = load_discharge_csv(cfg.path(cfg["discharge"]["path"]))
    clean, report = screen_discharge(raw, p["start"], p["end"],
                                     cfg["discharge"]["max_gap_days"])
    clean["q_mm"] = to_mm_per_day(clean["q_m3s"], cfg["basin"]["area_km2"])
    clean = add_time_columns(clean, cfg)
    return clean, report
