"""Hydrological-year bookkeeping, spring window and train/validation/test split."""

from __future__ import annotations

import pandas as pd


def hydro_year(index: pd.DatetimeIndex, start_month: int = 9) -> pd.Index:
    """Hydrological year labelled by its starting calendar year (2001 = 2001/02)."""
    return pd.Index(index.year - (index.month < start_month).astype(int), name="hydro_year")


def hydro_year_label(year: int) -> str:
    return f"{year}/{str(year + 1)[-2:]}"


def spring_mask(index: pd.DatetimeIndex, start: str = "03-01", end: str = "05-31") -> pd.Series:
    md = index.strftime("%m-%d")
    return pd.Series((md >= start) & (md <= end), index=index)


def assign_split(years: pd.Index, split_cfg: dict) -> pd.Series:
    """Map each hydrological year to 'train', 'validation' or 'test' (or None)."""
    out = pd.Series(index=years, dtype="object")
    for name, (y0, y1) in split_cfg.items():
        out[(years >= y0) & (years <= y1)] = name
    return out


def add_time_columns(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Add hydro_year, split and spring columns to a daily DataFrame indexed by date."""
    p = cfg["period"]
    df = df.copy()
    df["hydro_year"] = hydro_year(df.index, p["hydro_year_start_month"]).values
    df["split"] = assign_split(pd.Index(df["hydro_year"]), p["split"]).values
    df["spring"] = spring_mask(df.index, p["spring_start"], p["spring_end"]).values
    return df
