"""Synthetic demo inputs so the whole pipeline can be tested without real data.

Nothing produced from these files is a research result. The generator mimics the
shape of the real inputs (snowmelt-driven spring floods, steppe-dominated land cover)
so that every step of the code runs end to end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _write_raster(path: Path, arr, lon0, lat1, dlon, dlat, dtype, nodata=None):
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype=dtype, crs="EPSG:4326",
                       transform=from_origin(lon0, lat1, dlon, dlat), nodata=nodata) as dst:
        dst.write(arr.astype(dtype), 1)


def _smooth_field(rng, shape, scale=12):
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(rng.normal(size=shape), scale)


def make_demo_inputs(root: Path, seed: int = 1) -> None:
    """Create a demo data folder with the same layout the real pipeline expects."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    rng = np.random.default_rng(seed)
    raw = root / "data" / "raw"
    # --- basin: an irregular polygon of roughly 3,240 km2 near 51 N, 72.6 E
    ang = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    r = 1 + 0.15 * np.sin(3 * ang) + 0.08 * np.cos(5 * ang)
    rx, ry = 0.52 * r, 0.32 * r
    poly = Polygon(zip(72.6 + rx * np.cos(ang), 50.9 + ry * np.sin(ang)))
    gdf = gpd.GeoDataFrame({"name": ["Synthetic Upper Esil"]}, geometry=[poly], crs="EPSG:4326")
    area = gdf.to_crs("EPSG:32643").area.iloc[0] / 1e6
    poly = Polygon(zip(72.6 + rx * np.sqrt(3240 / area) * np.cos(ang),
                       50.9 + ry * np.sqrt(3240 / area) * np.sin(ang)))
    gdf = gpd.GeoDataFrame({"name": ["Synthetic Upper Esil"]}, geometry=[poly], crs="EPSG:4326")
    (raw / "basin").mkdir(parents=True, exist_ok=True)
    gdf.to_file(raw / "basin" / "upper_esil_basin.gpkg", driver="GPKG")

    # --- land cover (WorldCover codes), 0.002 deg (~200 m) for speed
    lon0, lat1, d = 71.7, 51.6, 0.002
    ny, nx = int(1.4 / d), int(1.8 / d)
    f1, f2 = _smooth_field(rng, (ny, nx)), _smooth_field(rng, (ny, nx), 6)
    lc = np.full((ny, nx), 30, dtype="uint8")                  # steppe grassland
    lc[f1 > np.quantile(f1, 0.70)] = 40                        # cropland ~30%
    lc[f2 > np.quantile(f2, 0.97)] = 20                        # shrubs
    lc[f2 < np.quantile(f2, 0.015)] = 10                       # sparse trees
    lc[(f1 < np.quantile(f1, 0.03))] = 60                      # bare / sparse
    lc[rng.random((ny, nx)) < 0.01] = 50                       # built-up / roads
    lc[(f2 > np.quantile(f2, 0.995))] = 80                     # water
    _write_raster(raw / "worldcover" / "demo_worldcover.tif", lc, lon0, lat1, d, d, "uint8", 0)

    # --- SoilGrids-like texture (g/kg) at three depths
    t = _smooth_field(rng, (ny, nx), 20)
    t = (t - t.min()) / (t.max() - t.min())
    for depth in ("0-5cm", "5-15cm", "15-30cm"):
        clay = 120 + 300 * t + rng.normal(0, 10, t.shape)
        sand = 520 - 380 * t + rng.normal(0, 10, t.shape)
        silt = 1000 - clay - sand
        for name, arr in (("sand", sand), ("silt", silt), ("clay", clay)):
            _write_raster(raw / "soilgrids" / f"{name}_{depth}_mean.tif",
                          np.clip(arr, 10, 900), lon0, lat1, d, d, "int16", -32768)

    # --- daily basin-mean forcing (stands in for processed ERA5-Land)
    days = pd.date_range("2001-09-01", "2024-08-31", freq="D")
    doy = days.dayofyear.values
    t2m = -1.5 - 19 * np.cos(2 * np.pi * (doy - 15) / 365.25) + rng.normal(0, 4, len(days))
    tp = np.where(rng.random(len(days)) < 0.28, rng.gamma(0.8, 3.0, len(days)), 0.0)
    snowfall = np.where(t2m < 0, tp, 0.0)
    swe = np.zeros(len(days))
    smlt = np.zeros(len(days))
    year_factor = rng.lognormal(0, 0.35, 30)
    years = (days.year - (days.month < 9)) - 2001
    for i in range(1, len(days)):
        melt = min(swe[i - 1], max(0.0, 3.5 * t2m[i])) if t2m[i] > 0 else 0.0
        swe[i] = swe[i - 1] + snowfall[i] * year_factor[years[i]] - melt
        smlt[i] = melt
    stl1 = 0.75 * t2m + 1.5
    swvl1 = np.clip(0.18 + 0.01 * pd.Series(smlt + tp).rolling(10, min_periods=1).sum().values
                    + rng.normal(0, 0.01, len(days)), 0.05, 0.45)
    forcing = pd.DataFrame({"tp_mm": tp, "smlt_mm": smlt, "sd_mm": swe, "t2m_c": t2m,
                            "swvl1": swvl1, "stl1_c": stl1}, index=days)
    forcing.index.name = "date"
    proc = root / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    forcing.to_csv(proc / "forcing_daily.csv")

    # --- discharge from a toy linear reservoir, with gaps and ice days
    def toy_q(cfac):
        inp = cfac * (smlt + 0.5 * tp) * (1 + 0.8 * (swvl1 - 0.2))
        q = np.zeros(len(days))
        for i in range(1, len(days)):
            q[i] = 0.93 * q[i - 1] + 0.07 * inp[i] + 0.02
        return q

    q_mm = toy_q(0.25) * rng.lognormal(0, 0.08, len(days))
    q_m3s = q_mm * 3240 * 1e6 / 1000 / 86400
    ice = ((days.month <= 3) | (days.month >= 12)) & (rng.random(len(days)) < 0.02)
    q_obs = q_m3s.copy()
    for start in rng.choice(len(days) - 20, 25, replace=False):
        q_obs[start:start + rng.integers(1, 15)] = np.nan
    (raw / "discharge").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "q_m3s": np.round(q_obs, 3),
                  "ice": ice.astype(int)}).to_csv(raw / "discharge" / "esil_turgen_daily.csv",
                                                  index=False)

    # --- SWAT+ style outputs for the three scenarios
    for name, fac in (("baseline", 0.24), ("degradation", 0.27), ("recovery", 0.22)):
        q = toy_q(fac) * 3240 * 1e6 / 1000 / 86400
        out = root / "data" / "swat" / name
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "channel_sd_day.txt", "w") as f:
            f.write("channel_sd_day: synthetic demo output\n")
            f.write(f"{'jday':>6}{'mon':>6}{'day':>6}{'yr':>6}{'unit':>6}{'gis_id':>8}"
                    f"{'name':>10}{'flo_out':>14}\n")
            f.write(f"{'':>6}{'':>6}{'':>6}{'':>6}{'':>6}{'':>8}{'':>10}{'m^3/s':>14}\n")
            for dt, v in zip(days, q):
                f.write(f"{dt.dayofyear:>6}{dt.month:>6}{dt.day:>6}{dt.year:>6}{1:>6}{1:>8}"
                        f"{'cha001':>10}{v:>14.4f}\n")
    pd.DataFrame({"variable": ["smlt_mm", "swvl1", "stl1_c"],
                  "delta_peak_pct": [18.0, 6.5, 3.0]}).to_csv(
        root / "data" / "swat" / "perturbation_sensitivity.csv", index=False)
