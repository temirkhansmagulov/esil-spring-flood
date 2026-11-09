"""Static layers on a common grid: land cover, soil texture, Ks, hydrologic soil group
and curve number (Section 6.2, Eqs. 1-4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HSG_CODES = {1: "A", 2: "B", 3: "C", 4: "D"}
HSG_LETTER = {v: k for k, v in HSG_CODES.items()}

TEXTURES = ["sand", "loamy sand", "sandy loam", "loam", "silt loam", "silt",
            "sandy clay loam", "clay loam", "silty clay loam", "sandy clay",
            "silty clay", "clay"]


# ---------------------------------------------------------------- grid helpers
def target_grid(polygon_ll, crs: str, res: float):
    """Transform, shape and basin mask of a projected grid covering the basin."""
    import geopandas as gpd
    from rasterio import features
    from rasterio.transform import from_origin

    poly = gpd.GeoSeries([polygon_ll], crs="EPSG:4326").to_crs(crs).iloc[0]
    minx, miny, maxx, maxy = poly.bounds
    minx, miny = np.floor(minx / res) * res - res, np.floor(miny / res) * res - res
    maxx, maxy = np.ceil(maxx / res) * res + res, np.ceil(maxy / res) * res + res
    width, height = int(round((maxx - minx) / res)), int(round((maxy - miny) / res))
    transform = from_origin(minx, maxy, res, res)
    mask = features.geometry_mask([poly], (height, width), transform, invert=True)
    return transform, (height, width), mask


def warp_to_grid(paths, transform, shape, crs, resampling: str, dtype="float32",
                 nodata=np.nan):
    """Reproject one or more source rasters (e.g. tiles) onto the target grid."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    out = np.full(shape, nodata, dtype=dtype)
    for p in paths:
        with rasterio.open(p) as src:
            tmp = np.full(shape, nodata, dtype=dtype)
            reproject(source=rasterio.band(src, 1), destination=tmp,
                      dst_transform=transform, dst_crs=crs, dst_nodata=nodata,
                      resampling=getattr(Resampling, resampling))
            if isinstance(nodata, float) and np.isnan(nodata):
                fill = np.isnan(out) & ~np.isnan(tmp)
            else:
                fill = (out == nodata) & (tmp != nodata)
            out[fill] = tmp[fill]
    return out


def write_tif(path: Path, arr, transform, crs, nodata=None):
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype=arr.dtype, crs=crs, transform=transform,
                       nodata=nodata, compress="deflate") as dst:
        dst.write(arr, 1)


# ---------------------------------------------------------------- soil science
def usda_texture(sand, silt, clay):
    """USDA texture class names from sand/silt/clay percentages (arrays)."""
    sand, silt, clay = (np.asarray(a, float) for a in (sand, silt, clay))
    conds = [
        silt + 1.5 * clay < 15,
        (silt + 1.5 * clay >= 15) & (silt + 2 * clay < 30),
        ((clay >= 7) & (clay < 20) & (sand > 52) & (silt + 2 * clay >= 30))
        | ((clay < 7) & (silt < 50) & (silt + 2 * clay >= 30)),
        (clay >= 7) & (clay < 27) & (silt >= 28) & (silt < 50) & (sand <= 52),
        ((silt >= 50) & (clay >= 12) & (clay < 27)) | ((silt >= 50) & (silt < 80) & (clay < 12)),
        (silt >= 80) & (clay < 12),
        (clay >= 20) & (clay < 35) & (silt < 28) & (sand > 45),
        (clay >= 27) & (clay < 40) & (sand > 20) & (sand <= 45),
        (clay >= 27) & (clay < 40) & (sand <= 20),
        (clay >= 35) & (sand > 45),
        (clay >= 40) & (silt >= 40),
        (clay >= 40) & (sand <= 45) & (silt < 40),
    ]
    out = np.select(conds, TEXTURES, default="loam").astype(object)
    out[~(np.isfinite(sand) & np.isfinite(silt) & np.isfinite(clay))] = None
    return out


def ks_from_texture(texture, ks_table: dict):
    ks = np.full(texture.shape, np.nan)
    for name, val in ks_table.items():
        ks[texture == name] = val
    return ks


def hsg_from_ks(ks, thresholds: dict):
    """HSG code 1-4 (A-D) from Ks (cm/h) using the configured thresholds."""
    hsg = np.full(ks.shape, 0, dtype="uint8")
    ok = np.isfinite(ks)
    hsg[ok] = 4
    hsg[ok & (ks >= thresholds["C"])] = 3
    hsg[ok & (ks >= thresholds["B"])] = 2
    hsg[ok & (ks >= thresholds["A"])] = 1
    return hsg


def hsg_from_hysogs(arr):
    """HYSOGs250m codes: 1-4 = A-D; dual groups 11-14 (A/D..D/D) taken as D (undrained)."""
    a = np.nan_to_num(np.asarray(arr, float), nan=0).astype(int)
    out = np.zeros(a.shape, dtype="uint8")
    for c in (1, 2, 3, 4):
        out[a == c] = c
    out[np.isin(a, [11, 12, 13, 14])] = 4
    return out


def cn_grid(landcover, hsg, cn_table: dict):
    """AMC II curve number for every cell from land cover and HSG."""
    cn = np.full(landcover.shape, np.nan, dtype="float32")
    for lc, row in cn_table.items():
        for code, letter in HSG_CODES.items():
            cn[(landcover == int(lc)) & (hsg == code)] = row[letter]
    return cn


def cn_amc(cn2, amc: int | np.ndarray):
    """Antecedent moisture adjustment from AMC II (Eq. 4). amc in {1, 2, 3}."""
    cn2 = np.asarray(cn2, float)
    cn1 = 4.2 * cn2 / (10 - 0.058 * cn2)
    cn3 = 23 * cn2 / (10 + 0.13 * cn2)
    return np.where(amc == 1, cn1, np.where(amc == 3, cn3, cn2))


def scs_runoff(p_mm, cn):
    """Direct runoff depth Q (mm) by the SCS-CN method (Eqs. 1-2)."""
    p = np.asarray(p_mm, float)
    cn = np.clip(np.asarray(cn, float), 1, 100)
    s = 25400.0 / cn - 254.0
    ia = 0.2 * s
    return np.where(p > ia, (p - ia) ** 2 / (p - ia + s), 0.0)


# ---------------------------------------------------------------- pipeline step
def _tifs(d) -> list[Path]:
    return sorted(Path(d).glob("*.tif")) if d and Path(d).exists() else []


def _soilgrids_mean(cfg, var, transform, shape, crs):
    """Thickness-weighted 0-30 cm mean of a SoilGrids property, converted to percent."""
    d = cfg.path(cfg["static"]["soilgrids_dir"])
    layers = [("0-5cm", 5), ("5-15cm", 10), ("15-30cm", 15)]
    acc, wsum = np.zeros(shape), np.zeros(shape)
    for depth, w in layers:
        f = d / f"{var}_{depth}_mean.tif"
        if not f.exists():
            raise FileNotFoundError(f"Missing SoilGrids file {f}")
        a = warp_to_grid([f], transform, shape, crs, "bilinear")
        a = np.where(a > 0, a / 10.0, np.nan)            # g/kg -> %
        ok = np.isfinite(a)
        acc[ok] += a[ok] * w
        wsum[ok] += w
    with np.errstate(invalid="ignore"):
        return np.where(wsum > 0, acc / wsum, np.nan)


def build_static_layers(cfg, polygon_ll) -> dict:
    """Build and save land cover, texture, Ks, HSG and CN grids clipped to the basin."""
    crs = cfg["basin"]["projected_crs"]
    res = cfg["static"]["target_resolution_m"]
    transform, shape, mask = target_grid(polygon_ll, crs, res)
    out_dir = cfg.processed / "static"

    wc = warp_to_grid(_tifs(cfg.path(cfg["static"]["worldcover_dir"])), transform, shape,
                      crs, "mode", dtype="uint8", nodata=0)
    wc[~mask] = 0

    sand = silt = clay = None
    ks = np.full(shape, np.nan)
    if _tifs(cfg.path(cfg["static"]["soilgrids_dir"])):
        sand = _soilgrids_mean(cfg, "sand", transform, shape, crs)
        silt = _soilgrids_mean(cfg, "silt", transform, shape, crs)
        clay = _soilgrids_mean(cfg, "clay", transform, shape, crs)
        tex = usda_texture(sand, silt, clay)
        ks = ks_from_texture(tex, cfg["ks_texture_cm_h"])

    hys = cfg["static"].get("hysogs_path")
    if hys and cfg.path(hys).exists():
        hsg = hsg_from_hysogs(warp_to_grid([cfg.path(hys)], transform, shape, crs, "nearest"))
    else:
        hsg = hsg_from_ks(ks, cfg["hsg_ks_thresholds_cm_h"])
    hsg[~mask] = 0
    ks[~mask] = np.nan
    cn2 = cn_grid(wc, hsg, cfg["curve_numbers"])
    cn2[~mask] = np.nan

    write_tif(out_dir / "landcover.tif", wc, transform, crs, nodata=0)
    write_tif(out_dir / "hsg.tif", hsg, transform, crs, nodata=0)
    write_tif(out_dir / "ks_cm_h.tif", ks.astype("float32"), transform, crs, nodata=np.nan)
    write_tif(out_dir / "cn2.tif", cn2, transform, crs, nodata=np.nan)

    layers = {"landcover": wc, "hsg": hsg, "ks": ks, "cn2": cn2, "mask": mask,
              "transform": transform, "crs": crs}
    gcn = cfg["static"].get("gcn250_path")
    if gcn and cfg.path(gcn).exists():
        layers["gcn250"] = warp_to_grid([cfg.path(gcn)], transform, shape, crs, "nearest")
    return layers


def class_statistics(layers: dict, cfg) -> pd.DataFrame:
    """Area share, dominant HSG, median Ks and area-weighted CN_II per land-cover class."""
    lc, hsg, ks, cn2, mask = (layers[k] for k in ("landcover", "hsg", "ks", "cn2", "mask"))
    valid = mask & (lc > 0)
    total = valid.sum()
    rows = []
    for code, row in cfg["curve_numbers"].items():
        m = valid & (lc == int(code))
        n = int(m.sum())
        if n == 0:
            continue
        h = hsg[m]
        h = h[h > 0]
        dom = HSG_CODES[int(np.bincount(h).argmax())] if len(h) else "NA"
        dom_mask = m & (hsg == HSG_LETTER.get(dom, -1))
        rows.append({"code": int(code), "class": row["name"], "cells": n,
                     "area_pct": 100.0 * n / total,
                     "dominant_hsg": dom,
                     "ks_cm_h": float(np.nanmedian(ks[dom_mask])) if np.isfinite(ks[dom_mask]).any() else np.nan,
                     "cn2_mean": float(np.nanmean(cn2[m]))})
    df = pd.DataFrame(rows).set_index("code")
    if "gcn250" in layers:
        g = layers["gcn250"]
        df["gcn250_mean"] = [float(np.nanmean(np.where(g > 0, g, np.nan)[valid & (lc == c)]))
                             for c in df.index]
    return df
