#!/usr/bin/env python
"""Download the static layers for the bounding box in config.yaml (basin.era5_area).

  * Copernicus DEM GLO-30 1x1 degree tiles (public AWS bucket, free licence)
  * ESA WorldCover 2021 v200 3x3 degree tiles (public AWS bucket, CC BY 4.0)
  * SoilGrids 2.0 sand/silt/clay mean for 0-5, 5-15, 15-30 cm (ISRIC, CC BY 4.0),
    cut to the box from the global VRTs and saved in EPSG:4326

Usage:  python scripts/download_static.py [--only dem worldcover soilgrids]
"""

from __future__ import annotations

import argparse
import math
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esilflood.config import load_config  # noqa: E402

DEM_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif")
WC_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
          "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
SG_VRT = "https://files.isric.org/soilgrids/latest/data/{var}/{var}_{depth}_mean.vrt"


def _fetch(url: str, out: Path) -> None:
    if out.exists():
        print("exists ", out.name)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    print("fetch  ", url)
    try:
        urllib.request.urlretrieve(url, out)
    except Exception as e:                 # e.g. ocean tile that does not exist
        print("  skipped:", e)


def _ns(lat):
    return f"N{abs(lat):02d}" if lat >= 0 else f"S{abs(lat):02d}"


def _ew(lon, width=3):
    return f"E{abs(lon):0{width}d}" if lon >= 0 else f"W{abs(lon):0{width}d}"


def download_dem(cfg, n, w, s, e):
    out = cfg.path(cfg["static"]["dem_dir"])
    for lat in range(math.floor(s), math.ceil(n)):
        for lon in range(math.floor(w), math.ceil(e)):
            name = (f"Copernicus_DSM_COG_10_{_ns(lat)}_00_{_ew(lon)}_00_DEM")
            _fetch(DEM_URL.format(name=name), out / f"{name}.tif")


def download_worldcover(cfg, n, w, s, e):
    out = cfg.path(cfg["static"]["worldcover_dir"])
    for lat in range(math.floor(s / 3) * 3, math.ceil(n / 3) * 3, 3):
        for lon in range(math.floor(w / 3) * 3, math.ceil(e / 3) * 3, 3):
            tile = f"{_ns(lat)}{_ew(lon)}"
            _fetch(WC_URL.format(tile=tile), out / f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")


def download_soilgrids(cfg, n, w, s, e):
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import from_bounds

    out = cfg.path(cfg["static"]["soilgrids_dir"])
    out.mkdir(parents=True, exist_ok=True)
    res = 0.0025                            # ~250 m in degrees
    for var in ("sand", "silt", "clay"):
        for depth in ("0-5cm", "5-15cm", "15-30cm"):
            target = out / f"{var}_{depth}_mean.tif"
            if target.exists():
                print("exists ", target.name)
                continue
            url = "/vsicurl/" + SG_VRT.format(var=var, depth=depth)
            print("fetch  ", url)
            with rasterio.open(url) as src, WarpedVRT(src, crs="EPSG:4326",
                                                      resolution=res) as vrt:
                win = from_bounds(w, s, e, n, vrt.transform)
                data = vrt.read(1, window=win)
                meta = vrt.meta.copy()
                meta.update(driver="GTiff", height=data.shape[0], width=data.shape[1],
                            transform=vrt.window_transform(win), compress="deflate")
                with rasterio.open(target, "w", **meta) as dst:
                    dst.write(data, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="+", choices=["dem", "worldcover", "soilgrids"],
                    default=["dem", "worldcover", "soilgrids"])
    args = ap.parse_args()
    cfg = load_config()
    n, w, s, e = cfg["basin"]["era5_area"]
    if "dem" in args.only:
        download_dem(cfg, n, w, s, e)
    if "worldcover" in args.only:
        download_worldcover(cfg, n, w, s, e)
    if "soilgrids" in args.only:
        download_soilgrids(cfg, n, w, s, e)


if __name__ == "__main__":
    main()
