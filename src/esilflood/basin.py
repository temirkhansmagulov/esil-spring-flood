"""Basin polygon: load an existing outline or delineate it from the Copernicus DEM."""

from __future__ import annotations

from pathlib import Path


def mosaic_dem(dem_dir: Path, out_path: Path) -> Path:
    """Merge Copernicus DEM GLO-30 tiles into one GeoTIFF."""
    import rasterio
    from rasterio.merge import merge

    tiles = sorted(Path(dem_dir).glob("*.tif"))
    if not tiles:
        raise FileNotFoundError(f"No DEM tiles in {dem_dir}")
    srcs = [rasterio.open(t) for t in tiles]
    data, transform = merge(srcs)
    meta = srcs[0].meta.copy()
    meta.update(height=data.shape[1], width=data.shape[2], transform=transform,
                compress="deflate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(data)
    for s in srcs:
        s.close()
    return out_path


def delineate_basin(dem_path: Path, gauge_lon: float, gauge_lat: float,
                    snap_acc_threshold: int = 5000):
    """Delineate the catchment above the gauge with pysheds (D8 routing).

    The gauge is snapped to the nearest cell with flow accumulation above the
    threshold. Always compare the resulting area with the published 3,240 km2.
    """
    import numpy as np
    import shapely
    from pysheds.grid import Grid

    if not hasattr(np, "in1d"):          # pysheds 0.5 still calls np.in1d (removed in NumPy 2.4)
        np.in1d = np.isin

    grid = Grid.from_raster(str(dem_path))
    dem = grid.read_raster(str(dem_path))
    dem = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem)))
    fdir = grid.flowdir(dem)
    acc = grid.accumulation(fdir)
    x, y = grid.snap_to_mask(acc > snap_acc_threshold, (gauge_lon, gauge_lat))
    catch = grid.catchment(x=x, y=y, fdir=fdir, xytype="coordinate")
    grid.clip_to(catch)
    shapes = [shapely.geometry.shape(s) for s, v in grid.polygonize() if v]
    return shapely.unary_union(shapes)


def load_or_delineate(cfg):
    """Return (polygon in EPSG:4326, area_km2). Saves a delineated polygon to disk."""
    import geopandas as gpd

    poly_path = cfg.path(cfg["basin"]["polygon_path"])
    if poly_path and poly_path.exists():
        gdf = gpd.read_file(poly_path).to_crs("EPSG:4326")
        polygon = gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union
    else:
        b = cfg["basin"]
        if b["gauge_lon"] is None or b["gauge_lat"] is None:
            raise ValueError("Set basin.gauge_lon/gauge_lat in config.yaml or provide "
                             "basin.polygon_path.")
        dem = mosaic_dem(cfg.path(cfg["static"]["dem_dir"]),
                         cfg.processed / "static" / "dem_mosaic.tif")
        polygon = delineate_basin(dem, b["gauge_lon"], b["gauge_lat"])
        gdf = gpd.GeoDataFrame({"name": [b["name"]]}, geometry=[polygon], crs="EPSG:4326")
        poly_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(poly_path, driver="GPKG")
    area = gpd.GeoSeries([polygon], crs="EPSG:4326").to_crs(
        cfg["basin"]["projected_crs"]).area.iloc[0] / 1e6
    return polygon, float(area)
