"""Download ERA5-Land hourly data and convert it to daily basin averages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Short names in the ERA5-Land NetCDF files.
SHORT = {
    "total_precipitation": "tp",
    "snowmelt": "smlt",
    "snow_depth_water_equivalent": "sd",
    "2m_temperature": "t2m",
    "volumetric_soil_water_layer_1": "swvl1",
    "volumetric_soil_water_layer_2": "swvl2",
    "volumetric_soil_water_layer_3": "swvl3",
    "volumetric_soil_water_layer_4": "swvl4",
    "soil_temperature_level_1": "stl1",
    "soil_temperature_level_2": "stl2",
    "soil_temperature_level_3": "stl3",
    "soil_temperature_level_4": "stl4",
}
ACCUMULATED = {"tp", "smlt"}           # accumulated from 00 UTC in ERA5-Land


def download_era5_land(cfg, overwrite: bool = False) -> list[Path]:
    """Request ERA5-Land hourly data month by month from the Copernicus CDS.

    Needs a CDS account and ~/.cdsapirc (see README). Files are skipped if present.
    """
    import cdsapi

    out_dir = cfg.path(cfg["era5"]["raw_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(cfg["period"]["start"])
    # One extra day so the last day's accumulation (at next-day 00 UTC) is available.
    end = pd.Timestamp(cfg["period"]["end"]) + pd.Timedelta(days=1)
    client = cdsapi.Client()
    files = []
    for month in pd.period_range(start, end, freq="M"):
        target = out_dir / f"era5land_{month.year}_{month.month:02d}.nc"
        files.append(target)
        if target.exists() and not overwrite:
            continue
        request = {
            "variable": cfg["era5"]["variables"],
            "year": str(month.year),
            "month": f"{month.month:02d}",
            "day": [f"{d:02d}" for d in range(1, month.days_in_month + 1)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": cfg["basin"]["era5_area"],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        client.retrieve("reanalysis-era5-land", request, str(target))
    return files


def _time_name(ds) -> str:
    for name in ("valid_time", "time"):
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError("No time coordinate in ERA5-Land file")


def hourly_to_daily(ds):
    """Daily totals for accumulated variables (value at next-day 00 UTC) and daily means
    for state variables. Units: tp/smlt/sd in mm, temperatures in deg C."""
    import xarray as xr

    t = _time_name(ds)
    ds = ds.rename({t: "time"}) if t != "time" else ds
    for extra in ("number", "expver"):
        if extra in ds.coords and extra not in ds.dims:
            ds = ds.drop_vars(extra)
    out = {}
    for var in ds.data_vars:
        da = ds[var]
        if var in ACCUMULATED:
            midnight = da.where(da["time"].dt.hour == 0, drop=True)
            midnight = midnight.assign_coords(time=midnight["time"] - np.timedelta64(1, "D"))
            daily = midnight * 1000.0                          # m -> mm
            daily = daily.clip(min=0)
        else:
            daily = da.resample(time="1D").mean()
            if var == "sd":
                daily = daily * 1000.0                         # m w.e. -> mm
            elif var in ("t2m", "stl1", "stl2", "stl3", "stl4"):
                daily = daily - 273.15                         # K -> deg C
        daily["time"] = daily["time"].dt.floor("D")
        out[var] = daily
    return xr.Dataset(out)


def basin_weights(lat: np.ndarray, lon: np.ndarray, polygon, supersample: int = 10):
    """Fraction of each grid cell inside the basin polygon times cos(latitude)."""
    import shapely

    dlat = abs(float(lat[1] - lat[0])) if len(lat) > 1 else 0.1
    dlon = abs(float(lon[1] - lon[0])) if len(lon) > 1 else 0.1
    offs = (np.arange(supersample) + 0.5) / supersample - 0.5
    w = np.zeros((len(lat), len(lon)))
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            xs, ys = np.meshgrid(lo + offs * dlon, la + offs * dlat)
            inside = shapely.contains_xy(polygon, xs.ravel(), ys.ravel())
            w[i, j] = inside.mean()
    w *= np.cos(np.deg2rad(lat))[:, None]
    return w


def basin_average(daily, polygon) -> tuple[pd.DataFrame, int]:
    """Area-weighted basin mean of every daily variable, as a DataFrame."""
    lat_name = "latitude" if "latitude" in daily.dims else "lat"
    lon_name = "longitude" if "longitude" in daily.dims else "lon"
    lat, lon = daily[lat_name].values, daily[lon_name].values
    w = basin_weights(lat, lon, polygon)
    if w.sum() == 0:
        raise ValueError("Basin polygon does not overlap the ERA5-Land grid; check era5_area.")
    import xarray as xr

    wda = xr.DataArray(w, coords={lat_name: lat, lon_name: lon}, dims=(lat_name, lon_name))
    cols = {}
    for var in daily.data_vars:
        da = daily[var]
        valid = da.notnull()
        num = (da.fillna(0) * wda).sum((lat_name, lon_name))
        den = (valid * wda).sum((lat_name, lon_name))
        cols[var] = (num / den).to_series()
    df = pd.DataFrame(cols)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df, int((w > 0).sum())


RENAME = {"tp": "tp_mm", "smlt": "smlt_mm", "sd": "sd_mm", "t2m": "t2m_c",
          "stl1": "stl1_c", "stl2": "stl2_c", "stl3": "stl3_c", "stl4": "stl4_c"}


def process_era5_land(cfg, polygon) -> tuple[pd.DataFrame, int]:
    """Read all monthly NetCDF files, convert to daily and average over the basin."""
    import xarray as xr

    files = sorted(cfg.path(cfg["era5"]["raw_dir"]).glob("era5land_*.nc"))
    if not files:
        raise FileNotFoundError("No ERA5-Land files found; run the download step first.")
    ds = xr.open_mfdataset([str(f) for f in files], combine="by_coords")
    daily = hourly_to_daily(ds).load()
    df, n_cells = basin_average(daily, polygon)
    df = df.rename(columns=RENAME)
    df = df.loc[cfg["period"]["start"]:cfg["period"]["end"]]
    return df, n_cells
