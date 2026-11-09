"""Unit tests for the formulas and data handling. Run with: pytest -q"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esilflood import metrics as M  # noqa: E402
from esilflood.config import load_config  # noqa: E402
from esilflood.discharge import screen_discharge, to_mm_per_day  # noqa: E402
from esilflood.rq2 import apply_conversions  # noqa: E402
from esilflood.static_layers import (cn_amc, hsg_from_hysogs, hsg_from_ks,  # noqa: E402
                                     scs_runoff, usda_texture)
from esilflood.timeutils import hydro_year, spring_mask  # noqa: E402

CFG = load_config()


# ---------------------------------------------------------------- metrics
def test_perfect_model_scores():
    o = np.array([1.0, 3.0, 2.0, 5.0])
    assert M.nse(o, o) == pytest.approx(1.0)
    assert M.kge(o, o) == pytest.approx(1.0)
    assert M.pbias(o, o) == pytest.approx(0.0)
    assert M.rmse(o, o) == pytest.approx(0.0)


def test_nse_of_mean_is_zero():
    o = np.array([1.0, 2.0, 3.0, 4.0])
    assert M.nse(o, np.full(4, o.mean())) == pytest.approx(0.0)


def test_pbias_sign_positive_means_underestimation():
    o = np.array([10.0, 10.0])
    assert M.pbias(o, o * 0.8) == pytest.approx(20.0)


def test_peak_errors():
    idx = pd.date_range("2020-03-01", periods=5)
    obs = pd.Series([1, 5, 3, 2, 1.0], idx)
    sim = pd.Series([1, 2, 4, 2, 1.0], idx)
    pe = M.peak_errors(obs, sim, pd.Series(2019, idx))
    assert pe.loc[2019, "Ep"] == pytest.approx(-20.0)
    assert pe.loc[2019, "dtp"] == 1


def test_wilcoxon_minimum_p_with_seven_years():
    # Exact two-sided test: smallest attainable p with n = 7 is 2/128 = 0.0156.
    a = np.arange(1, 8, dtype=float)
    assert M.wilcoxon_paired(a + 1, a) == pytest.approx(0.015625)


# ---------------------------------------------------------------- SCS-CN
def test_scs_runoff_values():
    # CN 80: S = 63.5 mm, Ia = 12.7 mm; P = 50 mm -> Q = 37.3^2 / 100.8 = 13.80 mm
    assert float(scs_runoff(50, 80)) == pytest.approx(13.80, abs=0.01)
    assert float(scs_runoff(10, 80)) == 0.0          # below initial abstraction


def test_amc_adjustment():
    assert float(cn_amc(80, 1)) == pytest.approx(4.2 * 80 / (10 - 0.058 * 80))
    assert float(cn_amc(80, 3)) == pytest.approx(23 * 80 / (10 + 0.13 * 80))
    assert float(cn_amc(80, 2)) == 80


# ---------------------------------------------------------------- soils
@pytest.mark.parametrize("sand,silt,clay,name", [
    (92, 4, 4, "sand"), (40, 40, 20, "loam"), (20, 65, 15, "silt loam"),
    (30, 35, 35, "clay loam"), (20, 20, 60, "clay"), (65, 20, 15, "sandy loam"),
])
def test_usda_texture(sand, silt, clay, name):
    assert usda_texture([sand], [silt], [clay])[0] == name


def test_hsg_from_ks_matches_texture_mapping():
    t = CFG["hsg_ks_thresholds_cm_h"]
    ks = CFG["ks_texture_cm_h"]
    got = {k: int(hsg_from_ks(np.array([v]), t)[0]) for k, v in ks.items()}
    assert got["sand"] == got["loamy sand"] == got["sandy loam"] == 1
    assert got["loam"] == got["silt loam"] == 2
    assert got["sandy clay loam"] == 3
    assert got["clay"] == got["clay loam"] == 4


def test_hysogs_dual_groups_are_d():
    assert list(hsg_from_hysogs(np.array([1, 2, 3, 4, 11, 14]))) == [1, 2, 3, 4, 4, 4]


# ---------------------------------------------------------------- time and discharge
def test_hydro_year_and_spring():
    idx = pd.DatetimeIndex(["2001-08-31", "2001-09-01", "2002-03-15", "2002-06-01"])
    assert list(hydro_year(idx)) == [2000, 2001, 2001, 2001]
    assert list(spring_mask(idx)) == [False, False, True, False]


def test_discharge_screening_rules():
    idx = pd.date_range("2010-01-01", periods=40)
    q = pd.Series(10.0, idx)
    q.iloc[5:8] = np.nan          # 3-day gap -> interpolated
    q.iloc[15:25] = np.nan        # 10-day gap -> removed
    ice = pd.Series(0, idx)
    ice.iloc[30] = 1              # ice day -> removed
    out, rep = screen_discharge(pd.DataFrame({"q_m3s": q, "ice": ice}),
                                str(idx[0].date()), str(idx[-1].date()), 7)
    assert out["q_m3s"].iloc[5:8].notna().all()
    assert out["q_m3s"].iloc[15:25].isna().all()
    assert np.isnan(out["q_m3s"].iloc[30])
    assert rep["interpolated_days"] == 3
    assert rep["removed_long_gap_days"] == 10
    assert rep["removed_ice_days"] == 1


def test_mm_conversion():
    # 37.5 m3/s over 3,240 km2 is 1 mm/day
    assert to_mm_per_day(37.5, 3240) == pytest.approx(1.0)


# ---------------------------------------------------------------- scenarios
def test_scenario_fractions():
    rng = np.random.default_rng(0)
    lc = np.array([40] * 1000 + [30] * 1000)
    rec = apply_conversions(lc, [{"from": 40, "to": 30, "fraction": 0.30},
                                 {"from": 40, "to": 10, "fraction": 0.05}], rng)
    assert (rec == 40).sum() == 650
    assert (rec == 10).sum() == 50
    assert (rec == 30).sum() == 1300


# ---------------------------------------------------------------- ERA5-Land
def test_hourly_to_daily_accumulations():
    xr = pytest.importorskip("xarray")
    from esilflood.era5 import hourly_to_daily

    t = pd.date_range("2020-01-01", periods=49, freq="h")
    # accumulated precipitation: 0.1 mm per hour, reset after 00 UTC
    hours = np.array([(ts.hour if ts.hour else 24) for ts in t]) * 1e-4
    hours[0] = 0.0
    ds = xr.Dataset({"tp": (("valid_time", "latitude", "longitude"), hours[:, None, None]),
                     "t2m": (("valid_time", "latitude", "longitude"),
                             np.full((49, 1, 1), 273.15))},
                    coords={"valid_time": t, "latitude": [51.0], "longitude": [72.0]})
    d = hourly_to_daily(ds)
    tp = d["tp"].sel(time="2020-01-01").item()
    assert tp == pytest.approx(2.4)            # 24 h x 0.1 mm
    assert d["t2m"].sel(time="2020-01-01").item() == pytest.approx(0.0)
