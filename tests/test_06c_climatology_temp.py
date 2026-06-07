"""Offline tests for Script 06c (ERA5-Land temperature climatology).

No network: CDS request schema, Hargreaves Ra/PET, the circular per-DOY pooling,
hourly->daily aggregation, and checkpoint round-trip.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "06c_climatology_temp.py"


@pytest.fixture(scope="module")
def mod():
    if not SCRIPT_PATH.exists():
        pytest.skip(f"script not found at {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("clim_temp", SCRIPT_PATH)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["clim_temp"] = m
    spec.loader.exec_module(m)
    return m


# ---- CDS request schema ----
class TestRequest:
    def test_request_shape(self, mod):
        req = mod.build_request(2003, 7, mod.DEFAULT_AREA)
        assert req["variable"] == ["2m_temperature"]
        assert req["year"] == "2003"
        assert req["month"] == "07"
        assert len(req["time"]) == 24
        assert req["data_format"] == "netcdf"

    def test_request_area_is_n_w_s_e(self, mod):
        req = mod.build_request(2003, 1, (37.0, 22.0, 30.0, 36.0))
        assert req["area"] == [37.0, 22.0, 30.0, 36.0]


# ---- Hargreaves radiation + PET ----
class TestHargreaves:
    def test_ra_equator_magnitude(self, mod):
        # Ra at the equator is ~36-38 MJ/m2/day year-round.
        ra = mod.hargreaves_ra(0.0, 80)
        assert 30.0 < float(ra) < 42.0

    def test_ra_accepts_array(self, mod):
        ra = mod.hargreaves_ra(np.array([0.0, 30.0, 60.0]), 172)
        assert ra.shape == (3,)
        assert np.all(ra > 0)

    def test_pet_zero_when_no_diurnal_range(self, mod):
        # sqrt(Tmax - Tmin) = 0 -> PET 0.
        pet = mod.hargreaves_pet(25.0, 25.0, 35.0)
        assert float(pet) == pytest.approx(0.0)

    def test_pet_positive_and_increases_with_heat(self, mod):
        cool = mod.hargreaves_pet(20.0, 10.0, 35.0)
        hot = mod.hargreaves_pet(40.0, 30.0, 35.0)
        assert float(cool) > 0
        assert float(hot) > float(cool)


# ---- per-DOY windowed pooling ----
class TestWindowedStats:
    def test_window1_is_plain_per_doy(self, mod):
        # Two observations on one DOY: mean and population std.
        s = np.zeros((3, 1, 1))
        sq = np.zeros((3, 1, 1))
        c = np.zeros((3, 1, 1))
        s[0, 0, 0] = 10.0 + 20.0
        sq[0, 0, 0] = 100.0 + 400.0
        c[0, 0, 0] = 2.0
        mean, std = mod.windowed_doy_stats(s, sq, c, window=1)
        assert mean[0, 0, 0] == pytest.approx(15.0)
        assert std[0, 0, 0] == pytest.approx(5.0)  # pop(10,20)=5

    def test_window3_pools_circularly(self, mod):
        s = np.array([1.0, 2.0, 3.0]).reshape(3, 1, 1)
        sq = s**2
        c = np.ones((3, 1, 1))
        mean, _ = mod.windowed_doy_stats(s, sq, c, window=3)
        # window 3 over 3 circular DOYs pools all -> mean (1+2+3)/3 everywhere
        assert np.allclose(mean[:, 0, 0], 2.0)

    def test_empty_doy_is_nan(self, mod):
        s = np.zeros((3, 1, 1))
        mean, std = mod.windowed_doy_stats(s, s.copy(), np.zeros((3, 1, 1)), window=1)
        assert np.isnan(mean[0, 0, 0])


# ---- hourly -> daily ----
class TestHourlyToDaily:
    def test_daily_max_min_in_celsius(self, mod):
        xr = pytest.importorskip("xarray")
        import pandas as pd

        # 2 days x 24 hours, one pixel; K values with a known daily swing.
        times = pd.date_range("2020-01-01", periods=48, freq="h")
        kelvin = np.concatenate(
            [273.15 + 10 + 5 * np.sin(np.linspace(0, np.pi, 24)) for _ in range(2)]
        )
        ds = xr.Dataset(
            {"t2m": (("valid_time", "latitude", "longitude"), kelvin[:, None, None])},
            coords={"valid_time": times, "latitude": [31.0], "longitude": [30.0]},
        )
        tmax, tmin = mod.hourly_to_daily(ds)
        assert tmax.shape[0] == 2
        # max of 10 + 5*sin over [0,pi] peaks near 15 degC; min is 10 degC.
        d0_max = float(tmax.isel(valid_time=0).values.squeeze())
        d0_min = float(tmin.isel(valid_time=0).values.squeeze())
        assert d0_max == pytest.approx(15.0, abs=0.05)
        assert d0_min == pytest.approx(10.0, abs=1e-3)


# ---- checkpoint ----
class TestCheckpoint:
    def test_roundtrip(self, mod, tmp_path):
        path = mod.ckpt_path(tmp_path)
        acc = {"doy_sum": np.ones((3, 2, 2)), "pet_labels": np.array([199101, 199102])}
        mod._save_checkpoint(path, acc, {1991})
        acc2, done = mod._load_checkpoint(path)
        assert done == {1991}
        assert np.array_equal(acc2["doy_sum"], acc["doy_sum"])
        assert np.array_equal(acc2["pet_labels"], acc["pet_labels"])

    def test_missing_is_empty(self, mod, tmp_path):
        acc, done = mod._load_checkpoint(tmp_path / "nope.npz")
        assert acc == {}
        assert done == set()


# ---- fold one month end-to-end ----
class TestFoldMonth:
    def test_fold_populates_accumulators(self, mod, tmp_path):
        xr = pytest.importorskip("xarray")
        pytest.importorskip("h5netcdf")
        import pandas as pd

        # 3 days x 24 h, 2x2 grid, Jan 2005; K with a daily swing.
        hours = 3 * 24
        times = pd.date_range("2005-01-01", periods=hours, freq="h")
        base = 273.15 + 15 + 5 * np.sin(np.linspace(0, 6 * np.pi, hours))
        data = np.broadcast_to(base[:, None, None], (hours, 2, 2)).astype("float64")
        ds = xr.Dataset(
            {"t2m": (("valid_time", "latitude", "longitude"), data)},
            coords={
                "valid_time": times,
                "latitude": [31.0, 31.1],
                "longitude": [30.0, 30.1],
            },
        )
        raw = tmp_path / "era5land_t2m_2005-01.nc"
        ds.to_netcdf(raw, engine="h5netcdf")

        acc, lat_grid = mod._fold_month(raw, {}, None, 2005, 1, np, xr)
        # DOYs 1,2,3 each saw one daily value -> count 1 on those, 0 elsewhere.
        assert acc["doy_count"][0, 0, 0] == 1
        assert acc["doy_count"][2, 0, 0] == 1
        assert acc["doy_count"][100, 0, 0] == 0
        # one PET slab, labelled 200501, non-negative.
        assert acc["pet_months"].shape == (1, 2, 2)
        assert int(acc["pet_labels"][0]) == 200501
        assert np.all(acc["pet_months"] >= 0)
        assert lat_grid is not None

    def test_sea_pixel_is_nan_in_pet(self, mod, tmp_path):
        xr = pytest.importorskip("xarray")
        pytest.importorskip("h5netcdf")
        import pandas as pd

        hours = 2 * 24
        times = pd.date_range("2005-06-01", periods=hours, freq="h")
        base = 273.15 + 25 + 8 * np.sin(np.linspace(0, 4 * np.pi, hours))
        data = np.broadcast_to(base[:, None, None], (hours, 1, 2)).astype("float64").copy()
        data[:, 0, 1] = np.nan  # second pixel is "sea" (no land data)
        ds = xr.Dataset(
            {"t2m": (("valid_time", "latitude", "longitude"), data)},
            coords={"valid_time": times, "latitude": [31.0], "longitude": [30.0, 30.1]},
        )
        raw = tmp_path / "era5land_t2m_2005-06.nc"
        ds.to_netcdf(raw, engine="h5netcdf")

        acc, _ = mod._fold_month(raw, {}, None, 2005, 6, np, xr)
        pet = acc["pet_months"][0]
        assert np.isfinite(pet[0, 0])  # land pixel has PET
        assert pet[0, 0] > 0
        assert np.isnan(pet[0, 1])  # sea pixel masked to NaN, not 0
