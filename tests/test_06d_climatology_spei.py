"""Offline tests for Script 06d (SPEI-3 water-balance combine). No downloads."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "06d_climatology_spei.py"


@pytest.fixture(scope="module")
def mod():
    if not SCRIPT_PATH.exists():
        pytest.skip(f"script not found at {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("clim_spei", SCRIPT_PATH)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["clim_spei"] = m
    spec.loader.exec_module(m)
    return m


def _monthly(values, lat, lon, start="1991-01"):
    """Build a (time, lat, lon) DataArray with monthly time from `start`."""
    xr = pytest.importorskip("xarray")
    import pandas as pd

    n = values.shape[0]
    time = pd.date_range(start, periods=n, freq="MS")
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
    )


# ---- coord standardization ----
def test_standardize_names(mod):
    xr = pytest.importorskip("xarray")
    ds = xr.Dataset(
        {"precip": (("latitude", "longitude"), np.ones((2, 2)))},
        coords={"latitude": [30.0, 31.0], "longitude": [22.0, 23.0]},
    )
    out = mod.standardize_names(ds)
    assert "lat" in out.coords and "lon" in out.coords
    assert "latitude" not in out.coords


# ---- time alignment ----
class TestAlignTime:
    def test_matching_months_share_time(self, mod):
        lat, lon = [30.0], [22.0]
        p = _monthly(np.ones((3, 1, 1)), lat, lon, "2000-01")
        # PET stamped mid-month, same calendar months
        e = _monthly(np.ones((3, 1, 1)), lat, lon, "2000-01")
        e = e.assign_coords(time=e["time"] + np.timedelta64(14, "D"))
        p2, e2 = mod.align_time(p, e)
        assert np.array_equal(e2["time"].values, p2["time"].values)

    def test_mismatched_months_raise(self, mod):
        lat, lon = [30.0], [22.0]
        p = _monthly(np.ones((3, 1, 1)), lat, lon, "2000-01")
        e = _monthly(np.ones((3, 1, 1)), lat, lon, "2010-01")  # different years
        with pytest.raises(ValueError, match="differ"):
            mod.align_time(p, e)


# ---- regridding ----
def test_regrid_coarse_to_fine_preserves_nodes(mod):
    pytest.importorskip("xarray")
    coarse = _monthly(np.arange(4.0).reshape(1, 2, 2), [30.0, 31.0], [22.0, 23.0]).isel(time=0)
    fine_lat = np.array([30.0, 30.5, 31.0])
    fine_lon = np.array([22.0, 22.5, 23.0])
    out = mod.regrid_to(coarse, fine_lat, fine_lon)
    assert out.sizes["lat"] == 3 and out.sizes["lon"] == 3
    # values at the original nodes are preserved by bilinear interp
    assert float(out.sel(lat=30.0, lon=22.0)) == pytest.approx(0.0)
    assert float(out.sel(lat=31.0, lon=23.0)) == pytest.approx(3.0)


# ---- water-balance stats ----
class TestWaterBalanceStats:
    def test_constant_balance(self, mod):
        # 3 years monthly, constant precip=10, pet=4 -> WB=6, WB_3=18.
        lat, lon = [30.0], [22.0]
        n = 36
        p = _monthly(np.full((n, 1, 1), 10.0), lat, lon)
        e = _monthly(np.full((n, 1, 1), 4.0), lat, lon)
        mean, std = mod.water_balance_stats(p, e, window=3)
        assert dict(mean.sizes) == {"month": 12, "lat": 1, "lon": 1}
        assert float(mean.sel(month=6).isel(lat=0, lon=0)) == pytest.approx(18.0)
        assert float(std.sel(month=6).isel(lat=0, lon=0)) == pytest.approx(0.0, abs=1e-6)

    def test_rolling_window_warmup_is_dropped(self, mod):
        # First 2 months have an incomplete 3-window; they must not crash stats.
        lat, lon = [30.0], [22.0]
        n = 24
        rng = np.random.default_rng(0)
        p = _monthly(rng.random((n, 1, 1)) + 5, lat, lon)
        e = _monthly(rng.random((n, 1, 1)), lat, lon)
        mean, std = mod.water_balance_stats(p, e, window=3)
        # every calendar month present and finite
        assert int(mean.sizes["month"]) == 12
        assert bool(np.isfinite(mean).all())

    def test_seasonal_signal_recovered(self, mod):
        # Wet winter, dry summer -> Jan 3-mo WB should exceed Jul.
        lat, lon = [30.0], [22.0]
        months = np.arange(36) % 12 + 1
        precip_seasonal = np.where(np.isin(months, [12, 1, 2]), 60.0, 2.0)
        p = _monthly(precip_seasonal.reshape(36, 1, 1), lat, lon)
        e = _monthly(np.full((36, 1, 1), 5.0), lat, lon)
        mean, _ = mod.water_balance_stats(p, e, window=3)
        jan = float(mean.sel(month=1).isel(lat=0, lon=0))
        jul = float(mean.sel(month=7).isel(lat=0, lon=0))
        assert jan > jul
