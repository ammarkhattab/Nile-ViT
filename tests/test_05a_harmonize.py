"""Unit tests for `scripts/data/05a_harmonize.py`.

Offline-only: exercises the pure date/filename parsing and MODIS-composite
selection helpers. The rioxarray reproject_match work needs real rasters and is
not tested in CI.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "05a_harmonize.py"


@pytest.fixture(scope="module")
def mod():
    if not SCRIPT_PATH.exists():
        pytest.skip(f"script not found at {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("harmonize_05a", SCRIPT_PATH)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["harmonize_05a"] = m
    spec.loader.exec_module(m)
    return m


# ---- doy_to_date -------------------------------------------------------------
class TestDoyToDate:
    def test_doy_1(self, mod):
        assert mod.doy_to_date(2023, 1) == date(2023, 1, 1)

    def test_doy_213_is_aug_1(self, mod):
        # 2023 is not a leap year: DOY 213 = Aug 1.
        assert mod.doy_to_date(2023, 213) == date(2023, 8, 1)

    def test_doy_209(self, mod):
        assert mod.doy_to_date(2023, 209) == date(2023, 7, 28)

    def test_leap_year(self, mod):
        # 2024 leap: DOY 60 = Feb 29.
        assert mod.doy_to_date(2024, 60) == date(2024, 2, 29)


# ---- hls_sensor_from_name ----------------------------------------------------
class TestHlsSensor:
    def test_s30(self, mod):
        assert mod.hls_sensor_from_name("HLS.S30.T36RUU.2023213T083559.v2.0.B04.tif") == "S30"

    def test_l30(self, mod):
        assert mod.hls_sensor_from_name("HLS.L30.T36RUU.2023215T075959.v2.0.B04.tif") == "L30"

    def test_none(self, mod):
        assert mod.hls_sensor_from_name("random_file.tif") is None


# ---- hls_date_from_name ------------------------------------------------------
class TestHlsDate:
    def test_s30_date(self, mod):
        d = mod.hls_date_from_name("HLS.S30.T36RUU.2023213T083559.v2.0.B04.tif")
        assert d == date(2023, 8, 1)

    def test_another_date(self, mod):
        d = mod.hls_date_from_name("HLS.S30.T36RUU.2023225T083601.v2.0.B04.tif")
        assert d == date(2023, 8, 13)

    def test_no_match(self, mod):
        assert mod.hls_date_from_name("not_an_hls_file.tif") is None

    def test_bad_doy(self, mod):
        assert mod.hls_date_from_name("HLS.S30.T36RUU.2023400T083559.v2.0.B04.tif") is None


# ---- modis_composite_from_name -----------------------------------------------
class TestModisComposite:
    def test_ndvi(self, mod):
        assert mod.modis_composite_from_name(
            "MOD13Q1.A2023209.h20v05.061.2023226000724.250m_16_days_NDVI.nile-em.tif"
        ) == (2023, 209)

    def test_lst(self, mod):
        assert mod.modis_composite_from_name(
            "MOD11A2.A2023233.h21v05.061.2023242035829.LST_Day_1km.nile-em.tif"
        ) == (2023, 233)

    def test_none(self, mod):
        assert mod.modis_composite_from_name("foo.tif") is None


# ---- select_modis_composite --------------------------------------------------
class TestSelectComposite:
    def test_picks_latest_started(self, mod):
        # NDVI composites 209, 225, 241; target Aug 1 (DOY 213) -> 209.
        doys = [209, 225, 241]
        assert mod.select_modis_composite(date(2023, 8, 1), doys, 2023) == 209

    def test_target_in_second_window(self, mod):
        # Aug 15 = DOY 227 -> composite 225.
        doys = [209, 225, 241]
        assert mod.select_modis_composite(date(2023, 8, 15), doys, 2023) == 225

    def test_target_exactly_on_start(self, mod):
        # Aug 13 = DOY 225 -> composite 225.
        doys = [209, 225, 241]
        assert mod.select_modis_composite(date(2023, 8, 13), doys, 2023) == 225

    def test_target_before_all_falls_back(self, mod):
        doys = [209, 225, 241]
        # DOY 100 precedes all -> earliest (209).
        assert mod.select_modis_composite(date(2023, 4, 10), doys, 2023) == 209

    def test_empty(self, mod):
        assert mod.select_modis_composite(date(2023, 8, 1), [], 2023) is None


# ---- HLS_BAND_MAP ------------------------------------------------------------
class TestBandMap:
    def test_s30_bands(self, mod):
        m = mod.HLS_BAND_MAP["S30"]
        assert m["red"] == "B04"
        assert m["nir_narrow"] == "B8A"
        assert m["swir2"] == "B12"

    def test_l30_bands(self, mod):
        m = mod.HLS_BAND_MAP["L30"]
        assert m["red"] == "B04"
        assert m["nir_narrow"] == "B05"
        assert m["swir2"] == "B07"

    def test_both_have_six_bands(self, mod):
        for sensor in ("S30", "L30"):
            assert len(mod.HLS_BAND_MAP[sensor]) == 6
