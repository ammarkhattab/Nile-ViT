"""Unit tests for `scripts/data/04_download_modis.py` (Planetary Computer version).

Offline-only: exercises the product catalog, date helpers, Terra filter, and
band-asset picker. The pystac-client search / rioxarray clip are not tested in
CI (they need network + MPC).
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "04_download_modis.py"


@pytest.fixture(scope="module")
def modis_mod():
    if not SCRIPT_PATH.exists():
        pytest.skip(f"script not found at {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("modis_download", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modis_download"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- SUPPORTED_PRODUCTS ------------------------------------------------------
class TestSupportedProducts:
    def test_both_terra_products(self, modis_mod):
        assert "MOD13Q1" in modis_mod.SUPPORTED_PRODUCTS
        assert "MOD11A2" in modis_mod.SUPPORTED_PRODUCTS

    def test_collections(self, modis_mod):
        assert modis_mod.SUPPORTED_PRODUCTS["MOD13Q1"]["collection"] == "modis-13Q1-061"
        assert modis_mod.SUPPORTED_PRODUCTS["MOD11A2"]["collection"] == "modis-11A2-061"

    def test_required_keys(self, modis_mod):
        required = {
            "collection",
            "long_name",
            "cadence",
            "resolution",
            "use",
            "band_pattern",
            "expected_asset",
        }
        for p, meta in modis_mod.SUPPORTED_PRODUCTS.items():
            assert required <= meta.keys(), f"{p} missing keys"

    def test_band_patterns(self, modis_mod):
        assert modis_mod.SUPPORTED_PRODUCTS["MOD13Q1"]["band_pattern"] == "NDVI"
        assert modis_mod.SUPPORTED_PRODUCTS["MOD11A2"]["band_pattern"] == "LST_DAY"


# ---- is_terra ----------------------------------------------------------------
class TestIsTerra:
    def test_terra_true(self, modis_mod):
        assert modis_mod.is_terra("MOD13Q1.A2023209.h21v06.061.2023226000401")
        assert modis_mod.is_terra("MOD11A2.A2023217.h20v05.061.2023230000000")

    def test_aqua_false(self, modis_mod):
        assert not modis_mod.is_terra("MYD13Q1.A2023209.h21v06.061.2023226000401")

    def test_case_insensitive(self, modis_mod):
        assert modis_mod.is_terra("mod13q1.a2023209.h21v06")

    def test_empty_false(self, modis_mod):
        assert not modis_mod.is_terra("")


# ---- pick_band_asset ---------------------------------------------------------
class TestPickBandAsset:
    def test_exact_match(self, modis_mod):
        keys = ["250m_16_days_NDVI", "250m_16_days_EVI", "hdf"]
        assert modis_mod.pick_band_asset(keys, "250m_16_days_NDVI") == "250m_16_days_NDVI"

    def test_substring_match_ndvi(self, modis_mod):
        keys = ["250m_16_days_NDVI", "250m_16_days_EVI", "hdf", "metadata"]
        assert modis_mod.pick_band_asset(keys, "NDVI") == "250m_16_days_NDVI"

    def test_substring_match_lst(self, modis_mod):
        keys = ["LST_Day_1km", "LST_Night_1km", "QC_Day", "hdf"]
        assert modis_mod.pick_band_asset(keys, "LST_DAY") == "LST_Day_1km"

    def test_case_insensitive(self, modis_mod):
        keys = ["250m_16_days_ndvi"]
        assert modis_mod.pick_band_asset(keys, "NDVI") == "250m_16_days_ndvi"

    def test_no_match_returns_none(self, modis_mod):
        keys = ["EVI", "hdf", "metadata"]
        assert modis_mod.pick_band_asset(keys, "NDVI") is None

    def test_prefers_exact_over_substring(self, modis_mod):
        # If both an exact and a longer substring match exist, exact wins.
        keys = ["NDVI", "250m_16_days_NDVI"]
        assert modis_mod.pick_band_asset(keys, "NDVI") == "NDVI"


# ---- parse_date --------------------------------------------------------------
class TestParseDate:
    def test_basic(self, modis_mod):
        assert modis_mod.parse_date("2023-08-15") == date(2023, 8, 15)

    @pytest.mark.parametrize("bad", ["", "2023-08", "2023/08/15", "abc"])
    def test_invalid_raises(self, modis_mod, bad):
        with pytest.raises(ValueError):
            modis_mod.parse_date(bad)


# ---- year_range --------------------------------------------------------------
class TestYearRange:
    def test_basic(self, modis_mod):
        start, end = modis_mod.year_range(2023)
        assert start == date(2023, 1, 1)
        assert end == date(2023, 12, 31)

    @pytest.mark.parametrize("bad", [1999, 1850, 2100])
    def test_implausible_raises(self, modis_mod, bad):
        with pytest.raises(ValueError):
            modis_mod.year_range(bad)


# ---- month_range -------------------------------------------------------------
class TestMonthRange:
    def test_basic(self, modis_mod):
        first, last = modis_mod.month_range("2023-08")
        assert first == date(2023, 8, 1)
        assert last == date(2023, 8, 31)

    def test_feb_leap(self, modis_mod):
        _, last = modis_mod.month_range("2024-02")
        assert last == date(2024, 2, 29)

    @pytest.mark.parametrize("bad", ["", "2023", "2023-13", "2023-00"])
    def test_invalid_raises(self, modis_mod, bad):
        with pytest.raises(ValueError):
            modis_mod.month_range(bad)


# ---- DEFAULT_BBOX ------------------------------------------------------------
class TestDefaultBbox:
    def test_order(self, modis_mod):
        w, s, e, n = modis_mod.DEFAULT_BBOX
        assert w < e and s < n

    def test_covers_nile_delta(self, modis_mod):
        w, s, e, n = modis_mod.DEFAULT_BBOX
        assert w < 31 < e and s < 30.5 < n


# ---- constants ---------------------------------------------------------------
class TestConstants:
    def test_mpc_stac_url(self, modis_mod):
        assert modis_mod.MPC_STAC_URL.startswith("https://planetarycomputer")
        assert modis_mod.MPC_STAC_URL.endswith("/stac/v1")

    def test_roi_tag(self, modis_mod):
        assert modis_mod.DEFAULT_ROI_TAG == "nile-em"
