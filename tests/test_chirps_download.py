"""Unit tests for `scripts/data/03_download_chirps.py`.

Offline-only: exercises filename/URL construction, year-month parsing,
month-range expansion, and the CHC base URL constant. Network probing and
downloading are not tested in CI (no internet in the GitHub runner).
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "03_download_chirps.py"


@pytest.fixture(scope="module")
def chirps_mod():
    if not SCRIPT_PATH.exists():
        pytest.skip(f"script not found at {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("chirps_download", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["chirps_download"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- monthly_filename --------------------------------------------------------
class TestMonthlyFilename:
    def test_basic(self, chirps_mod):
        assert chirps_mod.monthly_filename(2023, 8) == "chirps-v3.0.2023.08.days_p05.nc"

    def test_zero_padding(self, chirps_mod):
        # Confirmed against the CHC autoindex on 2026-05-28.
        assert chirps_mod.monthly_filename(1981, 1) == "chirps-v3.0.1981.01.days_p05.nc"

    def test_december(self, chirps_mod):
        assert chirps_mod.monthly_filename(2024, 12) == "chirps-v3.0.2024.12.days_p05.nc"


# ---- subset_filename ---------------------------------------------------------
class TestSubsetFilename:
    def test_default_tag(self, chirps_mod):
        assert chirps_mod.subset_filename(2023, 8) == "chirps-v3.0.2023.08.days_p05.nile-em.nc"

    def test_custom_tag(self, chirps_mod):
        assert (
            chirps_mod.subset_filename(2023, 8, "delta-core")
            == "chirps-v3.0.2023.08.days_p05.delta-core.nc"
        )


# ---- monthly_url -------------------------------------------------------------
class TestMonthlyUrl:
    def test_rnl_default(self, chirps_mod):
        url = chirps_mod.monthly_url(2023, 8)
        assert (
            url == "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/rnl"
            "/netcdf/byMonth/chirps-v3.0.2023.08.days_p05.nc"
        )

    def test_sat_flavor(self, chirps_mod):
        url = chirps_mod.monthly_url(2023, 8, flavor="sat")
        assert "/final/sat/netcdf/byMonth/" in url
        assert url.endswith("chirps-v3.0.2023.08.days_p05.nc")

    def test_url_components(self, chirps_mod):
        url = chirps_mod.monthly_url(2017, 1)
        assert url.startswith("https://data.chc.ucsb.edu")
        assert "CHIRPS/v3.0/daily/final/rnl/netcdf/byMonth" in url
        assert "chirps-v3.0.2017.01.days_p05.nc" in url


# ---- parse_year_month --------------------------------------------------------
class TestParseYearMonth:
    def test_august_2023(self, chirps_mod):
        assert chirps_mod.parse_year_month("2023-08") == (2023, 8)

    def test_january(self, chirps_mod):
        assert chirps_mod.parse_year_month("2017-01") == (2017, 1)

    def test_whitespace(self, chirps_mod):
        assert chirps_mod.parse_year_month("  2023-08  ") == (2023, 8)

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "2023",
            "2023-13",
            "2023-00",
            "1980-12",  # before CHIRPS v3 start
            "abc-de",
        ],
    )
    def test_invalid_raises(self, chirps_mod, bad):
        with pytest.raises(ValueError):
            chirps_mod.parse_year_month(bad)


# ---- months_in_range ---------------------------------------------------------
class TestMonthsInRange:
    def test_single_month(self, chirps_mod):
        result = chirps_mod.months_in_range((2023, 8), (2023, 8))
        assert result == [(2023, 8)]

    def test_same_year(self, chirps_mod):
        result = chirps_mod.months_in_range((2023, 6), (2023, 9))
        assert result == [(2023, 6), (2023, 7), (2023, 8), (2023, 9)]

    def test_cross_year(self, chirps_mod):
        result = chirps_mod.months_in_range((2022, 11), (2023, 2))
        assert result == [(2022, 11), (2022, 12), (2023, 1), (2023, 2)]

    def test_full_project_range(self, chirps_mod):
        # 2017-01 through 2024-12: 8 years * 12 months = 96 months.
        result = chirps_mod.months_in_range((2017, 1), (2024, 12))
        assert len(result) == 96
        assert result[0] == (2017, 1)
        assert result[-1] == (2024, 12)

    def test_reverse_order_raises(self, chirps_mod):
        with pytest.raises(ValueError):
            chirps_mod.months_in_range((2023, 12), (2023, 1))


# ---- Constants ---------------------------------------------------------------
class TestConstants:
    def test_chc_base_is_v3_final(self, chirps_mod):
        assert chirps_mod.CHC_BASE.startswith("https://data.chc.ucsb.edu")
        assert "CHIRPS/v3.0/daily/final" in chirps_mod.CHC_BASE

    def test_default_area_is_nile_em(self, chirps_mod):
        n, w, s, e = chirps_mod.DEFAULT_AREA
        # Sanity: northern hemisphere, around Egypt + East Med.
        assert s < n and w < e
        assert 25 < s < 35
        assert 35 < n < 40
        assert 20 < w < 25
        assert 30 < e < 40

    def test_default_roi_tag(self, chirps_mod):
        assert chirps_mod.DEFAULT_ROI_TAG == "nile-em"


# ---- parse_month backwards-compat shim ---------------------------------------
class TestParseMonthCompat:
    def test_returns_first_and_last_dates(self, chirps_mod):
        first, last = chirps_mod.parse_month("2023-08")
        assert first == date(2023, 8, 1)
        assert last == date(2023, 8, 31)

    def test_feb_leap(self, chirps_mod):
        _, last = chirps_mod.parse_month("2024-02")
        assert last == date(2024, 2, 29)

    def test_feb_non_leap(self, chirps_mod):
        _, last = chirps_mod.parse_month("2023-02")
        assert last == date(2023, 2, 28)
