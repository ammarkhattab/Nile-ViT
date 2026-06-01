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


# ---- attribute sanitizer (surrogate scrubbing) -------------------------------
class TestSanitizeAttrs:
    def test_clean_value_scrubs_surrogate_str(self, chirps_mod):
        bad = "ok\udce9bad"  # lone surrogate that breaks utf-8 encoding
        cleaned = chirps_mod._clean_value(bad)
        # must now round-trip through utf-8 without raising
        cleaned.encode("utf-8")
        assert "ok" in cleaned and "bad" in cleaned

    def test_clean_value_decodes_bytes(self, chirps_mod):
        assert chirps_mod._clean_value(b"hello") == "hello"

    def test_clean_value_passes_through_non_strings(self, chirps_mod):
        assert chirps_mod._clean_value(42) == 42
        assert chirps_mod._clean_value(3.5) == 3.5

    def test_sanitize_attrs_on_dataset(self, chirps_mod):
        xr = pytest.importorskip("xarray")
        import numpy as np

        ds = xr.Dataset(
            {"precip": ("x", np.arange(3.0))},
            coords={"x": [0, 1, 2]},
            attrs={"history": "made\udce9here", "ok": "fine"},
        )
        ds["precip"].attrs["note"] = "bad\udcffbyte"
        chirps_mod._sanitize_attrs(ds)
        # every attr must now encode cleanly
        for v in ds.attrs.values():
            if isinstance(v, str):
                v.encode("utf-8")
        for v in ds["precip"].attrs.values():
            if isinstance(v, str):
                v.encode("utf-8")
        assert ds.attrs["ok"] == "fine"


# ---- days_in_month -----------------------------------------------------------
class TestDaysInMonth:
    def test_august(self, chirps_mod):
        assert chirps_mod.days_in_month(2023, 8) == 31

    def test_september(self, chirps_mod):
        assert chirps_mod.days_in_month(2023, 9) == 30

    def test_feb_leap(self, chirps_mod):
        assert chirps_mod.days_in_month(2024, 2) == 29

    def test_feb_non_leap(self, chirps_mod):
        assert chirps_mod.days_in_month(2023, 2) == 28


# ---- subset_is_valid ---------------------------------------------------------
class TestSubsetIsValid:
    def _write(self, tmp_path, n_time, var="precip"):
        xr = pytest.importorskip("xarray")
        import numpy as np

        ds = xr.Dataset(
            {var: (("time", "latitude", "longitude"), np.zeros((n_time, 2, 2)))},
            coords={
                "time": np.arange(n_time),
                "latitude": [30.0, 31.0],
                "longitude": [22.0, 23.0],
            },
        )
        p = tmp_path / "sub.nc"
        ds.to_netcdf(p, engine="h5netcdf")
        return p

    def test_valid_file(self, chirps_mod, tmp_path):
        p = self._write(tmp_path, 30)
        assert chirps_mod.subset_is_valid(p, 30) is True

    def test_wrong_time_length(self, chirps_mod, tmp_path):
        # truncated: only 10 of 30 days
        p = self._write(tmp_path, 10)
        assert chirps_mod.subset_is_valid(p, 30) is False

    def test_missing_precip(self, chirps_mod, tmp_path):
        p = self._write(tmp_path, 30, var="other")
        assert chirps_mod.subset_is_valid(p, 30) is False

    def test_corrupt_file(self, chirps_mod, tmp_path):
        pytest.importorskip("xarray")
        bad = tmp_path / "bad.nc"
        bad.write_bytes(b"not a netcdf file at all")
        assert chirps_mod.subset_is_valid(bad, 30) is False

    def test_missing_file(self, chirps_mod, tmp_path):
        pytest.importorskip("xarray")
        assert chirps_mod.subset_is_valid(tmp_path / "nope.nc", 30) is False
