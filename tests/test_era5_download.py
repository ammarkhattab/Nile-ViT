"""Unit tests for `scripts/data/02_download_era5.py`.

Offline-only tests: exercises the pure-Python helpers (date/area parsing,
month chunking, defaults) without touching the CDS API.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "02_download_era5.py"


@pytest.fixture(scope="module")
def era5_mod():
    """Import the script as a module via importlib (leading digit in filename)."""
    if not SCRIPT_PATH.exists():
        pytest.skip(f"script not found at {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("era5_download", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["era5_download"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- parse_area --------------------------------------------------------------
class TestParseArea:
    def test_valid_nile_em(self, era5_mod):
        n, w, s, e = era5_mod.parse_area("37,22,30,36")
        assert (n, w, s, e) == (37.0, 22.0, 30.0, 36.0)

    def test_whitespace_tolerated(self, era5_mod):
        n, w, s, e = era5_mod.parse_area(" 37 , 22 , 30 , 36 ")
        assert (n, w, s, e) == (37.0, 22.0, 30.0, 36.0)

    def test_wrong_count_raises(self, era5_mod):
        with pytest.raises(ValueError):
            era5_mod.parse_area("37,22,30")

    def test_non_numeric_raises(self, era5_mod):
        with pytest.raises(ValueError):
            era5_mod.parse_area("a,b,c,d")

    def test_n_le_s_raises(self, era5_mod):
        # CDS area order is N,W,S,E - N must be strictly greater than S
        with pytest.raises(ValueError, match="N .* must be greater than S"):
            era5_mod.parse_area("30,22,37,36")

    def test_e_le_w_raises(self, era5_mod):
        with pytest.raises(ValueError, match="E .* must be greater than W"):
            era5_mod.parse_area("37,36,30,22")

    def test_lat_out_of_range_raises(self, era5_mod):
        with pytest.raises(ValueError, match="Latitudes"):
            era5_mod.parse_area("95,22,30,36")

    def test_lon_out_of_range_raises(self, era5_mod):
        with pytest.raises(ValueError, match="Longitudes"):
            era5_mod.parse_area("37,-200,30,36")


# ---- daterange ---------------------------------------------------------------
class TestDaterange:
    def test_single_day(self, era5_mod):
        result = era5_mod.daterange(date(2023, 8, 1), date(2023, 8, 1))
        assert result == [date(2023, 8, 1)]

    def test_three_days(self, era5_mod):
        result = era5_mod.daterange(date(2023, 8, 1), date(2023, 8, 3))
        assert result == [date(2023, 8, 1), date(2023, 8, 2), date(2023, 8, 3)]

    def test_full_august(self, era5_mod):
        result = era5_mod.daterange(date(2023, 8, 1), date(2023, 8, 31))
        assert len(result) == 31
        assert result[0] == date(2023, 8, 1)
        assert result[-1] == date(2023, 8, 31)

    def test_cross_month(self, era5_mod):
        result = era5_mod.daterange(date(2023, 7, 30), date(2023, 8, 2))
        assert len(result) == 4
        assert result == [
            date(2023, 7, 30),
            date(2023, 7, 31),
            date(2023, 8, 1),
            date(2023, 8, 2),
        ]


# ---- days_by_yearmonth -------------------------------------------------------
class TestDaysByYearMonth:
    def test_single_month(self, era5_mod):
        dates = [date(2023, 8, 1), date(2023, 8, 15), date(2023, 8, 31)]
        result = era5_mod.days_by_yearmonth(dates)
        assert result == {(2023, 8): [1, 15, 31]}

    def test_spans_two_months(self, era5_mod):
        dates = era5_mod.daterange(date(2023, 7, 30), date(2023, 8, 2))
        result = era5_mod.days_by_yearmonth(dates)
        assert result == {(2023, 7): [30, 31], (2023, 8): [1, 2]}

    def test_spans_year_boundary(self, era5_mod):
        dates = era5_mod.daterange(date(2023, 12, 30), date(2024, 1, 2))
        result = era5_mod.days_by_yearmonth(dates)
        assert result == {(2023, 12): [30, 31], (2024, 1): [1, 2]}


# ---- parse_month -------------------------------------------------------------
class TestParseMonth:
    def test_august(self, era5_mod):
        first, last = era5_mod.parse_month("2023-08")
        assert first == date(2023, 8, 1)
        assert last == date(2023, 8, 31)

    def test_february_leap(self, era5_mod):
        first, last = era5_mod.parse_month("2024-02")
        assert first == date(2024, 2, 1)
        assert last == date(2024, 2, 29)

    def test_december_wraps_correctly(self, era5_mod):
        first, last = era5_mod.parse_month("2023-12")
        assert first == date(2023, 12, 1)
        assert last == date(2023, 12, 31)

    @pytest.mark.parametrize("bad", ["", "2023", "2023-13", "abcd-ef"])
    def test_invalid_month_raises(self, era5_mod, bad):
        with pytest.raises(ValueError):
            era5_mod.parse_month(bad)


# ---- Constants ---------------------------------------------------------------
class TestDefaults:
    def test_default_area_covers_nile_delta(self, era5_mod):
        """Cairo at (31.2 E, 30.0 N) must fall inside the default area."""
        n, w, s, e = era5_mod.DEFAULT_AREA
        cairo_lon, cairo_lat = 31.24, 30.04
        assert s <= cairo_lat <= n
        assert w <= cairo_lon <= e

    def test_default_area_covers_eastern_med(self, era5_mod):
        """A point ~35 N, 30 E (Eastern Med) must be inside the area."""
        n, w, s, e = era5_mod.DEFAULT_AREA
        assert s <= 35.0 <= n
        assert w <= 30.0 <= e

    def test_default_variables_nonempty(self, era5_mod):
        assert len(era5_mod.DEFAULT_VARIABLES) >= 4

    def test_default_variables_include_essentials(self, era5_mod):
        defaults = era5_mod.DEFAULT_VARIABLES
        # The four meteo vars we listed in the PRD/TileSample schema.
        assert "2m_temperature" in defaults
        assert "volumetric_soil_water_layer_1" in defaults
        assert "total_evaporation" in defaults
        assert "total_precipitation" in defaults

    def test_var_to_shortname_maps_defaults(self, era5_mod):
        for var in era5_mod.DEFAULT_VARIABLES:
            assert var in era5_mod.VAR_TO_SHORTNAME
