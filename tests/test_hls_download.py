"""Unit tests for `scripts/data/01_download_hls.py`.

These tests only exercise the pure-Python helpers (date / bbox / region parsing,
default path resolution, MGRS tile catalogue). Network paths (Earthdata auth,
CMR search, download) are intentionally NOT exercised in CI — those belong in
manual smoke tests run against the live NASA endpoints.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

# ---- Load the script as a module ----------------------------------------------------
# scripts/data/01_download_hls.py isn't a "real" package (it has a leading digit
# and lives outside a package dir), so we load it via importlib.
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "01_download_hls.py"


@pytest.fixture(scope="module")
def hls_mod():
    """Import the script as a module."""
    if not SCRIPT_PATH.exists():
        pytest.skip(f"script not found at {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("hls_download", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hls_download"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- parse_month --------------------------------------------------------------------
class TestParseMonth:
    def test_august_2023(self, hls_mod):
        first, last = hls_mod.parse_month("2023-08")
        assert first == date(2023, 8, 1)
        assert last == date(2023, 8, 31)

    def test_february_non_leap(self, hls_mod):
        first, last = hls_mod.parse_month("2023-02")
        assert first == date(2023, 2, 1)
        assert last == date(2023, 2, 28)

    def test_february_leap(self, hls_mod):
        first, last = hls_mod.parse_month("2024-02")
        assert first == date(2024, 2, 1)
        assert last == date(2024, 2, 29)

    def test_december_wraps_year_correctly(self, hls_mod):
        first, last = hls_mod.parse_month("2023-12")
        assert first == date(2023, 12, 1)
        assert last == date(2023, 12, 31)

    @pytest.mark.parametrize("bad", ["", "2023", "2023-13", "2023/08", "abcd-ef"])
    def test_invalid_month_raises(self, hls_mod, bad):
        with pytest.raises(ValueError):
            hls_mod.parse_month(bad)


# ---- parse_bbox ---------------------------------------------------------------------
class TestParseBbox:
    def test_valid_nile_delta_bbox(self, hls_mod):
        result = hls_mod.parse_bbox("29.5,30.0,32.5,31.5")
        assert result == (29.5, 30.0, 32.5, 31.5)

    def test_whitespace_tolerated(self, hls_mod):
        result = hls_mod.parse_bbox("  29.5 , 30.0 , 32.5 , 31.5  ")
        assert result == (29.5, 30.0, 32.5, 31.5)

    def test_wrong_count_raises(self, hls_mod):
        with pytest.raises(ValueError):
            hls_mod.parse_bbox("29.5,30.0,32.5")

    def test_non_numeric_raises(self, hls_mod):
        with pytest.raises(ValueError):
            hls_mod.parse_bbox("a,b,c,d")

    def test_inverted_lon_raises(self, hls_mod):
        with pytest.raises(ValueError):
            hls_mod.parse_bbox("32.5,30.0,29.5,31.5")

    def test_inverted_lat_raises(self, hls_mod):
        with pytest.raises(ValueError):
            hls_mod.parse_bbox("29.5,31.5,32.5,30.0")


# ---- KNOWN_TILES catalogue ----------------------------------------------------------
class TestKnownTiles:
    def test_t36ruu_covers_cairo(self, hls_mod):
        """Cairo is ~31.2°E, 30.0°N — must fall inside T36RUU bounds."""
        lon_min, lat_min, lon_max, lat_max = hls_mod.KNOWN_TILES["T36RUU"]
        cairo_lon, cairo_lat = 31.24, 30.04
        assert lon_min <= cairo_lon <= lon_max
        assert lat_min <= cairo_lat <= lat_max

    def test_all_bboxes_are_well_formed(self, hls_mod):
        for tile, (lon_min, lat_min, lon_max, lat_max) in hls_mod.KNOWN_TILES.items():
            assert lon_min < lon_max, f"{tile} lon inverted"
            assert lat_min < lat_max, f"{tile} lat inverted"
            assert -180 <= lon_min <= 180, f"{tile} lon out of range"
            assert -90 <= lat_min <= 90, f"{tile} lat out of range"


# ---- default_data_root --------------------------------------------------------------
class TestDefaultDataRoot:
    def test_local_path_is_repo_relative(self, hls_mod):
        path = hls_mod.default_data_root()
        # Should be <repo>/data when running locally (not in Colab).
        # The fixture isn't running in Colab so we expect the local branch.
        if not hls_mod.is_colab():
            assert path.name == "data"
            # And it should live inside the repo, alongside scripts/
            assert (path.parent / "scripts").exists()
