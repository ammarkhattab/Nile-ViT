"""Unit tests for `scripts/data/05b_tile.py`.

Offline-only: tiling math, sample-id formatting, region classification, and the
cloud/valid fraction logic (with synthetic numpy arrays). The Zarr/GeoParquet
I/O needs real rasters and is not tested in CI.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "05b_tile.py"


@pytest.fixture(scope="module")
def mod():
    if not SCRIPT_PATH.exists():
        pytest.skip(f"script not found at {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("tile_05b", SCRIPT_PATH)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["tile_05b"] = m
    spec.loader.exec_module(m)
    return m


# ---- patch_grid --------------------------------------------------------------
class TestPatchGrid:
    def test_hls_tile_3660(self, mod):
        # 3660 // 224 = 16 -> 16x16 = 256 patches.
        grid = mod.patch_grid(3660, 3660)
        assert len(grid) == 256

    def test_origins_in_bounds(self, mod):
        grid = mod.patch_grid(3660, 3660)
        for _r, _c, y0, x0 in grid:
            assert y0 + mod.PATCH <= 3660
            assert x0 + mod.PATCH <= 3660

    def test_first_and_last(self, mod):
        grid = mod.patch_grid(672, 672, patch=224)  # exactly 3x3
        assert len(grid) == 9
        assert grid[0] == (0, 0, 0, 0)
        assert grid[-1] == (2, 2, 448, 448)

    def test_remainder_dropped(self, mod):
        # 500 // 224 = 2 -> 2x2 = 4, remainder dropped.
        assert len(mod.patch_grid(500, 500)) == 4


# ---- make_sample_id ----------------------------------------------------------
class TestSampleId:
    def test_format(self, mod):
        assert mod.make_sample_id("T36RUU", date(2023, 8, 1), 5, 8) == "T36RUU_2023-08-01_r05c08"

    def test_zero_pad(self, mod):
        assert mod.make_sample_id("T36RUU", date(2023, 8, 1), 0, 0).endswith("r00c00")


# ---- classify_region ---------------------------------------------------------
class TestClassifyRegion:
    def test_delta(self, mod):
        # Cairo / central Delta ~ 31E, 31N.
        assert mod.classify_region(31.0, 31.0) == "delta"

    def test_n_coast(self, mod):
        # North coast strip ~ 28E, 31.5N (outside delta lon-lat box -> n_coast).
        assert mod.classify_region(28.0, 31.5) == "n_coast"

    def test_em_shelf(self, mod):
        # Open Eastern Med ~ 28E, 34N.
        assert mod.classify_region(28.0, 34.0) == "em_shelf"


# ---- cloud_fraction ----------------------------------------------------------
class TestCloudFraction:
    def test_all_clear(self, mod):
        fmask = np.zeros((10, 10), dtype=np.uint8)
        assert mod.cloud_fraction(fmask) == 0.0

    def test_all_cloud(self, mod):
        # bit1 (cloud) = value 2 set everywhere.
        fmask = np.full((10, 10), 2, dtype=np.uint8)
        assert mod.cloud_fraction(fmask) == 1.0

    def test_half_cloud(self, mod):
        fmask = np.zeros((10, 10), dtype=np.uint8)
        fmask[:5, :] = 8  # bit3 cloud shadow on half
        assert mod.cloud_fraction(fmask) == pytest.approx(0.5)

    def test_fill_excluded(self, mod):
        # All-fill -> treated as fully unobserved -> 1.0 by convention.
        fmask = np.full((10, 10), mod.FMASK_FILL, dtype=np.uint8)
        assert mod.cloud_fraction(fmask) == 1.0

    def test_non_cloud_bits_ignored(self, mod):
        # bit5 water (value 32) is not a cloud bit.
        fmask = np.full((10, 10), 32, dtype=np.uint8)
        assert mod.cloud_fraction(fmask) == 0.0


# ---- valid_fraction ----------------------------------------------------------
class TestValidFraction:
    def test_all_valid(self, mod):
        red = np.full((10, 10), 1500, dtype=np.int16)
        assert mod.valid_fraction(red) == 1.0

    def test_all_fill(self, mod):
        red = np.full((10, 10), mod.HLS_FILL, dtype=np.int16)
        assert mod.valid_fraction(red) == 0.0

    def test_half_fill(self, mod):
        red = np.full((10, 10), 1500, dtype=np.int16)
        red[:5, :] = mod.HLS_FILL
        assert mod.valid_fraction(red) == pytest.approx(0.5)


# ---- HLS helpers reused from 05a (date/sensor parsing) -----------------------
class TestHlsParsing:
    def test_date(self, mod):
        d = mod.hls_date_from_name("HLS.S30.T36RUU.2023213T082611.v2.0.B04.tif")
        assert d == date(2023, 8, 1)

    def test_sensor(self, mod):
        assert mod.hls_sensor_from_name("HLS.S30.T36RUU.2023213T082611.v2.0.B04.tif") == "S30"


# ---- band_path ---------------------------------------------------------------
class TestBandPath:
    def test_substitution(self, mod):
        b04 = Path("HLS.S30.T36RUU.2023213T082611.v2.0.B04.tif")
        assert mod.band_path(b04, "B8A").name == ("HLS.S30.T36RUU.2023213T082611.v2.0.B8A.tif")

    def test_fmask(self, mod):
        b04 = Path("HLS.S30.T36RUU.2023213T082611.v2.0.B04.tif")
        assert mod.band_path(b04, "Fmask").name.endswith(".Fmask.tif")


# ---- constants ---------------------------------------------------------------
class TestConstants:
    def test_band_order(self, mod):
        assert mod.BAND_NAMES == [
            "blue",
            "green",
            "red",
            "nir_narrow",
            "swir1",
            "swir2",
        ]

    def test_s30_l30_band_maps(self, mod):
        assert mod.HLS_BAND_MAP["S30"]["nir_narrow"] == "B8A"
        assert mod.HLS_BAND_MAP["L30"]["nir_narrow"] == "B05"
