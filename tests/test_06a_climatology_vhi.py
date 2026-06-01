"""Unit tests for `scripts/data/06a_climatology_vhi.py`.

Offline-only: product registry, tile parsing, fill masking, the running
min/max fold, and checkpoint round-trip. STAC streaming is not tested in CI.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "06a_climatology_vhi.py"


@pytest.fixture(scope="module")
def mod():
    if not SCRIPT_PATH.exists():
        pytest.skip(f"script not found at {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("clim_vhi", SCRIPT_PATH)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["clim_vhi"] = m
    spec.loader.exec_module(m)
    return m


# ---- product registry --------------------------------------------------------
class TestProducts:
    def test_two_products(self, mod):
        assert set(mod.PRODUCTS) == {"MOD13Q1", "MOD11A2"}

    def test_ndvi_mapping(self, mod):
        p = mod.PRODUCTS["MOD13Q1"]
        assert p.collection == "modis-13Q1-061"
        assert p.asset == "250m_16_days_NDVI"
        assert p.var == "ndvi"

    def test_lst_mapping(self, mod):
        p = mod.PRODUCTS["MOD11A2"]
        assert p.collection == "modis-11A2-061"
        assert p.var == "lst"

    def test_three_v05_tiles(self, mod):
        assert mod.V05_TILES == ("h19v05", "h20v05", "h21v05")


# ---- tile_of -----------------------------------------------------------------
class TestTileOf:
    def test_extracts_tile(self, mod):
        assert mod.tile_of("MOD13Q1.A2023209.h19v05.061.2023226000350") == "h19v05"

    def test_none_when_absent(self, mod):
        assert mod.tile_of("no-tile-here") is None


# ---- valid_mask_for ----------------------------------------------------------
class TestValidMask:
    def test_ndvi_excludes_fill(self, mod):
        arr = np.array([-3000, -2000, 0, 5000, 10000, 11000])
        m = mod.valid_mask_for("MOD13Q1", arr)
        # -3000 (fill) and 11000 (>max) excluded; rest valid
        assert list(m) == [False, True, True, True, True, False]

    def test_lst_excludes_zero_fill(self, mod):
        arr = np.array([0, 1, 13000, 16000])
        m = mod.valid_mask_for("MOD11A2", arr)
        assert list(m) == [False, True, True, True]


# ---- update_minmax -----------------------------------------------------------
class TestUpdateMinmax:
    def test_seeds_on_first_call(self, mod):
        vals = np.array([[1.0, 2.0], [3.0, 4.0]])
        valid = np.ones_like(vals, dtype=bool)
        amin, amax = mod.update_minmax(None, None, vals, valid)
        assert np.array_equal(amin, vals)
        assert np.array_equal(amax, vals)

    def test_folds_min_and_max(self, mod):
        a = np.array([5.0, 5.0, 5.0])
        valid = np.ones(3, dtype=bool)
        amin, amax = mod.update_minmax(a.copy(), a.copy(), np.array([3.0, 7.0, 5.0]), valid)
        assert list(amin) == [3.0, 5.0, 5.0]
        assert list(amax) == [5.0, 7.0, 5.0]

    def test_invalid_pixels_ignored(self, mod):
        valid_all = np.ones(2, dtype=bool)
        amin, amax = mod.update_minmax(None, None, np.array([5.0, 5.0]), valid_all)
        # second update: a tiny value but marked invalid -> must NOT lower the min
        amin, amax = mod.update_minmax(amin, amax, np.array([1.0, 9.0]), np.array([False, True]))
        assert amin[0] == 5.0  # invalid 1.0 ignored
        assert amax[1] == 9.0  # valid 9.0 applied

    def test_nan_unseen_then_filled(self, mod):
        # first obs invalid everywhere -> all NaN; second obs valid -> fills
        amin, amax = mod.update_minmax(None, None, np.array([1.0, 2.0]), np.array([False, False]))
        assert np.isnan(amin).all()
        amin, amax = mod.update_minmax(amin, amax, np.array([4.0, 8.0]), np.array([True, True]))
        assert list(amin) == [4.0, 8.0]


# ---- checkpoint round-trip ---------------------------------------------------
class TestCheckpoint:
    def test_roundtrip(self, mod, tmp_path):
        path = mod.ckpt_path(tmp_path, "ndvi")
        acc = {
            "min_h19v05": np.array([1.0, 2.0]),
            "max_h19v05": np.array([3.0, 4.0]),
        }
        mod._save_checkpoint(path, acc, {2001, 2002})
        acc2, done = mod._load_checkpoint(path)
        assert done == {2001, 2002}
        assert np.array_equal(acc2["min_h19v05"], acc["min_h19v05"])
        assert np.array_equal(acc2["max_h19v05"], acc["max_h19v05"])

    def test_missing_checkpoint_is_empty(self, mod, tmp_path):
        acc, done = mod._load_checkpoint(tmp_path / "nope.npz")
        assert acc == {}
        assert done == set()

    def test_ckpt_path_naming(self, mod, tmp_path):
        assert mod.ckpt_path(tmp_path, "lst").name == ".ckpt_lst.npz"
