"""Offline tests for nilevit/select.py (label-driven patch selection)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from nilevit.select import (
    decide_patch,
    grid_cells,
    keep_by_prob,
    label_window,
    patch_class,
    selection_report,
)


def _label_da(grid: np.ndarray, lon0=30.0, lat0=30.0, res=0.05) -> xr.DataArray:
    ny, nx = grid.shape
    xs = lon0 + res * np.arange(nx)
    ys = lat0 + res * np.arange(ny)
    return xr.DataArray(grid.astype("uint8"), coords={"y": ys, "x": xs}, dims=("y", "x"))


def test_patch_class_priority_and_nodata() -> None:
    assert patch_class(np.array([0, 0, 1, 2, 3])) == 3  # compound wins
    assert patch_class(np.array([0, 2, 1])) == 2  # heat over drought
    assert patch_class(np.array([0, 1, 0])) == 1
    assert patch_class(np.array([0, 0, 0])) == 0
    assert patch_class(np.array([255, 255])) is None  # all nodata


def test_keep_by_prob_deterministic_and_order_independent() -> None:
    # Same id+seed -> same decision, regardless of when called.
    a = keep_by_prob("T36RUU_2023-08-01_r00c00", 0.1, seed=1)
    b = keep_by_prob("T36RUU_2023-08-01_r00c00", 0.1, seed=1)
    assert a == b
    # A different seed can differ; prob=1.0 always keeps; prob=0.0 never keeps.
    assert keep_by_prob("x", 1.0) is True
    assert keep_by_prob("x", 0.0) is False


def test_keep_by_prob_ratio_is_approximately_prob() -> None:
    ids = [f"s{i}" for i in range(4000)]
    kept = sum(keep_by_prob(i, 0.25) for i in ids)
    assert 0.22 < kept / len(ids) < 0.28  # ~25% within sampling noise


def test_decide_patch_events_always_kept_background_subsampled() -> None:
    # 3x3 label grid: one compound pixel, rest background.
    grid = np.zeros((3, 3), dtype="uint8")
    grid[1, 1] = 3
    da = _label_da(grid)
    # A bbox covering the whole grid contains the compound pixel -> always kept.
    cls, keep = decide_patch(
        "sid_event", (30.0, 30.0, 30.10, 30.10), da, keep_prob={0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0}
    )
    assert cls == 3 and keep is True
    # A background-only bbox with keep prob 0 for class 0 -> dropped.
    bg = np.zeros((3, 3), dtype="uint8")
    cls, keep = decide_patch(
        "sid_bg",
        (30.0, 30.0, 30.10, 30.10),
        _label_da(bg),
        keep_prob={0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0},
    )
    assert cls == 0 and keep is False


def test_decide_patch_all_nodata_dropped() -> None:
    nod = np.full((3, 3), 255, dtype="uint8")
    cls, keep = decide_patch(
        "s", (30.0, 30.0, 30.10, 30.10), _label_da(nod), keep_prob={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    )
    assert cls is None and keep is False


def test_label_window_falls_back_to_nearest() -> None:
    da = _label_da(np.array([[0, 1], [2, 3]], dtype="uint8"))
    # A bbox between cell centres (spans no centre) -> nearest single pixel.
    win = label_window((30.021, 30.021, 30.024, 30.024), da)
    assert np.asarray(win).size == 1


def test_grid_cells_shape() -> None:
    cells = grid_cells((30.0, 30.0, 31.0, 31.0), n=16)
    assert len(cells) == 256
    _r, _c, bbox = cells[0]
    assert bbox[0] == 30.0 and abs(bbox[2] - bbox[0] - 1 / 16) < 1e-9


def test_selection_report_counts_and_prevalence() -> None:
    # 16x16 label grid: a 2x2 compound block, rest background.
    grid = np.zeros((16, 16), dtype="uint8")
    grid[0:2, 0:2] = 3
    da = _label_da(grid, res=0.0625)  # 16 cells over 1 degree
    rep = selection_report(
        "T36RUU",
        "2023-08-01",
        (30.0, 30.0, 31.0, 31.0),
        da,
        keep_prob={0: 0.1, 1: 0.1, 2: 0.1, 3: 1.0},
        n=16,
    )
    assert rep["n_patches"] == 256
    assert rep["by_class"][3] >= 1  # compound patches present
    assert rep["kept_by_class"][3] == rep["by_class"][3]  # all events kept
    assert rep["n_kept"] <= rep["n_patches"]
    assert 0.0 <= rep["kept_compound_frac"] <= 1.0


@pytest.mark.parametrize("prob", [0.05, 0.2, 0.5])
def test_selection_report_size_scales_with_prob(prob) -> None:
    da = _label_da(np.zeros((16, 16), dtype="uint8"), res=0.0625)  # all background
    rep = selection_report(
        "T",
        "2023-08-01",
        (30.0, 30.0, 31.0, 31.0),
        da,
        keep_prob={0: prob, 1: prob, 2: prob, 3: prob},
        n=16,
    )
    # All-background tile: kept ≈ prob * 256, monotone in prob.
    assert rep["kept_by_class"][0] == rep["n_kept"]
    assert abs(rep["n_kept"] / 256 - prob) < 0.12
