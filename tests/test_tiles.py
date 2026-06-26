"""Offline tests for the M4 per-tile packaging helpers (PRD section 4.3)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from nilevit.tiles import (
    aggregate_counts,
    class_weights_from_counts,
    label_date_for,
    label_histogram,
    resample_label_to_tile,
    valid_fraction,
)


# --- pure statistics -----------------------------------------------------------
def test_label_histogram_ignores_nodata() -> None:
    arr = np.array([[0, 1, 1], [3, 255, 2]], dtype=np.uint8)
    assert label_histogram(arr) == {0: 1, 1: 2, 2: 1, 3: 1}


def test_valid_fraction() -> None:
    arr = np.array([0, 0, 255, 255], dtype=np.uint8)
    assert valid_fraction(arr) == pytest.approx(0.5)
    assert valid_fraction(np.array([], dtype=np.uint8)) == 0.0


def test_class_weights_median_freq_upweights_rare_class() -> None:
    counts = {0: 9000, 1: 500, 2: 400, 3: 100}  # compound is rarest
    weights = class_weights_from_counts(counts, scheme="median_freq")
    assert len(weights) == 4
    # Rarer classes get larger weights; the rarest (compound) the largest.
    assert weights[3] > weights[1] > weights[0]
    assert weights[3] == max(weights)


def test_class_weights_inverse_and_absent_classes() -> None:
    counts = {0: 100, 1: 0, 2: 0, 3: 100}
    weights = class_weights_from_counts(counts, scheme="inverse")
    assert weights[1] == 0.0 and weights[2] == 0.0  # absent -> zero weight
    assert weights[0] == pytest.approx(weights[3])  # equal counts -> equal weight


def test_class_weights_empty_and_bad_scheme() -> None:
    assert class_weights_from_counts({0: 0, 1: 0, 2: 0, 3: 0}) == [0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="scheme"):
        class_weights_from_counts({0: 1}, scheme="nope")


def test_aggregate_counts() -> None:
    merged = aggregate_counts([{0: 1, 3: 2}, {0: 4, 1: 5}])
    assert merged == {0: 5, 1: 5, 2: 0, 3: 2}


# --- date selection ------------------------------------------------------------
def test_label_date_for_picks_active_composite() -> None:
    dates = [dt.date(2023, 8, 13), dt.date(2023, 8, 29), dt.date(2023, 7, 28)]
    # 2023-08-20 falls in the composite that started 2023-08-13.
    assert label_date_for(dt.date(2023, 8, 20), dates) == dt.date(2023, 8, 13)
    # Exactly on a composite start -> that date.
    assert label_date_for(dt.date(2023, 8, 29), dates) == dt.date(2023, 8, 29)
    # Predates all -> earliest available.
    assert label_date_for(dt.date(2023, 1, 1), dates) == dt.date(2023, 7, 28)


def test_label_date_for_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        label_date_for(dt.date(2023, 1, 1), [])


# --- geospatial resampler (nearest, categorical, no-data filled) ---------------
def _coarse_label() -> object:
    import rioxarray  # noqa: F401  (registers the .rio accessor)
    import xarray as xr

    # 4x4 ROI label at 0.05deg with all four classes, EPSG:4326.
    data = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]], dtype=np.uint8)
    lon = np.array([22.025, 22.075, 22.125, 22.175])
    lat = np.array([30.175, 30.125, 30.075, 30.025])  # descending (north-up)
    da = xr.DataArray(data, coords={"y": lat, "x": lon}, dims=("y", "x"))
    da = da.rio.write_crs("EPSG:4326")
    return da.rio.write_nodata(255)


def _fine_template() -> object:
    import rioxarray  # noqa: F401  (registers the .rio accessor)
    import xarray as xr

    # Finer destination grid fully inside the coarse extent, EPSG:4326.
    lon = np.linspace(22.03, 22.17, 8)
    lat = np.linspace(30.17, 30.03, 8)
    data = np.zeros((8, 8), dtype=np.float32)
    da = xr.DataArray(data, coords={"y": lat, "x": lon}, dims=("y", "x"))
    return da.rio.write_crs("EPSG:4326")


def test_resample_label_to_tile_is_nearest_and_categorical() -> None:
    out = resample_label_to_tile(_coarse_label(), _fine_template())
    values = np.asarray(out.values)
    assert out.dtype == np.dtype("uint8")
    assert values.shape == (8, 8)
    # Nearest resampling introduces no new class codes (no interpolation).
    assert set(np.unique(values)).issubset({0, 1, 2, 3, 255})
    # All four original classes survive onto the finer grid.
    assert {0, 1, 2, 3}.issubset(set(np.unique(values)))
