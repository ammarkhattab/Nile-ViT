"""Offline tests for MGRS tile enumeration over the ROI/OOD (PRD §4.1)."""

from __future__ import annotations

import importlib.util

import pytest

from nilevit.roi_tiles import (
    enumerate_roi_tiles,
    mgrs_tiles_in_bbox,
    tile_bbox,
    tile_for_point,
    tiles_with_land,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mgrs") is None, reason="mgrs library not installed"
)


def test_tile_for_point_known_locations() -> None:
    assert tile_for_point(30.5, 31.6) == "T36RUU"  # central Nile Delta
    assert tile_for_point(33.0, -5.0) == "T30SUB"  # NW Africa


def test_tile_bbox_is_the_100km_square() -> None:
    west, south, east, north = tile_bbox("T36RUU")
    # ~1deg square straddling 30N, centred near 31.4E.
    assert 30.8 < west < 31.1
    assert 29.7 < south < 29.9
    assert 31.9 < east < 32.1
    assert 30.6 < north < 30.8


def test_small_bbox_enumeration() -> None:
    # A 0.2deg box in the central Delta resolves to a handful of tiles incl T36RUU.
    tiles = mgrs_tiles_in_bbox((31.4, 30.4, 31.6, 30.6), step_deg=0.05, buffer_deg=0.05)
    assert "T36RUU" in tiles
    for box in tiles.values():
        assert len(box) == 4 and box[0] < box[2] and box[1] < box[3]


def test_enumerate_roi_covers_delta_and_emed_and_separates_ood() -> None:
    doc = enumerate_roi_tiles(step_deg=0.1, buffer_deg=0.1)  # coarse for speed
    roi, ood = doc["roi_tiles"], doc["ood_tiles"]

    # Delta tiles present...
    assert "T36RUU" in roi
    # ...and Eastern-Mediterranean tiles too (the compound-bearing region).
    assert any(t.startswith("T35S") for t in roi)
    # Far more than the old 5-tile hand list.
    assert doc["n_roi_tiles"] > 50
    # OOD is NW-Africa and strictly disjoint from the ROI set.
    assert any(t.startswith(("T29", "T30")) for t in ood)
    assert set(roi).isdisjoint(set(ood))
    # The old hand list's out-of-ROI tile is correctly absent.
    assert "T35RPN" not in roi


def test_tiles_with_land_prunes_sea(tmp_path) -> None:
    import numpy as np
    import rioxarray  # noqa: F401
    import xarray as xr

    # Label raster: west half land (class 0), east half sea (nodata 255).
    lon = np.linspace(30.0, 32.0, 41)
    lat = np.linspace(30.0, 31.0, 21)
    grid = np.full((len(lat), len(lon)), 255, dtype="uint8")
    grid[:, lon < 31.0] = 0  # land on the western side
    da = xr.DataArray(grid, coords={"y": lat, "x": lon}, dims=("y", "x"))
    da = da.rio.write_crs("EPSG:4326")
    raster = tmp_path / "label.tif"
    da.rio.to_raster(raster)

    bboxes = {
        "T_LAND": (30.2, 30.2, 30.8, 30.8),  # over the land half
        "T_SEA": (31.2, 30.2, 31.8, 30.8),  # over the sea half
        "T_OUT": (40.0, 30.2, 41.0, 30.8),  # outside the raster entirely
    }
    land = tiles_with_land(bboxes, str(raster))
    assert land == ["T_LAND"]
