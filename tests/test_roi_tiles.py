"""Offline tests for MGRS tile enumeration over the ROI/OOD (PRD §4.1)."""

from __future__ import annotations

import importlib.util

import pytest

from nilevit.roi_tiles import (
    _BBOX_PAD_DEG,
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


def _true_square_extent(
    tile: str, around: tuple[float, float, float, float], step: float = 0.01
) -> tuple[float, float, float, float]:
    """The tile's real footprint, via the forward MGRS mapping on a fine grid.

    Forward ``toMGRS`` is the reliable direction (the inverse round-trip is the one
    that mis-resolves band-edge squares), so this is ground truth for the test.
    """
    west, south, east, north = around
    lats: list[float] = []
    lons: list[float] = []
    lat = south
    while lat <= north + 1e-9:
        lon = west
        while lon <= east + 1e-9:
            if tile_for_point(lat, lon) == tile:
                lats.append(lat)
                lons.append(lon)
            lon += step
        lat += step
    assert lats, f"{tile} not found in {around}"
    return min(lons), min(lats), max(lons), max(lats)


def test_tile_bbox_covers_the_100km_square() -> None:
    """``tile_bbox`` is a padded SEARCH box: it must contain the whole square.

    Since the band-boundary fix it returns sampled extents + ``_BBOX_PAD_DEG``
    (via configs/roi_tiles.json), not the raw square, so it is checked for coverage
    and a bounded margin rather than against raw-square corners. Too small would
    drop real scenes; too large is harmless (``filter_items_for_tile`` selects by
    tile id) but would mean the padding logic has drifted.
    """
    west, south, east, north = tile_bbox("T36RUU")
    t_west, t_south, t_east, t_north = _true_square_extent("T36RUU", (30.5, 29.5, 32.5, 31.0))

    # ~1deg square straddling 30N, centred near 31.4E (sanity on the truth itself).
    assert 30.8 < t_west < 31.1
    assert 31.9 < t_east < 32.1

    # Covers the square on every side (0.01 deg = the truth grid's resolution).
    tol = 0.01
    assert west <= t_west + tol
    assert south <= t_south + tol
    assert east >= t_east - tol
    assert north >= t_north - tol

    # ...and is not wildly larger: at most pad + one 0.05 deg sampling step per side.
    max_margin = _BBOX_PAD_DEG + 0.05 + tol
    assert t_west - west <= max_margin
    assert t_south - south <= max_margin
    assert east - t_east <= max_margin
    assert north - t_north <= max_margin


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
