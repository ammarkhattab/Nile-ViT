"""Enumerate the Sentinel-2 / HLS MGRS tiles covering the ROI and OOD regions.

The ROI tile coverage was previously a hand-typed 5-tile dict in
``01_download_hls.py`` -- Nile-Delta only, missing the entire Eastern-Mediterranean
shelf/coast where the §4.4 compound signal lives (and it included one tile,
T35RPN, that sits below the ROI's 30°N edge). This module derives the tile set
programmatically from the MGRS grid so the coverage is complete and correct.

Each tile id is the MGRS 100 km square (e.g. ``"T36RUU"``); the HLS product tile of
the same name covers that square (plus ~5 km overlap). Tiles are enumerated by
sampling the ROI/OOD bbox and collecting the unique squares the points fall in;
per-tile bboxes are the true 100 km-square corners. Pure logic with a lazy
``mgrs`` import -- offline-testable.

Note: the ROI bbox spans open Mediterranean, so the enumerated set is a geometric
superset. Ocean-dominated tiles produce no valid land samples and are pruned
downstream by the M3 land-validity mask (nodata=255); no separate land filter is
applied here.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

# §4.1 regions, bbox = (west, south, east, north).
ROI_BBOX: tuple[float, float, float, float] = (22.0, 30.0, 36.0, 37.0)
OOD_BBOX: tuple[float, float, float, float] = (-10.0, 30.0, 12.0, 37.0)

# 100 km-square corners at 1 m precision: SW, SE, NW, NE (easting5 northing5).
_CORNERS = ("0000000000", "9999900000", "0000099999", "9999999999")


def _mgrs_handle():
    import mgrs

    return mgrs.MGRS()


def tile_for_point(lat: float, lon: float, handle=None) -> str | None:
    """MGRS 100 km square id (with ``T`` prefix) for a point, or None if invalid."""
    handle = handle or _mgrs_handle()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ref = handle.toMGRS(float(lat), float(lon), MGRSPrecision=0)
        except (ValueError, RuntimeError):
            return None
    return f"T{ref}"


def tile_bbox(tile_id: str, handle=None) -> tuple[float, float, float, float]:
    """True (west, south, east, north) of a tile's 100 km square."""
    handle = handle or _mgrs_handle()
    square = tile_id.removeprefix("T")
    lats: list[float] = []
    lons: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for corner in _CORNERS:
            lat, lon = handle.toLatLon(square + corner)
            lats.append(lat)
            lons.append(lon)
    return (
        round(min(lons), 4),
        round(min(lats), 4),
        round(max(lons), 4),
        round(max(lats), 4),
    )


def mgrs_tiles_in_bbox(
    bbox: Sequence[float], *, step_deg: float = 0.05, buffer_deg: float = 0.1
) -> dict[str, tuple[float, float, float, float]]:
    """Map each MGRS tile intersecting ``bbox`` to its true 100 km-square bbox.

    Samples ``bbox`` (expanded by ``buffer_deg`` so edge tiles are caught) on a
    ``step_deg`` grid, collects the unique squares, and computes each tile's bbox.
    """
    west, south, east, north = bbox
    handle = _mgrs_handle()

    tiles: set[str] = set()
    lat = south - buffer_deg
    while lat <= north + buffer_deg + 1e-9:
        lon = west - buffer_deg
        while lon <= east + buffer_deg + 1e-9:
            tile = tile_for_point(lat, lon, handle)
            if tile is not None:
                tiles.add(tile)
            lon += step_deg
        lat += step_deg

    return {tile: tile_bbox(tile, handle) for tile in sorted(tiles)}


def enumerate_roi_tiles(
    *,
    roi_bbox: Sequence[float] = ROI_BBOX,
    ood_bbox: Sequence[float] = OOD_BBOX,
    step_deg: float = 0.05,
    buffer_deg: float = 0.1,
) -> dict:
    """Full ROI + OOD tile coverage document for ``configs/roi_tiles.json``."""
    roi = mgrs_tiles_in_bbox(roi_bbox, step_deg=step_deg, buffer_deg=buffer_deg)
    ood = mgrs_tiles_in_bbox(ood_bbox, step_deg=step_deg, buffer_deg=buffer_deg)
    # ROI and OOD bboxes are disjoint in longitude; keep them strictly separate.
    ood = {tile: box for tile, box in ood.items() if tile not in roi}
    return {
        "version": "v1",
        "roi_bbox": list(roi_bbox),
        "ood_bbox": list(ood_bbox),
        "step_deg": step_deg,
        "buffer_deg": buffer_deg,
        "n_roi_tiles": len(roi),
        "n_ood_tiles": len(ood),
        "note": (
            "Geometric superset; ocean-dominated tiles are pruned downstream by the "
            "M3 land-validity mask (nodata=255). Tile id = MGRS 100km square."
        ),
        "roi_tiles": roi,
        "ood_tiles": ood,
    }
