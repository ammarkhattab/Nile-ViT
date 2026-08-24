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

import json
import warnings
from collections.abc import Sequence
from pathlib import Path

# §4.1 regions, bbox = (west, south, east, north).
ROI_BBOX: tuple[float, float, float, float] = (22.0, 30.0, 36.0, 37.0)
OOD_BBOX: tuple[float, float, float, float] = (-10.0, 30.0, 12.0, 37.0)

# 100 km-square corners at 1 m precision: SW, SE, NW, NE (easting5 northing5).
_CORNERS = ("0000000000", "9999900000", "0000099999", "9999999999")

# Sampled tile extents are padded so the search box fully covers the 100 km square
# (a tile spans ~0.9-1.1 deg; sampling at 0.05 deg leaves at most half a step short).
_BBOX_PAD_DEG = 0.15

_CONFIG_BBOX_CACHE: dict[str, tuple[float, float, float, float]] | None = None


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


def _corner_bbox(tile_id: str, handle=None) -> tuple[float, float, float, float]:
    """(west, south, east, north) from the 100 km-square corners.

    UNRELIABLE near MGRS band boundaries -- the inverse round-trip clamps to the
    zone's central meridian and can describe a neighbouring square. Kept only as a
    fallback for tiles absent from ``configs/roi_tiles.json``; prefer ``tile_bbox``.
    """
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


def _config_bboxes(path: Path | None = None) -> dict[str, tuple[float, float, float, float]]:
    """Cached tile -> bbox map from configs/roi_tiles.json (ROI + OOD)."""
    global _CONFIG_BBOX_CACHE
    if _CONFIG_BBOX_CACHE is not None:
        return _CONFIG_BBOX_CACHE
    path = path or Path("configs/roi_tiles.json")
    boxes: dict[str, tuple[float, float, float, float]] = {}
    if path.exists():
        doc = json.loads(path.read_text(encoding="utf-8"))
        for key in ("roi_tiles", "ood_tiles"):
            for tile, box in doc.get(key, {}).items():
                boxes[tile] = tuple(box)  # type: ignore[assignment]
    _CONFIG_BBOX_CACHE = boxes
    return boxes


def tile_bbox(tile_id: str, handle=None) -> tuple[float, float, float, float]:
    """Search bbox (west, south, east, north) covering a tile.

    Prefers the enumerated box in ``configs/roi_tiles.json`` (derived from the
    reliable forward MGRS mapping); falls back to the corner round-trip only for
    tiles not in that config. See ``mgrs_tiles_in_bbox`` for why the round-trip is
    not trusted.
    """
    box = _config_bboxes().get(tile_id)
    if box is not None:
        return box
    return _corner_bbox(tile_id, handle)


def mgrs_tiles_in_bbox(
    bbox: Sequence[float], *, step_deg: float = 0.05, buffer_deg: float = 0.1
) -> dict[str, tuple[float, float, float, float]]:
    """Map each MGRS tile intersecting ``bbox`` to a bbox that covers it.

    Samples ``bbox`` (expanded by ``buffer_deg`` so edge tiles are caught) on a
    ``step_deg`` grid and accumulates, per tile, the extent of the sample points
    that fall in it -- then pads by ``_BBOX_PAD_DEG``.

    Why sampled extents rather than the 100 km-square corners: the *forward*
    mapping (``toMGRS``) is reliable, but the inverse round-trip
    (``toLatLon(square + "0000000000")``) mis-resolves squares near MGRS band
    boundaries -- it clamps to the zone's central meridian and returns a
    neighbouring square's footprint. That produced STAC searches which returned
    only foreign tiles, so those tiles silently yielded zero samples. Bboxes here
    are search bounds, and ``hls_stac.filter_items_for_tile`` does the exact
    per-tile selection, so erring slightly large is safe; erring *wrong* is not.
    """
    west, south, east, north = bbox
    handle = _mgrs_handle()

    extents: dict[str, list[float]] = {}
    lat = south - buffer_deg
    while lat <= north + buffer_deg + 1e-9:
        lon = west - buffer_deg
        while lon <= east + buffer_deg + 1e-9:
            tile = tile_for_point(lat, lon, handle)
            if tile is not None:
                box = extents.setdefault(tile, [lon, lat, lon, lat])
                box[0] = min(box[0], lon)
                box[1] = min(box[1], lat)
                box[2] = max(box[2], lon)
                box[3] = max(box[3], lat)
            lon += step_deg
        lat += step_deg

    return {
        tile: (
            round(box[0] - _BBOX_PAD_DEG, 4),
            round(box[1] - _BBOX_PAD_DEG, 4),
            round(box[2] + _BBOX_PAD_DEG, 4),
            round(box[3] + _BBOX_PAD_DEG, 4),
        )
        for tile, box in sorted(extents.items())
    }


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


def tiles_with_land(
    tile_bboxes: dict[str, Sequence[float]],
    label_raster_path: str,
    *,
    nodata: int = 255,
) -> list[str]:
    """Tiles whose bbox overlaps any valid (non-``nodata``) pixel of an M3 label raster.

    The label raster's validity mask (sea / no-data = 255) is the project's own
    land definition, so a tile with no valid pixels yields no useful compound
    labels and need not be downloaded. Order-agnostic coordinate masking (no
    reliance on ascending/descending lat). Lazy rioxarray import.
    """
    import numpy as np
    import rioxarray

    raster = rioxarray.open_rasterio(label_raster_path).squeeze()
    xs = np.asarray(raster["x"].values)
    ys = np.asarray(raster["y"].values)
    values = np.asarray(raster.values)

    land: list[str] = []
    for tile, (west, south, east, north) in tile_bboxes.items():
        x_in = (xs >= west) & (xs <= east)
        y_in = (ys >= south) & (ys <= north)
        if not x_in.any() or not y_in.any():
            continue
        window = values[np.ix_(y_in, x_in)]
        if (window != nodata).any():
            land.append(tile)
    return sorted(land)
