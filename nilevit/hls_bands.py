"""Canonical HLS band definitions (single source of truth).

The 6 Prithvi surface-reflectance bands in PRD §4.3 order, mapped to HLS v2.0
asset codes per sensor. S30 (Sentinel-2) and L30 (Landsat) diverge on
nir_narrow/swir1/swir2 (the classic HLS gotcha): S30 uses B8A/B11/B12, L30 uses
B05/B06/B07. Confirmed against the live PC catalog (asset keys are raw band codes).

Both ``scripts/data/05b_tile.py`` (disk tiling) and ``nilevit/hls_stac.py`` (STAC
streaming) consume these, so the mapping is defined exactly once.
"""

from __future__ import annotations

BAND_NAMES: list[str] = ["blue", "green", "red", "nir_narrow", "swir1", "swir2"]

HLS_BAND_MAP: dict[str, dict[str, str]] = {
    "S30": {
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "nir_narrow": "B8A",
        "swir1": "B11",
        "swir2": "B12",
    },
    "L30": {
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "nir_narrow": "B05",
        "swir1": "B06",
        "swir2": "B07",
    },
}

HLS_FILL = -9999  # HLS scaled-reflectance fill value
FMASK_FILL = 255  # HLS Fmask no-observation value
# Fmask cloud-related bits: bit1 cloud (2), bit2 adjacent (4), bit3 shadow (8).
FMASK_CLOUD_BITS = 0b1110  # = 14
