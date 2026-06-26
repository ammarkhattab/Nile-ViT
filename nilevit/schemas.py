"""Canonical tile-sample schema for Nile-ViT (PRD section 4.3).

One :class:`TileSample` is one tile-time record: a 224x224 HLS image cube, a
90-day meteo series, a 224x224 label map, quality flags, and the CV split. The
GeoParquet index (05b) and the per-tile label packaging (M4) both conform to
this contract. Pure-Python (pydantic only) so it is fully offline-testable.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field

# --- label classes (PRD section 4.4) -------------------------------------------
CLASS_NAMES: dict[int, str] = {0: "none", 1: "drought", 2: "heat", 3: "compound"}
NUM_CLASSES: int = 4
LABEL_NODATA: int = 255

# --- regions (PRD section 4.1), bbox = (west, south, east, north) --------------
ROI_BBOX: tuple[float, float, float, float] = (22.0, 30.0, 36.0, 37.0)
DELTA_BBOX: tuple[float, float, float, float] = (29.5, 30.5, 32.5, 31.5)
N_COAST_BBOX: tuple[float, float, float, float] = (25.0, 31.0, 35.0, 32.0)
OOD_BBOX: tuple[float, float, float, float] = (-10.0, 30.0, 12.0, 37.0)

Region = Literal["delta", "n_coast", "em_shelf", "ood_nw_africa"]
Split = Literal["train", "val", "test", "ood"]


class TileSample(BaseModel):
    """One tile-time dataset record (PRD section 4.3)."""

    sample_id: str  # e.g. "T36RUU_2023-08-15_R000"
    mgrs_tile: str  # Sentinel-2 / HLS MGRS tile id
    center_lon: float = Field(ge=-180.0, le=180.0)
    center_lat: float = Field(ge=-90.0, le=90.0)
    date: dt.date  # acquisition date (UTC) of the central frame
    region: Region

    # Imagery cube (T=3, C=6, H=224, W=224) uint16 scaled HLS reflectance.
    # Bands: blue, green, red, nir_narrow, swir1, swir2; T = t-30d, t, t+15d.
    image_path: str  # path inside the Zarr store

    # Meteo series (T_m=90, V=7) float32, 90 days ending at `date`.
    # V: era5_t2m, era5_swvl1, era5_e, era5_tp, chirps_p, chirts_tmax, chirts_tmin
    meteo_path: str

    # Pixel labels (H=224, W=224) uint8 in {0,1,2,3}, nodata 255.
    label_path: str

    # Quality flags.
    cloud_pct: float = Field(ge=0.0, le=1.0)  # from S2/HLS QA
    valid_pct: float = Field(ge=0.0, le=1.0)  # fraction of finite pixels
    split: Split


def _in_bbox(lon: float, lat: float, bbox: tuple[float, float, float, float]) -> bool:
    west, south, east, north = bbox
    return west <= lon <= east and south <= lat <= north


def region_for_point(lon: float, lat: float) -> Region:
    """Map a tile centre to its PRD section 4.1 region.

    Precedence ``delta > n_coast > em_shelf`` reflects "em_shelf = ROI minus the
    delta and coast sub-regions". NW-Africa (disjoint from the ROI) is checked
    first. Raises if the point is in neither the ROI nor the OOD region.
    """
    if _in_bbox(lon, lat, OOD_BBOX):
        return "ood_nw_africa"
    if _in_bbox(lon, lat, DELTA_BBOX):
        return "delta"
    if _in_bbox(lon, lat, N_COAST_BBOX):
        return "n_coast"
    if _in_bbox(lon, lat, ROI_BBOX):
        return "em_shelf"
    raise ValueError(f"point ({lon}, {lat}) is outside the ROI and OOD regions")
