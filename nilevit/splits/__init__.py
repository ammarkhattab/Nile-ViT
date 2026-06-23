"""Spatial-block cross-validation and temporal splits for Nile-ViT (PRD §4.5).

Pure-stdlib and fully offline-testable. The split is defined on a 1deg x 1deg
grid over the ROI (not on samples), so it is deterministic from the grid plus a
seed -- no HLS scenes needed. The dataset loader (M4+) imports
:func:`split_for_sample` to assign every tile-time sample a split by the cell it
falls in, with a leakage buffer near differently-assigned cell boundaries.

PRD recap (section 4.5):
  1. Tile the ROI on a 1deg x 1deg grid.
  2. Randomly assign each cell to {train, val, test} = 70/15/15, seed=20260519;
     persist the cell->split map as configs/splits/v1.json.
  3. Within a cell, all tile-time samples inherit the cell's split.
  4. Temporal hold-out (R5): a second file uses 2017-2022 train, 2023 test,
     2024 OOD-time.
  5. OOD-space (R6): all NW-Africa tiles get their own split "ood".

Leakage guard: the literal acceptance check ("no test (lon,lat) within 25 km of
any train (lon,lat)") cannot hold under naive inheritance when a test cell abuts
a train cell -- two points either side of the shared edge can be metres apart.
:func:`split_for_sample` therefore returns ``"buffer"`` (excluded from
train/eval) for any sample within ``buffer_km`` of a differently-assigned
neighbouring cell. The cell->split JSON itself stays pure per the PRD.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

# --- constants (PRD section 4.1, 4.5) ------------------------------------------
ROI_BBOX: tuple[float, float, float, float] = (22.0, 30.0, 36.0, 37.0)
OOD_BBOX: tuple[float, float, float, float] = (-10.0, 30.0, 12.0, 37.0)
SEED: int = 20260519
CELL_SIZE_DEG: float = 1.0
RATIO: tuple[float, float, float] = (0.70, 0.15, 0.15)  # train, val, test
BUFFER_KM: float = 25.0
EARTH_RADIUS_KM: float = 6371.0088

SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")

# Temporal hold-out years (PRD section 4.5 step 4).
TEMPORAL_TRAIN_YEARS: tuple[int, ...] = (2017, 2018, 2019, 2020, 2021, 2022)
TEMPORAL_TEST_YEARS: tuple[int, ...] = (2023,)
TEMPORAL_OOD_YEARS: tuple[int, ...] = (2024,)


# --- geometry ------------------------------------------------------------------
def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in kilometres between two lon/lat points."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def cell_sw_corner(lon: float, lat: float, cell_size: float = CELL_SIZE_DEG) -> tuple[int, int]:
    """Integer south-west corner (lon, lat) of the cell containing the point."""
    step = int(cell_size)
    return (
        math.floor(lon / cell_size) * step,
        math.floor(lat / cell_size) * step,
    )


def cell_key(lon: float, lat: float, cell_size: float = CELL_SIZE_DEG) -> str:
    """Stable string key for a cell, e.g. ``"22_30"`` or ``"-10_30"``."""
    sw_lon, sw_lat = cell_sw_corner(lon, lat, cell_size)
    return f"{sw_lon}_{sw_lat}"


def parse_cell_key(key: str) -> tuple[int, int]:
    """Inverse of :func:`cell_key`: ``"-10_30" -> (-10, 30)``."""
    lon_str, lat_str = key.split("_")
    return int(lon_str), int(lat_str)


def bbox_cells(bbox: Sequence[float], cell_size: float = CELL_SIZE_DEG) -> list[tuple[int, int]]:
    """Integer SW corners of every whole cell fully inside ``bbox`` (W,S,E,N)."""
    west, south, east, north = bbox
    step = int(cell_size)
    return [
        (lon, lat)
        for lon in range(math.floor(west), math.ceil(east), step)
        for lat in range(math.floor(south), math.ceil(north), step)
        if lon + step <= east and lat + step <= north
    ]


def point_to_cell_min_km(
    lon: float, lat: float, sw_lon: int, sw_lat: int, cell_size: float
) -> float:
    """Minimum great-circle distance (km) from a point to a cell rectangle."""
    nearest_lon = min(max(lon, sw_lon), sw_lon + cell_size)
    nearest_lat = min(max(lat, sw_lat), sw_lat + cell_size)
    return haversine_km(lon, lat, nearest_lon, nearest_lat)


# --- cell assignment -----------------------------------------------------------
def assign_cells(
    *,
    bbox: Sequence[float] = ROI_BBOX,
    ood_bbox: Sequence[float] = OOD_BBOX,
    cell_size: float = CELL_SIZE_DEG,
    ratio: tuple[float, float, float] = RATIO,
    seed: int = SEED,
) -> dict[str, str]:
    """Deterministic cell->split map: ROI cells get train/val/test, OOD cells "ood".

    Cells are sorted, shuffled with a seeded RNG, then partitioned by count so the
    ratio is hit exactly (up to rounding) and the result is fully reproducible.
    """
    keys = sorted(f"{lon}_{lat}" for lon, lat in bbox_cells(bbox, cell_size))
    shuffled = keys[:]
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = round(ratio[0] * n)
    n_val = round(ratio[1] * n)

    mapping: dict[str, str] = {}
    for key in shuffled[:n_train]:
        mapping[key] = "train"
    for key in shuffled[n_train : n_train + n_val]:
        mapping[key] = "val"
    for key in shuffled[n_train + n_val :]:
        mapping[key] = "test"

    if ood_bbox is not None:
        for lon, lat in bbox_cells(ood_bbox, cell_size):
            mapping[f"{lon}_{lat}"] = "ood"

    return mapping


def split_for_sample(
    lon: float,
    lat: float,
    cell_map: Mapping[str, str],
    *,
    cell_size: float = CELL_SIZE_DEG,
    buffer_km: float = BUFFER_KM,
) -> str:
    """Split for a sample at ``(lon, lat)``.

    Returns the containing cell's split, or ``"buffer"`` if the point is within
    ``buffer_km`` of a neighbouring cell with a different split, or ``"none"`` if
    the point lies in no defined cell. Because surviving samples are kept clear of
    differently-assigned cell rectangles, any two surviving samples from different
    splits are at least ``buffer_km`` apart.
    """
    sw_lon, sw_lat = cell_sw_corner(lon, lat, cell_size)
    base = cell_map.get(f"{sw_lon}_{sw_lat}")
    if base is None:
        return "none"
    if buffer_km <= 0.0:
        return base

    step = int(cell_size)
    for d_lon in (-step, 0, step):
        for d_lat in (-step, 0, step):
            if d_lon == 0 and d_lat == 0:
                continue
            nbr_lon, nbr_lat = sw_lon + d_lon, sw_lat + d_lat
            neighbour = cell_map.get(f"{nbr_lon}_{nbr_lat}")
            if neighbour is None or neighbour == base:
                continue
            if point_to_cell_min_km(lon, lat, nbr_lon, nbr_lat, cell_size) < buffer_km:
                return "buffer"
    return base


# --- temporal split ------------------------------------------------------------
def temporal_split_for_year(year: int) -> str:
    """Temporal-holdout split for a calendar year (PRD section 4.5 step 4)."""
    if year in TEMPORAL_TRAIN_YEARS:
        return "train"
    if year in TEMPORAL_TEST_YEARS:
        return "test"
    if year in TEMPORAL_OOD_YEARS:
        return "ood_time"
    return "none"


def temporal_split_for_date(date_str: str) -> str:
    """Temporal-holdout split for an ISO ``YYYY-MM-DD`` date string."""
    return temporal_split_for_year(int(date_str[:4]))


# --- serialisable documents (what the CLI writes) ------------------------------
def build_spatial_split_doc(
    *,
    seed: int = SEED,
    buffer_km: float = BUFFER_KM,
    cell_size: float = CELL_SIZE_DEG,
) -> dict:
    """Build the ``configs/splits/v1.json`` document."""
    cell_map = assign_cells(seed=seed, cell_size=cell_size)
    counts: dict[str, int] = {}
    for value in cell_map.values():
        counts[value] = counts.get(value, 0) + 1
    ordered = dict(sorted(cell_map.items(), key=lambda kv: parse_cell_key(kv[0])))
    return {
        "version": "v1",
        "kind": "spatial_block_cv",
        "description": (
            "PRD section 4.5 spatial-block CV. Samples inherit their 1deg cell's "
            "split; a leakage buffer (buffer_km) excludes samples near "
            "differently-assigned neighbouring cells."
        ),
        "seed": seed,
        "cell_size_deg": cell_size,
        "ratio": {"train": RATIO[0], "val": RATIO[1], "test": RATIO[2]},
        "roi_bbox": list(ROI_BBOX),
        "ood_bbox": list(OOD_BBOX),
        "buffer_km": buffer_km,
        "counts": counts,
        "cells": ordered,
    }


def build_temporal_split_doc() -> dict:
    """Build the ``configs/splits/v1_temporal.json`` document."""
    years = range(
        min(TEMPORAL_TRAIN_YEARS),
        max(TEMPORAL_OOD_YEARS) + 1,
    )
    return {
        "version": "v1_temporal",
        "kind": "temporal_holdout",
        "description": (
            "PRD section 4.5 temporal hold-out (R5): 2017-2022 train, 2023 test, "
            "2024 OOD-time. Samples are assigned by their calendar year."
        ),
        "rule": "by_calendar_year",
        "train_years": list(TEMPORAL_TRAIN_YEARS),
        "test_years": list(TEMPORAL_TEST_YEARS),
        "ood_time_years": list(TEMPORAL_OOD_YEARS),
        "assignment": {str(y): temporal_split_for_year(y) for y in years},
    }
