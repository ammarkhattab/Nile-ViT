"""Dataset assembly, schema validation, and leak-checks (M4 acceptance).

Consolidates the per-(tile, year) labelled sidecar indices into one dataset index,
validates every row against the section 4.3 ``TileSample`` contract, and runs the
section 4.5 leak-checks on real tile coordinates -- the M4 binary acceptance
("splits assigned, pytest leak-check passes"). Because ``split_for_sample`` already
excludes buffer tiles, the spatial leak-check verifies that guard held end-to-end.

Pure helpers (pandas/geopandas + the haversine from nilevit.splits), fully
offline-testable; heavy imports are lazy per the project convention.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from pathlib import Path
from typing import get_args

from nilevit.schemas import Split, TileSample
from nilevit.splits import haversine_km, temporal_split_for_date

# Dataset-member splits (mirrors the §4.3 TileSample.split literal exactly).
MEMBER_SPLITS: tuple[str, ...] = get_args(Split)


def filter_members(gdf):
    """Split the index into dataset members (train/val/test/ood) and excluded rows.

    Buffer/none tiles are §4.5 leakage spacing, not dataset records, so they are
    not ``TileSample``s and must not enter the index. Returns
    ``(members_gdf, excluded_counts)`` where ``excluded_counts`` maps each excluded
    split label to its tile count.
    """
    is_member = gdf["split"].isin(MEMBER_SPLITS)
    members = gdf[is_member].reset_index(drop=True)
    excluded = gdf[~is_member]
    excluded_counts: dict[str, int] = {}
    for split in excluded["split"]:
        excluded_counts[str(split)] = excluded_counts.get(str(split), 0) + 1
    return members, excluded_counts


def derive_image_path(mgrs_tile: str, date: dt.date, *, suffix: str = "image") -> str:
    """Reconstruct the 05b image store reference for a row (the sidecar convention)."""
    return f"tiles_{mgrs_tile}_{date.year}.zarr::{suffix}"


def consolidate_index(parquet_paths: Sequence[Path]):
    """Concatenate labelled sidecar GeoParquets into one GeoDataFrame."""
    import geopandas as gpd
    import pandas as pd

    if not parquet_paths:
        raise ValueError("no sidecar parquet paths given")
    frames = [gpd.read_parquet(path) for path in parquet_paths]
    merged = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=frames[0].crs)


def validate_tilesamples(gdf) -> list[tuple[str, str]]:
    """Validate each row against ``TileSample``; return ``(sample_id, error)`` pairs."""
    from pydantic import ValidationError

    errors: list[tuple[str, str]] = []
    for record in gdf.to_dict("records"):
        sample_id = str(record.get("sample_id", "?"))
        try:
            date = dt.date.fromisoformat(str(record["date"]))
            TileSample(
                sample_id=sample_id,
                mgrs_tile=str(record["mgrs_tile"]),
                center_lon=float(record["center_lon"]),
                center_lat=float(record["center_lat"]),
                date=date,
                region=str(record["region"]),
                image_path=derive_image_path(str(record["mgrs_tile"]), date),
                meteo_path=str(record["meteo_path"]),
                label_path=str(record["label_path"]),
                cloud_pct=float(record["cloud_pct"]),
                valid_pct=float(record["valid_pct"]),
                split=str(record["split"]),
            )
        except (ValidationError, ValueError, KeyError) as exc:
            errors.append((sample_id, str(exc).splitlines()[0]))
    return errors


def spatial_leak_violations(
    gdf,
    *,
    buffer_km: float = 25.0,
    near: str = "test",
    far: str = "train",
) -> list[tuple[str, str, float]]:
    """Pairs ``(near_id, far_id, km)`` where a ``near`` tile is under ``buffer_km``.

    The section 4.5 acceptance: no ``test`` tile within 25 km of any ``train`` tile.
    An empty list means the split is leak-free for that pair.
    """
    near_rows = gdf[gdf["split"] == near]
    far_rows = gdf[gdf["split"] == far]
    far_points = list(
        zip(
            far_rows["sample_id"],
            far_rows["center_lon"],
            far_rows["center_lat"],
            strict=True,
        )
    )

    violations: list[tuple[str, str, float]] = []
    for near_id, lon, lat in zip(
        near_rows["sample_id"],
        near_rows["center_lon"],
        near_rows["center_lat"],
        strict=True,
    ):
        for far_id, far_lon, far_lat in far_points:
            distance = haversine_km(lon, lat, far_lon, far_lat)
            if distance < buffer_km:
                violations.append((str(near_id), str(far_id), distance))
    return violations


def temporal_leak_violations(gdf) -> list[str]:
    """Sample_ids whose date lands in two temporal partitions (should be none).

    The temporal split (section 4.5) is year-based and disjoint, so any sample_id
    that resolves to a date appearing in both the temporal-train and temporal-test
    year sets is a contradiction. Returns the offending sample_ids.
    """
    train_dates: set[str] = set()
    test_dates: set[str] = set()
    for record in gdf.to_dict("records"):
        date_str = str(record["date"])
        partition = temporal_split_for_date(date_str)
        if partition == "train":
            train_dates.add(date_str)
        elif partition == "test":
            test_dates.add(date_str)
    overlap = train_dates & test_dates
    return [
        str(record["sample_id"])
        for record in gdf.to_dict("records")
        if str(record["date"]) in overlap
    ]


def split_region_year_counts(gdf) -> dict[str, dict[str, int]]:
    """Cross-tabulate tiles by split, region, and year for the assembly report."""
    counts: dict[str, dict[str, int]] = {"split": {}, "region": {}, "year": {}}
    for record in gdf.to_dict("records"):
        split = str(record["split"])
        region = str(record["region"])
        year = str(record["date"])[:4]
        counts["split"][split] = counts["split"].get(split, 0) + 1
        counts["region"][region] = counts["region"].get(region, 0) + 1
        counts["year"][year] = counts["year"].get(year, 0) + 1
    return counts
