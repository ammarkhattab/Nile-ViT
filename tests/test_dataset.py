"""Offline tests for dataset assembly + M4 acceptance checks (PRD section 4.5)."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

import pytest

from nilevit.dataset import (
    derive_image_path,
    filter_members,
    spatial_leak_violations,
    split_region_year_counts,
    temporal_leak_violations,
    validate_tilesamples,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "data" / "assemble_dataset.py"


def _row(sample_id, lon, lat, split, *, date="2023-08-01", region="delta"):
    return {
        "sample_id": sample_id,
        "mgrs_tile": "T36RUU",
        "date": date,
        "center_lon": lon,
        "center_lat": lat,
        "region": region,
        "cloud_pct": 0.0,
        "valid_pct": 1.0,
        "split": split,
        "label_path": "tiles_T36RUU_2023_labels.zarr::label",
        "meteo_path": "tiles_T36RUU_2023_meteo.zarr::meteo",
    }


def _frame(rows):
    import geopandas as gpd
    from shapely.geometry import Point

    for row in rows:
        row["geometry"] = Point(row["center_lon"], row["center_lat"])
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def test_derive_image_path() -> None:
    assert derive_image_path("T36RUU", dt.date(2023, 8, 1)) == "tiles_T36RUU_2023.zarr::image"


def test_filter_members_excludes_buffer_and_none() -> None:
    gdf = _frame(
        [
            _row("m0", 31.0, 30.7, "train"),
            _row("m1", 24.0, 33.0, "test"),
            _row("b0", 31.1, 30.7, "buffer"),
            _row("n0", 31.2, 29.9, "none"),
        ]
    )
    members, excluded = filter_members(gdf)
    assert set(members["split"]) == {"train", "test"}
    assert len(members) == 2
    assert excluded == {"buffer": 1, "none": 1}
    # Members alone validate as TileSamples (buffer/none never could).
    assert validate_tilesamples(members) == []


def test_validate_tilesamples_passes_and_catches() -> None:
    good = _frame([_row("s0", 31.0, 30.7, "train")])
    assert validate_tilesamples(good) == []

    bad = _frame([_row("s1", 31.0, 30.7, "train")])
    bad.loc[0, "cloud_pct"] = 1.5  # out of [0, 1]
    errors = validate_tilesamples(bad)
    assert len(errors) == 1
    assert errors[0][0] == "s1"


def test_spatial_leak_free_when_far_apart() -> None:
    # Train near the delta, test ~1.5deg east (~140 km) -> no violation.
    gdf = _frame(
        [
            _row("train0", 31.0, 30.7, "train"),
            _row("test0", 32.5, 30.7, "test"),
        ]
    )
    assert spatial_leak_violations(gdf, buffer_km=25.0) == []


def test_spatial_leak_caught_when_too_close() -> None:
    # Test tile ~10 km from a train tile -> a violation the check must report.
    gdf = _frame(
        [
            _row("train0", 31.0, 30.7, "train"),
            _row("test0", 31.1, 30.7, "test"),
        ]
    )
    violations = spatial_leak_violations(gdf, buffer_km=25.0)
    assert len(violations) == 1
    near, far, km = violations[0]
    assert near == "test0" and far == "train0"
    assert km < 25.0


def test_temporal_leak_free_by_year() -> None:
    gdf = _frame(
        [
            _row("a", 31.0, 30.7, "train", date="2022-08-01"),  # temporal train
            _row("b", 31.0, 30.7, "test", date="2023-08-01"),  # temporal test
        ]
    )
    assert temporal_leak_violations(gdf) == []


def test_counts() -> None:
    gdf = _frame(
        [
            _row("a", 31.0, 30.7, "train", region="delta", date="2023-08-01"),
            _row("b", 24.0, 33.0, "test", region="em_shelf", date="2024-08-01"),
        ]
    )
    counts = split_region_year_counts(gdf)
    assert counts["split"] == {"train": 1, "test": 1}
    assert counts["region"] == {"delta": 1, "em_shelf": 1}
    assert counts["year"] == {"2023": 1, "2024": 1}


# --- CLI end-to-end ------------------------------------------------------------
@pytest.mark.skipif(not SCRIPT_PATH.exists(), reason="assemble script not found")
def test_assemble_cli_end_to_end(tmp_path: Path) -> None:
    import geopandas as gpd
    from typer.testing import CliRunner

    # Two sidecars; tiles are far apart -> leak-free, all valid. A buffer tile in
    # the first sidecar must be excluded from the index (not a TileSample).
    side_a = tmp_path / "tiles_T36RUU_2023_labeled.parquet"
    side_b = tmp_path / "tiles_T35RUU_2023_labeled.parquet"
    _frame(
        [
            _row("a", 31.0, 30.7, "train"),
            _row("buf", 31.1, 30.7, "buffer"),
        ]
    ).to_parquet(side_a)
    frame_b = _frame([_row("b", 24.0, 33.0, "test", region="em_shelf")])
    frame_b.loc[0, "mgrs_tile"] = "T35RUU"
    frame_b.to_parquet(side_b)

    out = tmp_path / "dataset_v1.parquet"
    report_out = tmp_path / "report.json"

    spec = importlib.util.spec_from_file_location("assemble_dataset", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = CliRunner().invoke(
        module.app,
        [
            "--sidecar-glob",
            str(tmp_path / "*_labeled.parquet"),
            "--out",
            str(out),
            "--report-out",
            str(report_out),
        ],
    )
    assert result.exit_code == 0, result.output

    merged = gpd.read_parquet(out)
    assert len(merged) == 2  # buffer tile excluded
    assert "buffer" not in set(merged["split"])
    assert "image_path" in merged.columns
    report = json.loads(report_out.read_text())
    assert report["acceptance"]["passed"] is True
    assert report["n_members"] == 2
    assert report["excluded_counts"] == {"buffer": 1}
