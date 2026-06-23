"""Offline tests for spatial-block CV + temporal splits (PRD section 4.5)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from nilevit.splits import (
    BUFFER_KM,
    CELL_SIZE_DEG,
    OOD_BBOX,
    ROI_BBOX,
    SEED,
    assign_cells,
    bbox_cells,
    build_spatial_split_doc,
    build_temporal_split_doc,
    cell_key,
    haversine_km,
    parse_cell_key,
    split_for_sample,
    temporal_split_for_date,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "data" / "07_make_splits.py"


# --- geometry ------------------------------------------------------------------
def test_haversine_one_degree_latitude() -> None:
    # One degree of latitude is ~111 km anywhere on Earth.
    assert haversine_km(30.0, 30.0, 30.0, 31.0) == pytest.approx(111.2, abs=0.5)


def test_cell_key_roundtrip_handles_negative_lon() -> None:
    assert cell_key(22.4, 30.9) == "22_30"
    assert cell_key(-9.1, 30.0) == "-10_30"
    assert parse_cell_key("-10_30") == (-10, 30)
    assert parse_cell_key("35_36") == (35, 36)


# --- grid enumeration ----------------------------------------------------------
def test_roi_grid_is_14x7() -> None:
    cells = bbox_cells(ROI_BBOX)
    assert len(cells) == 98
    lons = {lon for lon, _ in cells}
    lats = {lat for _, lat in cells}
    assert lons == set(range(22, 36))  # 22..35
    assert lats == set(range(30, 37))  # 30..36


def test_ood_grid_is_22x7() -> None:
    assert len(bbox_cells(OOD_BBOX)) == 154


# --- assignment ----------------------------------------------------------------
def test_assignment_is_deterministic() -> None:
    assert assign_cells(seed=SEED) == assign_cells(seed=SEED)


def test_assignment_ratio_and_coverage() -> None:
    mapping = assign_cells(seed=SEED)
    counts = dict.fromkeys(("train", "val", "test", "ood"), 0)
    for value in mapping.values():
        counts[value] += 1
    # Every ROI cell assigned exactly once, plus all OOD cells.
    assert counts["train"] + counts["val"] + counts["test"] == 98
    assert counts["ood"] == 154
    # 70/15/15 up to integer rounding (98 -> 69/15/14).
    assert counts["train"] == 69
    assert counts["val"] == 15
    assert counts["test"] == 14
    # No ROI cell ever labelled "ood" and vice versa.
    for lon, lat in bbox_cells(ROI_BBOX):
        assert mapping[f"{lon}_{lat}"] in ("train", "val", "test")
    for lon, lat in bbox_cells(OOD_BBOX):
        assert mapping[f"{lon}_{lat}"] == "ood"


def test_different_seed_changes_assignment() -> None:
    assert assign_cells(seed=SEED) != assign_cells(seed=SEED + 1)


# --- buffer behaviour ----------------------------------------------------------
def test_sample_outside_all_cells_is_none() -> None:
    mapping = assign_cells(seed=SEED)
    # A point far west of both ROI and OOD grids.
    assert split_for_sample(-40.0, 30.5, mapping) == "none"


def test_buffer_disabled_returns_raw_cell_split() -> None:
    mapping = assign_cells(seed=SEED)
    # Centre of a cell, buffer off -> always the cell's own split.
    for lon, lat in bbox_cells(ROI_BBOX):
        raw = split_for_sample(lon + 0.5, lat + 0.5, mapping, buffer_km=0.0)
        assert raw == mapping[f"{lon}_{lat}"]


# --- ACCEPTANCE (PRD section 4.5): no test sample within 25 km of any train ----
def _grid_points(spacing_deg: float = 0.25):
    west, south, east, north = ROI_BBOX
    lon = west
    while lon < east:
        lat = south
        while lat < north:
            yield (round(lon, 6), round(lat, 6))
            lat += spacing_deg
        lon += spacing_deg


def test_no_test_sample_within_25km_of_train() -> None:
    mapping = assign_cells(seed=SEED)
    train_pts, test_pts = [], []
    for lon, lat in _grid_points():
        split = split_for_sample(lon, lat, mapping, buffer_km=BUFFER_KM)
        if split == "train":
            train_pts.append((lon, lat))
        elif split == "test":
            test_pts.append((lon, lat))

    assert train_pts, "expected some train samples on the grid"
    assert test_pts, "expected some test samples on the grid"

    for t_lon, t_lat in test_pts:
        for r_lon, r_lat in train_pts:
            assert haversine_km(t_lon, t_lat, r_lon, r_lat) >= BUFFER_KM


# --- ACCEPTANCE (PRD section 4.5): temporal hold-out has no date leakage --------
def test_temporal_no_date_overlap() -> None:
    dates = [f"{year}-07-15" for year in range(2017, 2025)]
    by_split: dict[str, set[str]] = {}
    for date in dates:
        by_split.setdefault(temporal_split_for_date(date), set()).add(date)
    assert by_split["train"].isdisjoint(by_split["test"])
    assert temporal_split_for_date("2023-08-01") == "test"
    assert temporal_split_for_date("2024-08-01") == "ood_time"


# --- documents -----------------------------------------------------------------
def test_spatial_doc_is_complete_and_ordered() -> None:
    doc = build_spatial_split_doc()
    assert doc["seed"] == SEED
    assert doc["cell_size_deg"] == CELL_SIZE_DEG
    assert doc["buffer_km"] == BUFFER_KM
    assert len(doc["cells"]) == 98 + 154
    # Cells are sorted by (lon, lat) for stable diffs.
    keys = list(doc["cells"])
    assert keys == sorted(keys, key=parse_cell_key)


def test_temporal_doc_matches_year_rule() -> None:
    doc = build_temporal_split_doc()
    assert doc["assignment"]["2022"] == "train"
    assert doc["assignment"]["2023"] == "test"
    assert doc["assignment"]["2024"] == "ood_time"


# --- CLI smoke test (skips cleanly if the numeric-named script is absent) -------
@pytest.mark.skipif(not SCRIPT_PATH.exists(), reason="07 script not found")
def test_cli_writes_both_files(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    spec = importlib.util.spec_from_file_location("make_splits_07", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = CliRunner().invoke(module.app, ["--out-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output

    spatial = json.loads((tmp_path / "v1.json").read_text())
    temporal = json.loads((tmp_path / "v1_temporal.json").read_text())
    assert len(spatial["cells"]) == 98 + 154
    assert temporal["assignment"]["2023"] == "test"
