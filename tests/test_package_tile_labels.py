"""End-to-end offline test for scripts/labels/package_tile_labels.py.

Builds a tiny synthetic 05b store (a 2-sample Zarr cube + GeoParquet index), an
ROI label raster, and a v1 split map, then runs the packaging CLI and checks the
label store, enriched index, and class-weights report.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "labels" / "package_tile_labels.py"

# Two real T36RUU r00c00 / r00c01 centres (UTM zone 36N, ~Nile delta), plus one
# out-of-ROI centre (lat < 30.0) that the default --drop-out-of-roi must remove.
TILE_CENTRES = [
    ("T36RUU_2023-08-01_r00c00", 0, 0, 30.947267, 30.686391),
    ("T36RUU_2023-08-01_r00c01", 0, 1, 31.017387, 30.687480),
    ("T36RUU_2023-08-01_r09c00", 9, 0, 30.962381, 29.900000),
]


def _build_store(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    import geopandas as gpd
    import numpy as np
    import rioxarray  # noqa: F401  (registers the .rio accessor)
    import xarray as xr
    from shapely.geometry import Point

    size = 16
    sample_ids = [c[0] for c in TILE_CENTRES]

    # Cube: image (sample, band, y, x).
    cube = xr.Dataset(
        {
            "image": (
                ("sample", "band", "y", "x"),
                np.zeros((len(sample_ids), 6, size, size), dtype="uint16"),
            )
        },
        coords={
            "sample": np.array(sample_ids),
            "band": np.array(["blue", "green", "red", "nir_narrow", "swir1", "swir2"]),
        },
    )
    tiles_zarr = tmp_path / "tiles_T36RUU_2023.zarr"
    cube.to_zarr(tiles_zarr, mode="w", consolidated=False)

    # Index parquet.
    records = [
        {
            "sample_id": sid,
            "mgrs_tile": "T36RUU",
            "date": "2023-08-01",
            "row": row,
            "col": col,
            "center_lon": lon,
            "center_lat": lat,
            "region": "delta",
            "cloud_pct": 0.0,
            "valid_pct": 1.0,
            "sensor": "S30",
            "geometry": Point(lon, lat),
        }
        for sid, row, col, lon, lat in TILE_CENTRES
    ]
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    tiles_parquet = tmp_path / "tiles_T36RUU_2023.parquet"
    gdf.to_parquet(tiles_parquet)

    # ROI label raster covering the tile centres, EPSG:4326, all classes present.
    labels_dir = tmp_path / "labels_2023"
    labels_dir.mkdir()
    lon_axis = np.array([30.90, 30.95, 31.00, 31.05])
    lat_axis = np.array([30.75, 30.70, 30.65, 30.60])
    grid = np.array([[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]], dtype="uint8")
    label_da = xr.DataArray(grid, coords={"y": lat_axis, "x": lon_axis}, dims=("y", "x"))
    label_da = label_da.rio.write_crs("EPSG:4326").rio.write_nodata(255)
    # Tile date 2023-08-01 -> active composite is the latest date on/before it.
    label_da.rio.to_raster(labels_dir / "label_2023-07-28.tif")

    # v1 split map covering the tiles' 1deg cell (lon 31, lat 30).
    splits_json = tmp_path / "v1.json"
    splits_json.write_text(
        json.dumps({"buffer_km": 0.0, "cells": {"30_30": "train", "31_30": "train"}}),
        encoding="utf-8",
    )
    return tiles_zarr, tiles_parquet, labels_dir, splits_json


def _load_cli():
    spec = importlib.util.spec_from_file_location("package_tile_labels", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not SCRIPT_PATH.exists(), reason="packaging script not found")
def test_package_tile_labels_end_to_end(tmp_path: Path) -> None:
    import xarray as xr
    from typer.testing import CliRunner

    tiles_zarr, tiles_parquet, labels_dir, splits_json = _build_store(tmp_path)
    weights_out = tmp_path / "class_weights_v1.json"
    module = _load_cli()

    result = CliRunner().invoke(
        module.app,
        [
            "--tiles-zarr",
            str(tiles_zarr),
            "--tiles-parquet",
            str(tiles_parquet),
            "--labels-dir",
            str(labels_dir),
            "--splits-json",
            str(splits_json),
            "--weights-out",
            str(weights_out),
        ],
    )
    assert result.exit_code == 0, result.output

    # Label store written, sample-aligned, uint8 — out-of-ROI tile dropped (3 -> 2).
    label_zarr = tiles_zarr.with_name(f"{tiles_zarr.stem}_labels.zarr")
    label_ds = xr.open_zarr(label_zarr, consolidated=False)
    assert label_ds["label"].dtype == "uint8"
    assert label_ds["label"].sizes == {"sample": 2, "y": 16, "x": 16}

    # Enriched index has the new columns and the right split.
    import geopandas as gpd

    enriched = gpd.read_parquet(tiles_parquet.with_name(f"{tiles_parquet.stem}_labeled.parquet"))
    assert len(enriched) == 2  # the sub-30N tile is gone
    assert {"label_path", "split", "label_date", "label_valid_pct"}.issubset(enriched.columns)
    assert set(enriched["split"]) == {"train"}
    assert "none" not in set(enriched["split"])
    assert set(enriched["label_date"]) == {"2023-07-28"}

    # Class-weights report is well-formed and records the drop.
    report = json.loads(weights_out.read_text())
    assert report["n_samples"] == 3
    assert report["n_dropped_out_of_roi"] == 1
    assert report["n_kept"] == 2
    assert report["n_usable_tiles"] == 2
    assert len(report["class_weights_median_freq"]) == 4
    assert sum(report["counts"].values()) == 2 * 16 * 16


def test_discover_label_rasters(tmp_path: Path) -> None:
    module = _load_cli()
    (tmp_path / "label_2023-08-13.tif").write_bytes(b"")
    (tmp_path / "label_2023-07-28.tif").write_bytes(b"")
    (tmp_path / "ignore.txt").write_bytes(b"")
    found = module._discover_label_rasters(tmp_path)
    assert set(found) == {dt.date(2023, 7, 28), dt.date(2023, 8, 13)}
