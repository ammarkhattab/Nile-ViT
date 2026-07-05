"""Offline test for the select_samples driver (synthetic label rasters)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from typer.testing import CliRunner

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "data" / "select_samples.py"


def _load():
    spec = importlib.util.spec_from_file_location("select_samples", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_label(path: Path, grid: np.ndarray, lon0=30.0, lat0=30.0, res=0.05) -> None:
    ny, nx = grid.shape
    xs = lon0 + res * np.arange(nx)
    ys = lat0 + res * np.arange(ny)
    da = xr.DataArray(grid.astype("uint8"), coords={"y": ys, "x": xs}, dims=("y", "x"))
    da = da.rio.write_crs("EPSG:4326")
    da.rio.to_raster(path)


def test_select_samples_end_to_end(tmp_path) -> None:
    m = _load()

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    grid = np.zeros((20, 20), dtype="uint8")
    grid[0:4, 0:4] = 3  # a compound block
    grid[5:7, 5:7] = 2  # heat
    _write_label(labels_dir / "label_2023-08-13.tif", grid)
    _write_label(labels_dir / "label_2023-08-29.tif", np.zeros((20, 20), "uint8"))

    roi = tmp_path / "roi_tiles.json"
    roi.write_text(
        json.dumps(
            {
                "roi_tiles": {"T_A": [30.0, 30.0, 31.0, 31.0]},
                "land_roi_tiles": ["T_A"],
            }
        )
    )

    out = tmp_path / "keeplist.json"
    report_out = tmp_path / "report.json"
    result = CliRunner().invoke(
        m.app,
        [
            "--labels-dir",
            str(labels_dir),
            "--roi-tiles",
            str(roi),
            "--keep-drought",
            "0.1",
            "--years",
            "8",
            "--out",
            str(out),
            "--report-out",
            str(report_out),
        ],
    )
    assert result.exit_code == 0, result.output

    report = json.loads(report_out.read_text())
    assert report["n_composite_dates"] == 2
    assert report["n_land_tiles"] == 1
    assert report["kept_by_class"]["3"] >= 1  # all compound kept
    assert report["est_full_samples"] == report["n_kept_this_year"] * 8
    # Keep-list has entries for the compound date.
    keeplist = json.loads(out.read_text())
    assert "T_A" in keeplist
    assert "2023-08-13" in keeplist["T_A"]
