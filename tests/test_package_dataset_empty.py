"""package_dataset must report all-outside-ROI tiles as EMPTY, not FAILED.

Such a tile leaves a zero-row labeled parquet, so meteo has no dates and exits 2.
That used to surface as "N failed" on every run (5 tiles in 2019, 4 in 2018) and
invite pointless retries. Covered here: the row-count probe, the per-tile outcome,
the global pools ignoring empty tiles, and the CLI end to end.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "data" / "package_dataset.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("package_dataset_empty", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["package_dataset_empty"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pkg() -> Any:
    return _load()


def _labeled(path: Path, n: int, split: str = "train") -> Path:
    """Write a GeoParquet shaped like package_tile_labels' output, with ``n`` rows."""
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {
            "sample_id": [f"S{i}" for i in range(n)],
            "split": [split] * n,
            "geometry": [Point(31.0, 30.5)] * n,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    gdf.to_parquet(path)
    return path


def _done(returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode)


# ---- labeled_row_count ---------------------------------------------------------


def test_row_count_missing_file_is_none(pkg: Any, tmp_path: Path) -> None:
    assert pkg.labeled_row_count(tmp_path / "nope.parquet") is None


def test_row_count_reads_metadata(pkg: Any, tmp_path: Path) -> None:
    assert pkg.labeled_row_count(_labeled(tmp_path / "a.parquet", 0)) == 0
    assert pkg.labeled_row_count(_labeled(tmp_path / "b.parquet", 3)) == 3


def test_row_count_unreadable_is_none(pkg: Any, tmp_path: Path) -> None:
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"not a parquet file")
    assert pkg.labeled_row_count(bad) is None


# ---- package_tile --------------------------------------------------------------


STEPS = [("labels", ["labels"]), ("meteo", ["meteo"])]


def test_empty_tile_skips_meteo(pkg: Any, tmp_path: Path) -> None:
    labeled = tmp_path / "tiles_T36SUG_2018_labeled.parquet"
    ran: list[str] = []

    def run(cmd: list[str], check: bool) -> subprocess.CompletedProcess:
        ran.append(cmd[0])
        if cmd[0] == "labels":
            _labeled(labeled, 0)
        return _done()

    assert pkg.package_tile("T36SUG", labeled, STEPS, run=run) == (pkg.EMPTY, None)
    assert ran == ["labels"]  # meteo never invoked


def test_tile_with_rows_runs_both_steps(pkg: Any, tmp_path: Path) -> None:
    labeled = tmp_path / "tiles_T36RUU_2018_labeled.parquet"
    ran: list[str] = []

    def run(cmd: list[str], check: bool) -> subprocess.CompletedProcess:
        ran.append(cmd[0])
        if cmd[0] == "labels":
            _labeled(labeled, 5)
        return _done()

    assert pkg.package_tile("T36RUU", labeled, STEPS, run=run) == (pkg.OK, None)
    assert ran == ["labels", "meteo"]


def test_labels_failure_is_failed(pkg: Any, tmp_path: Path) -> None:
    labeled = tmp_path / "x_labeled.parquet"
    outcome = pkg.package_tile("T", labeled, STEPS, run=lambda cmd, check: _done(1))
    assert outcome == (pkg.FAILED, "labels")


def test_real_meteo_failure_is_still_failed(pkg: Any, tmp_path: Path) -> None:
    """A tile WITH rows whose meteo fails must stay a failure - only empties change."""
    labeled = tmp_path / "x_labeled.parquet"

    def run(cmd: list[str], check: bool) -> subprocess.CompletedProcess:
        if cmd[0] == "labels":
            _labeled(labeled, 4)
            return _done()
        return _done(2)

    assert pkg.package_tile("T", labeled, STEPS, run=run) == (pkg.FAILED, "meteo")


def test_missing_labeled_parquet_still_runs_meteo(pkg: Any, tmp_path: Path) -> None:
    """Unknown row count is not 'empty': meteo runs and reports its own error."""
    ran: list[str] = []

    def run(cmd: list[str], check: bool) -> subprocess.CompletedProcess:
        ran.append(cmd[0])
        return _done()

    pkg.package_tile("T", tmp_path / "absent.parquet", STEPS, run=run)
    assert ran == ["labels", "meteo"]


# ---- global pools ----------------------------------------------------------------


def test_class_weight_pool_ignores_empty_tiles(pkg: Any, tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    full = {"0": 10, "1": 20, "2": 30, "3": 40}
    empty = {"0": 0, "1": 0, "2": 0, "3": 0}
    (scratch / "weights_T36RUU_2018.json").write_text(json.dumps({"counts": full}))
    (scratch / "weights_T36SUG_2018.json").write_text(json.dumps({"counts": empty}))

    doc = pkg.aggregate_class_weights(scratch, 2018, tmp_path / "w.json", force=True)

    assert doc["n_tiles"] == 1
    assert doc["counts"] == {0: 10, 1: 20, 2: 30, 3: 40}


# ---- CLI end to end --------------------------------------------------------------


def test_cli_reports_empty_not_failed(
    pkg: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    interim = tmp_path / "interim"
    interim.mkdir()
    for tile in ("T36RUU", "T36SUG"):
        (interim / f"tiles_{tile}_2018.parquet").write_bytes(b"")  # 05b index marker
    rows = {"T36RUU": 3, "T36SUG": 0}

    def fake_run(cmd: list[str], check: bool) -> subprocess.CompletedProcess:
        script = Path(cmd[1]).name
        zarr_arg = Path(cmd[cmd.index("--tiles-zarr") + 1]).name
        tile = zarr_arg.split("_")[1]
        if script == "package_tile_labels.py":
            _labeled(interim / f"tiles_{tile}_2018_labeled.parquet", rows[tile])
            return _done()
        # package_tile_meteo: the real script exits 2 on an empty tile.
        return _done(2 if rows[tile] == 0 else 0)

    monkeypatch.setattr(pkg.subprocess, "run", fake_run)
    monkeypatch.setattr(pkg, "aggregate_class_weights", lambda *a, **k: _fake_weights())
    monkeypatch.setattr(pkg, "aggregate_meteo_norm", lambda *a, **k: {"n_train_samples": 3})

    result = CliRunner().invoke(
        pkg.app,
        ["--year", "2018", "--interim", str(interim), "--scratch", str(tmp_path / "s")],
    )

    assert result.exit_code == 0, result.output
    assert "EMPTY T36SUG" in result.output
    assert "1 ok, 1 empty" in result.output
    assert "0 failed" in result.output
    assert "Re-run to retry" not in result.output


def _fake_weights() -> dict:
    return {
        "n_tiles": 1,
        "counts": {0: 1, 1: 1, 2: 1, 3: 1},
        "selected_sample_compound_fraction": 0.25,
        "class_weights_median_freq": [1.0, 1.0, 1.0, 1.0],
    }


def test_meteo_norm_pool_ignores_empty_tiles(pkg: Any, tmp_path: Path) -> None:
    """A stale meteo store next to a zero-row labeled parquet must not count as a tile."""
    import numpy as np
    import xarray as xr

    for tile, n in (("T36RUU", 2), ("T36SUG", 0)):
        _labeled(tmp_path / f"tiles_{tile}_2018_labeled.parquet", n)
        meteo = np.zeros((max(n, 1), 90, 7), dtype="float32")
        xr.Dataset(
            {"meteo": (("sample", "t", "channel"), meteo)},
            coords={"sample": [f"S{i}" for i in range(max(n, 1))]},
        ).to_zarr(tmp_path / f"tiles_{tile}_2018_meteo.zarr", mode="w")

    doc = pkg.aggregate_meteo_norm(tmp_path, 2018, tmp_path / "norm.json", force=True)

    assert doc["n_train_samples"] == 2
    assert "2018" in doc["scope"]
    assert doc["scope"].startswith("global over 1 packaged tiles")
