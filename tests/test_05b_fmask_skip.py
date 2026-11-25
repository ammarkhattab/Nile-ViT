"""05b must SKIP a date whose Fmask exists but cannot be read.

The old behaviour treated an unreadable Fmask as "all clear", writing every patch of
that date with ``cloud_pct = 0`` whatever the real sky - a silently wrong quality
flag. These tests drive the real ``main`` loop on a synthetic 2x2-patch scene with
the network reads stubbed, and check both sides: unreadable Fmask -> nothing
written and the date counted as unreadable; readable Fmask -> patches written.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TILE_SCRIPT = REPO_ROOT / "scripts" / "data" / "05b_tile.py"

SIDE = 448  # 2 x 2 patches of 224
FMASK_SRC = "fake://fmask"
DAY = date(2017, 5, 9)


def _load_05b() -> Any:
    spec = importlib.util.spec_from_file_location("tile_05b_fmask", TILE_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["tile_05b_fmask"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tile_05b() -> Any:
    return _load_05b()


def _red_grid() -> Any:
    """A georeferenced (1, H, W) raster standing in for the opened red band."""
    import rioxarray  # noqa: F401  (registers the .rio accessor)
    import xarray as xr

    xs = 300_000.0 + 30.0 * np.arange(SIDE) + 15.0
    ys = 3_400_000.0 - 30.0 * np.arange(SIDE) - 15.0
    da = xr.DataArray(
        np.full((1, SIDE, SIDE), 1000, dtype=np.int16),
        dims=("band", "y", "x"),
        coords={"band": [1], "y": ys, "x": xs},
    )
    return da.rio.write_crs("EPSG:32636")


def _run(tile_05b: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, fmask_ok: bool) -> Any:
    from typer.testing import CliRunner

    from nilevit.hls_bands import BAND_NAMES

    sources = {name: f"fake://{name}" for name in BAND_NAMES}
    scenes = {DAY: ("S30", sources, FMASK_SRC)}
    monkeypatch.setattr(tile_05b, "_resolve_scenes", lambda *_a, **_k: scenes)
    monkeypatch.setattr(tile_05b, "open_raster_with_retry", lambda _src, **_k: _red_grid())

    def _read(src: str, **_kwargs: Any) -> np.ndarray:
        if src == FMASK_SRC:
            if not fmask_ok:
                raise RuntimeError(f"failed to read after 5 attempts: {src}")
            return np.zeros((SIDE, SIDE), dtype=np.uint8)
        return np.full((SIDE, SIDE), 1000, dtype=np.int16)

    monkeypatch.setattr(tile_05b, "read_band_with_retry", _read)

    return CliRunner().invoke(
        tile_05b.app,
        [
            "--tile",
            "T36RUU",
            "--year",
            "2017",
            "--source",
            "stac",
            "--catalog",
            "cmr",
            "--all-dates",
            "--output-dir",
            str(tmp_path),
        ],
    )


def test_unreadable_fmask_skips_the_date(
    tile_05b: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = _run(tile_05b, monkeypatch, tmp_path, fmask_ok=False)

    assert result.exit_code == 0, result.output
    assert "Fmask unreadable" in result.output
    assert "1 date(s) unreadable" in result.output
    assert "treating as clear" not in result.output
    # Nothing written: no patch was given a fabricated cloud_pct of 0.
    assert not (tmp_path / "tiles_T36RUU_2017.zarr").exists()
    assert not (tmp_path / "tiles_T36RUU_2017.parquet").exists()


def test_readable_fmask_still_writes_patches(
    tile_05b: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Control: the happy path is unchanged by the fix."""
    import geopandas as gpd

    result = _run(tile_05b, monkeypatch, tmp_path, fmask_ok=True)

    assert result.exit_code == 0, result.output
    assert "Done: 4 written" in result.output
    index = gpd.read_parquet(tmp_path / "tiles_T36RUU_2017.parquet")
    assert len(index) == 4
    assert (index["cloud_pct"] == 0.0).all()
