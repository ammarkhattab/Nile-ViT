"""Offline tests for 05b's scene resolution (disk paths + STAC dispatch)."""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "data" / "05b_tile.py"


def _load_05b():
    spec = importlib.util.spec_from_file_location("tile05b", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _touch(p: Path) -> None:
    p.write_bytes(b"")


def test_disk_scenes_builds_band_sources(tmp_path) -> None:
    m = _load_05b()
    # One S30 granule (all 6 bands + Fmask) on day 213 = 2023-08-01.
    stem = "HLS.S30.T36RUU.2023213T082611.v2.0"
    for code in ("B02", "B03", "B04", "B8A", "B11", "B12", "Fmask"):
        _touch(tmp_path / f"{stem}.{code}.tif")

    scenes = m._disk_scenes(tmp_path)
    assert list(scenes) == [date(2023, 8, 1)]
    sensor, srcs, fmask = scenes[date(2023, 8, 1)]
    assert sensor == "S30"
    assert Path(srcs["nir_narrow"]).name.endswith(".B8A.tif")  # S30 nir = B8A
    assert set(srcs) == {"blue", "green", "red", "nir_narrow", "swir1", "swir2"}
    assert fmask is not None and Path(fmask).name.endswith(".Fmask.tif")


def test_disk_scenes_fmask_optional(tmp_path) -> None:
    m = _load_05b()
    stem = "HLS.L30.T36RUU.2023213T082611.v2.0"
    for code in ("B02", "B03", "B04", "B05", "B06", "B07"):  # no Fmask
        _touch(tmp_path / f"{stem}.{code}.tif")
    scenes = m._disk_scenes(tmp_path)
    sensor, srcs, fmask = scenes[date(2023, 8, 1)]
    assert sensor == "L30"
    assert Path(srcs["nir_narrow"]).name.endswith(".B05.tif")  # L30 nir = B05
    assert fmask is None


def test_resolve_scenes_dispatches_to_stac(monkeypatch, tmp_path) -> None:
    m = _load_05b()
    sentinel = {date(2023, 8, 1): ("S30", {"red": "https://x/B04.tif"}, "https://x/F.tif")}
    monkeypatch.setattr(m, "_stac_scenes", lambda *a, **k: sentinel)
    out = m._resolve_scenes("stac", tmp_path, "T36RUU", 2023, None, 50.0)
    assert out is sentinel
