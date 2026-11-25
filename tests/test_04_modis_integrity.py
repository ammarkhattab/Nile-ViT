"""04_download_modis must not trust a file just because it exists.

An interrupted run left 9 truncated 2017 MODIS clips that still OPENED (header
intact) but failed at pixel decode ("ZIPDecode ... incorrect header check"). Every
re-run printed ``SKIP (exists)`` and ``build_labels`` later crashed on them. Covered:
the integrity probe on a genuinely truncated deflate GeoTIFF, the skip decision,
and the atomic .part -> rename write.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "data" / "04_download_modis.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("modis_04", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["modis_04"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modis() -> Any:
    return _load()


def _clip(size: int = 256) -> Any:
    """A sinusoidal-ish georeferenced raster like a MODIS ROI clip."""
    import rioxarray  # noqa: F401
    import xarray as xr

    rng = np.random.default_rng(0)
    da = xr.DataArray(
        rng.integers(-2000, 9000, size=(1, size, size), dtype=np.int16),
        dims=("band", "y", "x"),
        coords={
            "band": [1],
            "y": 3_400_000.0 - 231.7 * np.arange(size),
            "x": 2_000_000.0 + 231.7 * np.arange(size),
        },
    )
    return da.rio.write_crs("EPSG:4326")


def _write_good(path: Path) -> Path:
    _clip().rio.to_raster(path, driver="GTiff", compress="deflate")
    return path


def _truncate(path: Path, keep: float = 0.6) -> Path:
    data = path.read_bytes()
    path.write_bytes(data[: int(len(data) * keep)])
    return path


# ---- is_readable_raster ---------------------------------------------------------


def test_good_file_is_readable(modis: Any, tmp_path: Path) -> None:
    assert modis.is_readable_raster(_write_good(tmp_path / "ok.tif"))


def test_truncated_file_opens_but_is_not_readable(modis: Any, tmp_path: Path) -> None:
    """The real failure mode: header intact (opens), pixels unreadable."""
    import rasterio

    bad = _truncate(_write_good(tmp_path / "bad.tif"))
    with rasterio.open(bad):  # opening alone does NOT detect it...
        pass
    assert not modis.is_readable_raster(bad)  # ...a full read does


def test_garbage_file_is_not_readable(modis: Any, tmp_path: Path) -> None:
    junk = tmp_path / "junk.tif"
    junk.write_bytes(b"<html>login page</html>")
    assert not modis.is_readable_raster(junk)


# ---- should_skip_existing --------------------------------------------------------


def test_skip_decisions(modis: Any, tmp_path: Path) -> None:
    good = _write_good(tmp_path / "good.tif")
    bad = _truncate(_write_good(tmp_path / "bad.tif"))
    missing = tmp_path / "missing.tif"

    assert modis.should_skip_existing(good, overwrite=False) == (True, "exists")
    assert modis.should_skip_existing(bad, overwrite=False) == (False, "corrupt")
    assert modis.should_skip_existing(missing, overwrite=False) == (False, "missing")
    assert modis.should_skip_existing(good, overwrite=True) == (False, "overwrite")


# ---- write_atomic ------------------------------------------------------------------


def test_partial_name_never_matches_tif_glob(modis: Any, tmp_path: Path) -> None:
    part = modis.partial_path(tmp_path / "MOD13Q1.A2017145.h21v05.NDVI.nile-em.tif")
    part.write_bytes(b"x")
    assert list(tmp_path.glob("*.tif")) == []


def test_write_atomic_produces_readable_file(modis: Any, tmp_path: Path) -> None:
    out = tmp_path / "MOD11A2.A2017145.h21v05.LST_Day_1km.nile-em.tif"
    modis.write_atomic(_clip(), out)

    assert modis.is_readable_raster(out)
    assert not modis.partial_path(out).exists()


def test_interrupted_write_leaves_no_output(modis: Any, tmp_path: Path) -> None:
    """A crash mid-write must leave neither a final .tif nor a stray .part."""
    out = tmp_path / "MOD13Q1.A2017177.h21v05.NDVI.nile-em.tif"

    class _Dies:
        class rio:  # noqa: N801  (mimics the .rio accessor)
            @staticmethod
            def to_raster(path: Path, **_kwargs: Any) -> None:
                Path(path).write_bytes(b"half a file")
                raise OSError("No space left on device")

    with pytest.raises(OSError):
        modis.write_atomic(_Dies(), out)

    assert not out.exists()
    assert not modis.partial_path(out).exists()


def test_write_atomic_replaces_corrupt_file(modis: Any, tmp_path: Path) -> None:
    out = _truncate(_write_good(tmp_path / "redo.tif"))
    assert not modis.is_readable_raster(out)

    modis.write_atomic(_clip(), out)

    assert modis.is_readable_raster(out)
