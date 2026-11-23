"""05b must SKIP a tile whose catalog is unreachable, never abort the campaign.

Regression guard for the 2018 pass that lost 46 tiles: ``open_catalog`` sat outside
the try/except around the search, so a CMR ConnectTimeout escaped ``_stac_scenes``
and exited 1 per tile instead of returning no scenes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TILE_SCRIPT = REPO_ROOT / "scripts" / "data" / "05b_tile.py"


def _load_05b() -> Any:
    """Import 05b_tile.py by path (its name is not a valid module identifier)."""
    spec = importlib.util.spec_from_file_location("tile_05b", TILE_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["tile_05b"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tile_05b() -> Any:
    return _load_05b()


def test_stac_scenes_returns_empty_when_catalog_unreachable(
    tile_05b: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A RuntimeError from open_catalog becomes {} - the driver's skip signal."""
    from nilevit import hls_stac

    def _unreachable(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("opening STAC catalog 'cmr' failed after 4 attempts")

    monkeypatch.setattr(hls_stac, "open_catalog", _unreachable)

    scenes = tile_05b._stac_scenes("T37SBT", 2018, None, 50.0)

    assert scenes == {}


def test_stac_scenes_does_not_swallow_unexpected_errors(
    tile_05b: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only RuntimeError means 'unreachable'; a bug must still surface loudly."""
    from nilevit import hls_stac

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise KeyError("unknown catalog")

    monkeypatch.setattr(hls_stac, "open_catalog", _boom)

    with pytest.raises(KeyError):
        tile_05b._stac_scenes("T37SBT", 2018, None, 50.0)


def test_stac_scenes_returns_empty_when_search_fails(
    tile_05b: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-existing search-failure path must keep working after the move."""
    from nilevit import hls_stac

    monkeypatch.setattr(hls_stac, "open_catalog", lambda *_a, **_k: object())

    def _search_fails(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("STAC search failed after 4 attempts")

    monkeypatch.setattr(hls_stac, "search_hls_items", _search_fails)

    assert tile_05b._stac_scenes("T37SBT", 2018, None, 50.0) == {}


def test_stac_scenes_opens_catalog_before_searching(
    tile_05b: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sanity: the happy path still threads the opened catalog into the search."""
    from nilevit import hls_stac

    sentinel = object()
    seen: dict[str, Any] = {}

    monkeypatch.setattr(hls_stac, "open_catalog", lambda *_a, **_k: sentinel)

    def _search(catalog: Any, tile: str, *_args: Any, **_kwargs: Any) -> list[Any]:
        seen["catalog"] = catalog
        seen["tile"] = tile
        return []

    monkeypatch.setattr(hls_stac, "search_hls_items", _search)

    tile_05b._stac_scenes("T37SBT", 2018, None, 50.0)

    assert seen["catalog"] is sentinel
    assert seen["tile"] == "T37SBT"
