"""Microsoft Planetary Computer STAC access for HLS v2.0 (streaming, no bulk DL).

Per PRD §10.3 raw HLS is streamed, not bulk-downloaded: 05b (``--source stac``)
resolves each (tile, date) to signed COG hrefs via this module and reads bands on
demand with rioxarray. The ``(tile, date, sensor, {band: href})`` manifest this
builds is the dataset's reproducibility artifact ("re-tile from public STAC").

Confirmed against the live catalog (probe, 2023-08 T36RUU):
  * collections: ``hls2-s30`` (Sentinel-2), ``hls2-l30`` (Landsat); anonymous sign.
  * asset keys are raw band codes B01..B12/B8A + ``Fmask``.
  * item ids look like ``HLS.S30.T36RUU.2023243T082611.v2.0`` -> sensor + tile.

The band-code map lives in ``nilevit/hls_bands.py`` (shared with 05b's disk path),
so S30/L30 logic is single-sourced; all catalog/network calls are injected, so the
pure logic is offline-testable.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from datetime import date
from typing import Any

from nilevit.hls_bands import HLS_BAND_MAP

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
HLS_COLLECTIONS: tuple[str, ...] = ("hls2-s30", "hls2-l30")
FMASK_ASSET = "Fmask"


def sensor_from_id(item_id: str) -> str | None:
    """'HLS.S30.T36RUU.2023243T082611.v2.0' -> 'S30' | 'L30' | None."""
    parts = item_id.split(".")
    if len(parts) >= 2 and parts[1] in ("S30", "L30"):
        return parts[1]
    return None


def tile_from_id(item_id: str) -> str | None:
    """'HLS.S30.T36RUU.2023243T082611.v2.0' -> 'T36RUU' | None."""
    parts = item_id.split(".")
    if len(parts) >= 3 and parts[2].startswith("T"):
        return parts[2]
    return None


def date_from_item(item: Any) -> date:
    """Acquisition date from item.datetime (or properties['datetime'])."""
    dtv = getattr(item, "datetime", None)
    if dtv is not None:
        return dtv.date() if hasattr(dtv, "date") else date.fromisoformat(str(dtv)[:10])
    return date.fromisoformat(str(item.properties["datetime"])[:10])


def item_band_hrefs(
    item: Any, band_map: dict[str, dict[str, str]] | None = None
) -> dict[str, str] | None:
    """Map an item's assets to ``{prithvi_band: href}`` + ``Fmask``.

    Returns None if the sensor is unrecognised or a required band asset is absent.
    Reuses the 05b band map so S30/L30 band codes stay single-sourced.
    """
    if band_map is None:
        band_map = HLS_BAND_MAP
    sensor = sensor_from_id(item.id)
    if sensor is None or sensor not in band_map:
        return None
    assets = item.assets
    out: dict[str, str] = {}
    for band, code in band_map[sensor].items():
        asset = assets.get(code)
        if asset is None:
            return None
        out[band] = asset.href
    fmask = assets.get(FMASK_ASSET)
    if fmask is None:
        return None
    out[FMASK_ASSET] = fmask.href
    return out


def filter_items_for_tile(items: Iterable[Any], tile: str) -> list[Any]:
    """Keep only items whose id encodes exactly ``tile`` (bbox search over-returns)."""
    return [it for it in items if tile_from_id(it.id) == tile]


def open_catalog(stac_url: str = PC_STAC_URL) -> Any:
    """Open the PC STAC catalog with anonymous asset signing (network)."""
    import planetary_computer as pc
    import pystac_client

    return pystac_client.Client.open(stac_url, modifier=pc.sign_inplace)


def search_hls_items(
    catalog: Any,
    tile: str,
    bbox: Sequence[float],
    date_range: str,
    *,
    cloud_max: float = 50.0,
    collections: Sequence[str] = HLS_COLLECTIONS,
) -> list[Any]:
    """Search both HLS collections for one tile/date-range, cloud-filtered (network).

    Filters to the exact tile id (bbox search returns neighbouring tiles too) and
    sorts by acquisition date.
    """
    query = {"eo:cloud_cover": {"lt": cloud_max}}
    found: list[Any] = []
    for coll in collections:
        search = catalog.search(
            collections=[coll],
            bbox=list(bbox),
            datetime=date_range,
            query=query,
        )
        found.extend(filter_items_for_tile(search.items(), tile))
    found.sort(key=lambda it: (date_from_item(it), it.id))
    return found


def manifest_rows_for_items(items: Iterable[Any]) -> Iterator[dict[str, Any]]:
    """Yield ``{tile, date, sensor, item_id, cloud, hrefs}`` for streamable items."""
    for it in items:
        hrefs = item_band_hrefs(it)
        if hrefs is None:
            continue
        yield {
            "tile": tile_from_id(it.id),
            "date": date_from_item(it).isoformat(),
            "sensor": sensor_from_id(it.id),
            "item_id": it.id,
            "cloud": it.properties.get("eo:cloud_cover"),
            "hrefs": hrefs,
        }
