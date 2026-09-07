"""HLS v2.0 STAC access for streaming (no bulk download), from PC or NASA CMR.

Per PRD §10.3 raw HLS is streamed, not bulk-downloaded: 05b (``--source stac``)
resolves each (tile, date) to COG hrefs via this module and reads bands on demand
with rioxarray. The ``(tile, date, sensor, {band: href})`` manifest this builds is
the dataset's reproducibility artifact ("re-tile from public STAC").

TWO catalogs, because neither covers the full span:
  * ``pc``  - Microsoft Planetary Computer. Anonymous, fast, but its ``hls2-*``
              collections start **2020-01-01** (confirmed from collection extents).
  * ``cmr`` - NASA CMR / LP DAAC. Covers 2013+ (L30) and 2015+ (S30), so it is the
              only source for 2017-2019. Assets live behind Earthdata auth, so GDAL
              must be pointed at a ``~/_netrc`` (see ``configure_earthdata_gdal``).

Granule ids are identical across both (``HLS.S30.T36RUU.2023243T082611.v2.0``), and
both expose raw band codes + ``Fmask`` as assets, so every downstream helper here
(sensor/tile parsing, band mapping, window selection) works unchanged. The band-code
map lives in ``nilevit/hls_bands.py``, shared with 05b's disk path.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from nilevit.hls_bands import HLS_BAND_MAP

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
CMR_STAC_URL = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
HLS_COLLECTIONS: tuple[str, ...] = ("hls2-s30", "hls2-l30")
CMR_COLLECTIONS: tuple[str, ...] = ("HLSS30.v2.0", "HLSL30.v2.0")
FMASK_ASSET = "Fmask"

# PC's HLS collections begin here; earlier years must come from CMR.
PC_FIRST_YEAR = 2020

CATALOG_SPECS: dict[str, dict[str, Any]] = {
    "pc": {
        "url": PC_STAC_URL,
        "collections": HLS_COLLECTIONS,
        "sign": True,  # planetary_computer.sign_inplace
        "server_side_cloud": True,  # honours the `query` extension
    },
    "cmr": {
        "url": CMR_STAC_URL,
        "collections": CMR_COLLECTIONS,
        "sign": False,  # Earthdata auth happens at GDAL read time
        "server_side_cloud": False,  # filter client-side; CMR STAC is limited
    },
}


def catalog_for_year(year: int) -> str:
    """'pc' for years PC hosts (>=2020), else 'cmr' (NASA LP DAAC)."""
    return "pc" if year >= PC_FIRST_YEAR else "cmr"


def configure_earthdata_gdal(cookie_file: str | None = None) -> None:
    """Point GDAL at the Earthdata ``_netrc`` so protected LP DAAC COGs can stream.

    LP DAAC assets sit behind an Earthdata Login redirect; without cookie handling
    GDAL receives the HTML login page and reports "not recognized as a supported
    file format". Credentials come from ``~/_netrc`` (Windows) or ``~/.netrc``,
    which ``earthaccess.login(persist=True)`` writes. Idempotent.
    """
    cookies = cookie_file or str(Path.home() / ".gdal_earthdata_cookies.txt")
    os.environ.setdefault("GDAL_HTTP_COOKIEFILE", cookies)
    os.environ.setdefault("GDAL_HTTP_COOKIEJAR", cookies)
    os.environ.setdefault("GDAL_HTTP_NETRC", "YES")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")


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


def open_catalog(catalog: str = "pc", stac_url: str | None = None) -> Any:
    """Open a STAC catalog: ``pc`` (anonymous signing) or ``cmr`` (Earthdata auth).

    For ``cmr`` this also configures GDAL for Earthdata so the protected COGs can be
    streamed later by rioxarray.
    """
    import pystac_client

    spec = CATALOG_SPECS[catalog]
    url = stac_url or spec["url"]
    if spec["sign"]:
        import planetary_computer as pc

        return pystac_client.Client.open(url, modifier=pc.sign_inplace)
    configure_earthdata_gdal()
    return pystac_client.Client.open(url)


def cloud_cover(item: Any) -> float:
    """``eo:cloud_cover`` as a float; 100.0 when absent (treated as fully cloudy)."""
    value = item.properties.get("eo:cloud_cover")
    return 100.0 if value is None else float(value)


def _search_with_retry(catalog: Any, kwargs: dict[str, Any], *, attempts: int, delay: float):
    """Run one STAC search, retrying transient catalog errors.

    NASA CMR intermittently answers with an HTML error page instead of JSON under
    load, which pystac_client surfaces as ``APIError: <html>``. Retrying with
    exponential backoff clears it; without this a single blip kills a whole tile.
    """
    import time

    def _try_search():
        try:
            return list(catalog.search(**kwargs).items()), None
        except Exception as exc:
            return None, exc

    last: Exception | None = None
    for attempt in range(attempts):
        items, last = _try_search()
        if items is not None:
            return items
        if attempt < attempts - 1:
            time.sleep(delay * (2**attempt))
    raise RuntimeError(f"STAC search failed after {attempts} attempts") from last


def search_hls_items(
    catalog: Any,
    tile: str,
    bbox: Sequence[float],
    date_range: str,
    *,
    cloud_max: float = 50.0,
    collections: Sequence[str] | None = None,
    catalog_name: str = "pc",
    attempts: int = 4,
    retry_delay: float = 3.0,
) -> list[Any]:
    """Search the HLS collections for one tile/date-range, cloud-filtered (network).

    Filters to the exact tile id (a bbox search also returns neighbouring tiles) and
    sorts by acquisition date. Cloud filtering is server-side where the catalog
    supports the ``query`` extension (PC) and client-side otherwise (CMR), so the
    result is the same either way. Searches are retried (CMR is flaky under load).
    """
    spec = CATALOG_SPECS[catalog_name]
    colls = list(collections) if collections is not None else list(spec["collections"])
    server_side = bool(spec["server_side_cloud"])

    found: list[Any] = []
    for coll in colls:
        kwargs: dict[str, Any] = {
            "collections": [coll],
            "bbox": list(bbox),
            "datetime": date_range,
        }
        if server_side:
            kwargs["query"] = {"eo:cloud_cover": {"lt": cloud_max}}
        raw = _search_with_retry(catalog, kwargs, attempts=attempts, delay=retry_delay)
        items = filter_items_for_tile(raw, tile)
        if not server_side:
            items = [it for it in items if cloud_cover(it) < cloud_max]
        found.extend(items)
    found.sort(key=lambda it: (date_from_item(it), it.id))
    return found


def hrefs_by_date(items: Iterable[Any]) -> dict[str, dict[str, Any]]:
    """Group streamable items into ``{iso_date: {"sensor", "item_id", "hrefs"}}``.

    One entry per date; the earlier-sorted item wins when both S30 and L30 cover a
    date (callers pass date-sorted items). Mirrors 05b's one-scene-per-date disk
    behaviour so the STAC source is a drop-in for the tiler.
    """
    out: dict[str, dict[str, Any]] = {}
    for it in items:
        hrefs = item_band_hrefs(it)
        if hrefs is None:
            continue
        iso = date_from_item(it).isoformat()
        if iso in out:
            continue
        out[iso] = {
            "sensor": sensor_from_id(it.id),
            "item_id": it.id,
            "hrefs": hrefs,
        }
    return out


def one_scene_per_window(
    items: Iterable[Any],
    composite_dates: Sequence[str],
    *,
    window_days: int = 16,
) -> dict[str, dict[str, Any]]:
    """Pick the least-cloudy HLS scene per MODIS-composite window (D7 temporal rule).

    Each composite date ``d`` defines the window ``[d, d + window_days)``. Among all
    items falling in a window, the one with the lowest ``eo:cloud_cover`` wins - this
    realises the sampling model's "one central date per composite window" and stops
    the ~2-3 day HLS revisit from multiplying samples. Returns
    ``{acquisition_date_iso: {sensor, item_id, hrefs, window, cloud}}`` for the
    chosen scenes only.
    """
    windows = sorted(date.fromisoformat(d) for d in composite_dates)
    best: dict[date, tuple[float, Any]] = {}
    for it in items:
        acq = date_from_item(it)
        cloud = cloud_cover(it)
        window = next((w for w in windows if w <= acq < w + timedelta(days=window_days)), None)
        if window is None:
            continue
        if window not in best or cloud < best[window][0]:
            best[window] = (cloud, it)

    out: dict[str, dict[str, Any]] = {}
    for window, (cloud, it) in best.items():
        hrefs = item_band_hrefs(it)
        if hrefs is None:
            continue
        out[date_from_item(it).isoformat()] = {
            "sensor": sensor_from_id(it.id),
            "item_id": it.id,
            "hrefs": hrefs,
            "window": window.isoformat(),
            "cloud": cloud,
        }
    return out


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
