"""Probe the Microsoft Planetary Computer STAC catalog for HLS.

Confirms — against the LIVE catalog (network) — the facts the streaming layer
(`nilevit/hls_stac.py`) must hardcode correctly:
  * the HLS collection id(s) on PC (S30 = Sentinel-2, L30 = Landsat),
  * the per-item ASSET KEYS (the band names we map to the 6 Prithvi bands + Fmask),
  * that `planetary_computer.sign` works anonymously (no PC_SDK_SUBSCRIPTION_KEY),
  * that a (tile-bbox, month) search actually returns items with cloud metadata.

Run this once; paste the output back so the mapping constants are built from
reality rather than memory. Read-only; downloads nothing.

Usage:
    uv run python scripts/data/probe_pc_hls.py
    uv run python scripts/data/probe_pc_hls.py --tile T36RUU --month 2023-08
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import typer

with contextlib.suppress(ImportError):
    import nilevit  # noqa: F401

app = typer.Typer(add_completion=False, help=__doc__)

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
# Prithvi's 6 surface-reflectance bands + the quality mask we need per §4.3.
WANT_BANDS = ("blue", "green", "red", "nir_narrow", "swir1", "swir2", "Fmask")


def _tile_bbox(tile: str) -> tuple[float, float, float, float]:
    """Read the tile's bbox from configs/roi_tiles.json, or fall back to T36RUU."""
    cfg = Path("configs/roi_tiles.json")
    if cfg.exists():
        doc = json.loads(cfg.read_text())
        for key in ("roi_tiles", "ood_tiles"):
            if tile in doc.get(key, {}):
                return tuple(doc[key][tile])  # type: ignore[return-value]
    return (30.911, 29.814, 31.965, 30.729)  # T36RUU fallback


@app.command()
def main(
    tile: str = typer.Option("T36RUU", "--tile", help="MGRS tile to probe."),
    month: str = typer.Option("2023-08", "--month", help="YYYY-MM to search."),
) -> None:
    """Open PC STAC, find HLS collections, and dump one item's asset keys."""
    import planetary_computer as pc
    import pystac_client

    year, mon = (int(x) for x in month.split("-"))
    last = 28 if mon == 2 else (30 if mon in (4, 6, 9, 11) else 31)
    date_range = f"{year:04d}-{mon:02d}-01/{year:04d}-{mon:02d}-{last:02d}"
    bbox = _tile_bbox(tile)
    typer.echo(f"tile={tile} bbox={bbox} range={date_range}")

    # 1) Anonymous signing?
    try:
        catalog = pystac_client.Client.open(PC_STAC_URL, modifier=pc.sign_inplace)
        typer.echo("sign_inplace: OK (anonymous)")
    except Exception as exc:
        typer.echo(f"sign_inplace FAILED: {exc!r}")
        raise typer.Exit(code=1) from exc

    # 2) Which collections are HLS?
    hls_collections = []
    try:
        for coll in catalog.get_collections():
            if "hls" in coll.id.lower() or "hls" in (coll.title or "").lower():
                hls_collections.append(coll.id)
                typer.echo(f"  HLS collection: {coll.id!r}  ({coll.title})")
    except Exception as exc:
        typer.echo(f"collection listing FAILED: {exc!r}")
    if not hls_collections:
        typer.echo("no HLS collections matched 'hls' — check catalog manually.")

    # 3) Search each HLS collection over (bbox, month); dump one item's assets.
    for coll_id in hls_collections:
        typer.echo(f"\n=== search {coll_id} ===")
        try:
            search = catalog.search(collections=[coll_id], bbox=bbox, datetime=date_range, limit=5)
            items = list(search.items())
        except Exception as exc:
            typer.echo(f"  search FAILED: {exc!r}")
            continue
        typer.echo(f"  items returned: {len(items)}")
        if not items:
            continue
        it = items[0]
        typer.echo(f"  first item id: {it.id}")
        typer.echo(f"  datetime: {it.properties.get('datetime')}")
        cloud = it.properties.get("eo:cloud_cover")
        typer.echo(f"  eo:cloud_cover: {cloud}")
        # MGRS / sensor hints in properties (name varies by collection).
        hints = {
            k: v
            for k, v in it.properties.items()
            if any(t in k.lower() for t in ("mgrs", "tile", "sentinel", "landsat"))
        }
        typer.echo(f"  tile/sensor properties: {hints}")
        typer.echo(f"  ASSET KEYS: {sorted(it.assets)}")
        # Show which of our wanted bands appear verbatim as asset keys.
        present = [b for b in WANT_BANDS if b in it.assets]
        typer.echo(f"  of {list(WANT_BANDS)} present verbatim: {present}")
        # Peek one asset href to confirm signing produced a URL.
        any_key = next(iter(it.assets))
        head = it.assets[any_key].href[:90]
        typer.echo(f"  sample asset[{any_key}].href head: {head}")


if __name__ == "__main__":
    app()
