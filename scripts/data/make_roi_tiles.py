# ruff: noqa: B008
"""Generate configs/roi_tiles.json -- the canonical ROI + OOD MGRS tile coverage.

Replaces the hand-typed (Delta-only) KNOWN_TILES in 01_download_hls.py with the
complete tile set covering the §4.1 ROI (Nile Delta + Eastern Mediterranean) and
the NW-Africa OOD region, enumerated from the MGRS grid. Deterministic and
idempotent.

Usage:
    uv run python scripts/data/make_roi_tiles.py
    uv run python scripts/data/make_roi_tiles.py --out configs/roi_tiles.json
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import typer

from nilevit.roi_tiles import enumerate_roi_tiles, tiles_with_land

with contextlib.suppress(ImportError):
    import nilevit  # noqa: F401

app = typer.Typer(add_completion=False, help=__doc__)


def validate_against_hls(
    tiles: list[str],
    bboxes: dict[str, list[float]],
    *,
    datetime_range: str = "2020-07-01/2020-07-31",
    progress_every: int = 20,
) -> tuple[list[str], list[str]]:
    """Split ``tiles`` into (served_by_hls, phantom) by querying the HLS catalog.

    ``tile_for_point`` can emit MGRS squares the Sentinel-2 / HLS grid never
    produces -- e.g. ``T34REA``/``T34RFA``/``T34RGA`` near the 32°N band edge, where
    the ``*A`` row actually belongs to band S. Those tiles are valid MGRS strings
    with no product behind them, so tiling them always yields zero samples and a
    spurious failure. One catalog pass removes them for good.
    """
    from nilevit.hls_stac import HLS_COLLECTIONS, open_catalog, tile_from_id

    catalog = open_catalog()
    served: list[str] = []
    phantom: list[str] = []
    for i, tile in enumerate(tiles, start=1):
        search = catalog.search(
            collections=list(HLS_COLLECTIONS),
            bbox=list(bboxes[tile]),
            datetime=datetime_range,
        )
        found = {tile_from_id(item.id) for item in search.items()}
        (served if tile in found else phantom).append(tile)
        if progress_every and i % progress_every == 0:
            typer.echo(f"    checked {i}/{len(tiles)} ...")
    return served, phantom


@app.command()
def main(
    out: Path = typer.Option(Path("configs/roi_tiles.json"), help="Output tile-coverage JSON."),
    step_deg: float = typer.Option(0.05, help="Sampling step (degrees)."),
    buffer_deg: float = typer.Option(0.1, help="Bbox expansion for edge tiles."),
    label_raster: Path | None = typer.Option(
        None,
        help="Optional M3 label_*.tif: flag ROI tiles overlapping valid land.",
    ),
    validate_hls: bool = typer.Option(
        False,
        "--validate-hls",
        help="Query the HLS catalog and drop land tiles with no product (network).",
    ),
) -> None:
    """Enumerate ROI + OOD MGRS tiles and write the coverage JSON."""
    doc = enumerate_roi_tiles(step_deg=step_deg, buffer_deg=buffer_deg)

    if label_raster is not None:
        land = tiles_with_land(doc["roi_tiles"], str(label_raster))
        doc["land_reference"] = str(label_raster)
        doc["n_land_roi_tiles"] = len(land)
        doc["land_roi_tiles"] = land

        if validate_hls:
            typer.echo(f"  validating {len(land)} land tiles against the HLS catalog...")
            served, phantom = validate_against_hls(land, doc["roi_tiles"])
            doc["land_roi_tiles"] = served
            doc["n_land_roi_tiles"] = len(served)
            doc["hls_phantom_tiles"] = phantom
            doc["hls_validated"] = True

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    typer.echo(f"wrote {out}")
    typer.echo(f"  ROI tiles: {doc['n_roi_tiles']}  (was 5 hand-typed, Delta-only)")
    typer.echo(f"  OOD tiles: {doc['n_ood_tiles']}  (NW-Africa, for M9 OOD eval)")
    if "n_land_roi_tiles" in doc:
        typer.echo(
            f"  land ROI tiles: {doc['n_land_roi_tiles']}  "
            f"({doc['n_roi_tiles'] - doc['n_land_roi_tiles']} ocean/no-data pruned)"
        )
    if doc.get("hls_phantom_tiles"):
        phantom = doc["hls_phantom_tiles"]
        typer.echo(f"  HLS phantom tiles dropped: {len(phantom)}  {phantom[:8]}")
    sample = list(doc["roi_tiles"])[:8]
    typer.echo(f"  e.g. {sample} ...")


if __name__ == "__main__":
    app()
