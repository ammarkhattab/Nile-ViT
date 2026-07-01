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


@app.command()
def main(
    out: Path = typer.Option(Path("configs/roi_tiles.json"), help="Output tile-coverage JSON."),
    step_deg: float = typer.Option(0.05, help="Sampling step (degrees)."),
    buffer_deg: float = typer.Option(0.1, help="Bbox expansion for edge tiles."),
    label_raster: Path | None = typer.Option(
        None,
        help="Optional M3 label_*.tif: flag ROI tiles overlapping valid land.",
    ),
) -> None:
    """Enumerate ROI + OOD MGRS tiles and write the coverage JSON."""
    doc = enumerate_roi_tiles(step_deg=step_deg, buffer_deg=buffer_deg)

    if label_raster is not None:
        land = tiles_with_land(doc["roi_tiles"], str(label_raster))
        doc["land_reference"] = str(label_raster)
        doc["n_land_roi_tiles"] = len(land)
        doc["land_roi_tiles"] = land

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
    sample = list(doc["roi_tiles"])[:8]
    typer.echo(f"  e.g. {sample} ...")


if __name__ == "__main__":
    app()
