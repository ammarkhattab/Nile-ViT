# ruff: noqa: B008
"""07 -- generate spatial-block CV and temporal split files (PRD section 4.5).

Writes two deterministic JSON files (no HLS scenes required):

  configs/splits/v1.json           cell -> {train, val, test, ood} on a 1deg grid
  configs/splits/v1_temporal.json  year -> {train, test, ood_time}

The split is a property of the 1deg ROI grid and the seed, so this run is
idempotent: re-running with the same seed/buffer reproduces byte-identical files.

Usage (from the repo root):
    uv run python scripts/data/07_make_splits.py
    uv run python scripts/data/07_make_splits.py --seed 20260519 --buffer-km 25
    uv run python scripts/data/07_make_splits.py --no-temporal
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import typer

from nilevit.splits import (
    BUFFER_KM,
    SEED,
    build_spatial_split_doc,
    build_temporal_split_doc,
)

with contextlib.suppress(ImportError):
    import nilevit  # noqa: F401

app = typer.Typer(add_completion=False, help=__doc__)


def _write_json(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


@app.command()
def main(
    out_dir: Path = typer.Option(
        Path("configs/splits"),
        help="Directory for the split JSON files.",
    ),
    seed: int = typer.Option(SEED, help="RNG seed for cell assignment."),
    buffer_km: float = typer.Option(
        BUFFER_KM,
        help="Leakage buffer (km) recorded for the loader's sample assignment.",
    ),
    temporal: bool = typer.Option(
        True,
        "--temporal/--no-temporal",
        help="Also write the temporal hold-out file (v1_temporal.json).",
    ),
) -> None:
    """Generate the spatial-block CV (and temporal hold-out) split files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    spatial = build_spatial_split_doc(seed=seed, buffer_km=buffer_km)
    spatial_path = out_dir / "v1.json"
    _write_json(spatial_path, spatial)

    counts = spatial["counts"]
    roi_total = sum(counts.get(name, 0) for name in ("train", "val", "test"))
    typer.echo(f"wrote {spatial_path}")
    typer.echo(
        f"  ROI cells: {roi_total} "
        f"(train={counts.get('train', 0)}, "
        f"val={counts.get('val', 0)}, "
        f"test={counts.get('test', 0)})  |  ood cells: {counts.get('ood', 0)}"
    )
    if roi_total:
        typer.echo(
            "  fractions: "
            f"train={counts.get('train', 0) / roi_total:.3f}, "
            f"val={counts.get('val', 0) / roi_total:.3f}, "
            f"test={counts.get('test', 0) / roi_total:.3f}"
        )
    typer.echo(f"  seed={seed}  buffer_km={buffer_km}")

    if temporal:
        temporal_doc = build_temporal_split_doc()
        temporal_path = out_dir / "v1_temporal.json"
        _write_json(temporal_path, temporal_doc)
        typer.echo(f"wrote {temporal_path}")
        typer.echo(
            "  years: "
            f"train={temporal_doc['train_years']}, "
            f"test={temporal_doc['test_years']}, "
            f"ood_time={temporal_doc['ood_time_years']}"
        )


if __name__ == "__main__":
    app()
