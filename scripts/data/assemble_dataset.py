# ruff: noqa: B008
"""Assemble the canonical dataset index and run M4 acceptance checks (PRD section 4.5).

Globs the per-(tile, year) labelled sidecar parquets, consolidates them into one
GeoParquet index with `image_path` added, validates every row against the
section 4.3 TileSample contract, and runs the section 4.5 spatial leak-check on the
real tile coordinates. Fails (non-zero exit) if any TileSample is invalid or any
test tile is within the leakage buffer of a train tile -- the M4 binary acceptance
("splits assigned, pytest leak-check passes").

Usage:
    uv run python scripts/data/assemble_dataset.py \
        --sidecar-glob "data/interim/*_labeled.parquet" \
        --out data/processed/dataset_v1.parquet
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
from pathlib import Path

import typer

from nilevit.dataset import (
    consolidate_index,
    derive_image_path,
    filter_members,
    spatial_leak_violations,
    split_region_year_counts,
    temporal_leak_violations,
    validate_tilesamples,
)
from nilevit.splits import BUFFER_KM, temporal_split_for_date

with contextlib.suppress(ImportError):
    import nilevit  # noqa: F401

app = typer.Typer(add_completion=False, help=__doc__)

MAX_REPORTED = 10


@app.command()
def main(
    sidecar_glob: str = typer.Option(
        "data/interim/*_labeled.parquet",
        help="Glob for the labelled per-(tile, year) sidecar parquets.",
    ),
    out: Path = typer.Option(
        Path("data/processed/dataset_v1.parquet"),
        help="Output consolidated dataset index.",
    ),
    report_out: Path = typer.Option(
        Path("data/processed/dataset_v1_report.json"),
        help="Acceptance + distribution report.",
    ),
    buffer_km: float = typer.Option(BUFFER_KM, help="Spatial leakage buffer (km)."),
    strict: bool = typer.Option(
        True,
        "--strict/--no-strict",
        help="Exit non-zero if validation or the leak-check fails.",
    ),
) -> None:
    """Consolidate, validate, and leak-check the tiled dataset."""
    from glob import glob

    paths = sorted(Path(p) for p in glob(sidecar_glob))
    if not paths:
        raise typer.BadParameter(f"no sidecars matched {sidecar_glob!r}")

    gdf = consolidate_index(paths)
    gdf["image_path"] = [
        derive_image_path(str(mgrs), dt.date.fromisoformat(str(date)))
        for mgrs, date in zip(gdf["mgrs_tile"], gdf["date"], strict=True)
    ]

    # §4.5 defines TWO independent protocols; keep them in separate columns.
    #   `split`          - spatial-block CV (train/val/test; `ood` = the NW-Africa
    #                      OOD region, which is only populated once those tiles are
    #                      tiled - see R6).
    #   `temporal_split` - temporal holdout (R5): 2017-2022 train, 2023 test,
    #                      2024 ood_time. Composing them into one column would
    #                      destroy the spatial split, so they stay orthogonal.
    gdf["temporal_split"] = [
        temporal_split_for_date(dt.date.fromisoformat(str(date))) for date in gdf["date"]
    ]

    # Buffer/none tiles are §4.5 leakage spacing, not dataset members -> excluded.
    n_total = len(gdf)
    gdf, excluded_counts = filter_members(gdf)

    schema_errors = validate_tilesamples(gdf)
    spatial = spatial_leak_violations(gdf, buffer_km=buffer_km)
    temporal = temporal_leak_violations(gdf)
    counts = split_region_year_counts(gdf)
    temporal_counts = {str(k): int(v) for k, v in gdf["temporal_split"].value_counts().items()}

    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out)

    passed = not schema_errors and not spatial and not temporal
    report = {
        "n_sidecars": len(paths),
        "n_tiles_total": n_total,
        "n_members": len(gdf),
        "n_excluded": int(sum(excluded_counts.values())),
        "excluded_counts": excluded_counts,
        "counts": counts,
        "temporal_split_counts": temporal_counts,
        "acceptance": {
            "schema_valid": not schema_errors,
            "spatial_leak_free": not spatial,
            "temporal_leak_free": not temporal,
            "passed": passed,
        },
        "n_schema_errors": len(schema_errors),
        "n_spatial_violations": len(spatial),
        "schema_errors_sample": schema_errors[:MAX_REPORTED],
        "spatial_violations_sample": [
            {"test": a, "train": b, "km": round(km, 3)} for a, b, km in spatial[:MAX_REPORTED]
        ],
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    typer.echo(
        f"wrote {out}  ({len(gdf)} member tiles from {len(paths)} sidecars; "
        f"{sum(excluded_counts.values())} excluded {excluded_counts or ''})"
    )
    typer.echo(f"wrote {report_out}")
    typer.echo(f"  split:  {counts['split']}")
    typer.echo(f"  region: {counts['region']}")
    typer.echo(f"  year:   {counts['year']}")
    typer.echo(f"  temporal_split (R5): {temporal_counts}")
    typer.echo(
        "  acceptance: "
        f"schema={'PASS' if not schema_errors else f'FAIL ({len(schema_errors)})'}, "
        f"spatial_leak={'PASS' if not spatial else f'FAIL ({len(spatial)})'}, "
        f"temporal_leak={'PASS' if not temporal else f'FAIL ({len(temporal)})'}"
    )
    if schema_errors:
        typer.echo(f"  first schema error: {schema_errors[0]}")
    if spatial:
        near, far, km = spatial[0]
        typer.echo(f"  first spatial violation: {near} <-> {far} = {km:.2f} km")

    if strict and not passed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
