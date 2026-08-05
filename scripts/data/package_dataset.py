# ruff: noqa: B008
"""Package every tiled (tile, year) - labels then meteo - resumably, then aggregate.

Runs `package_tile_labels.py` and `package_tile_meteo.py` over each
`data/interim/tiles_<TILE>_<YEAR>.parquet` produced by 05b / tile_dataset.

Why an aggregation pass: both per-tile scripts write GLOBAL config artifacts
(`configs/class_weights_v1.json`, `configs/meteo_norm_v1.json`) from their own tile
only, so running them 72x leaves each file reflecting whichever tile ran last. This
driver points the per-tile runs at throwaway scratch paths, then recomputes both
artifacts over ALL packaged tiles:
  * class weights - pool the per-tile class histograms, then `class_weights_from_counts`.
  * meteo norm    - refit `meteo_channel_stats` over every TRAIN-split series.

Usage:
  uv run python scripts/data/package_dataset.py --year 2023 --dry-run
  uv run python scripts/data/package_dataset.py --year 2023
  uv run python scripts/data/package_dataset.py --year 2023 --tiles T36SXA,T36SYB
  uv run python scripts/data/package_dataset.py --year 2023 --aggregate-only
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

with contextlib.suppress(ImportError):
    import nilevit  # noqa: F401

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()

TRAIN_SPLIT = "train"
USABLE_SPLITS = ("train", "val", "test", "ood")


def tiled_inputs(interim: Path, year: int, subset: set[str] | None) -> list[tuple[str, Path]]:
    """Find (tile, parquet) for every 05b index of the given year."""
    out: list[tuple[str, Path]] = []
    for path in sorted(interim.glob(f"tiles_*_{year}.parquet")):
        stem = path.stem  # tiles_<TILE>_<YEAR>
        if stem.endswith("_labeled"):
            continue
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        tile = parts[1]
        if subset and tile not in subset:
            continue
        out.append((tile, path))
    return out


def labels_cmd(
    py: str,
    scripts: Path,
    tile: str,
    year: int,
    interim: Path,
    labels_dir: Path,
    splits_json: Path,
    scratch: Path,
) -> list[str]:
    """package_tile_labels invocation for one tile (weights -> scratch)."""
    return [
        py,
        str(scripts / "labels" / "package_tile_labels.py"),
        "--tiles-zarr",
        str(interim / f"tiles_{tile}_{year}.zarr"),
        "--tiles-parquet",
        str(interim / f"tiles_{tile}_{year}.parquet"),
        "--labels-dir",
        str(labels_dir),
        "--splits-json",
        str(splits_json),
        "--weights-out",
        str(scratch / f"weights_{tile}_{year}.json"),
    ]


def meteo_cmd(
    py: str,
    scripts: Path,
    tile: str,
    year: int,
    interim: Path,
    era5_dir: Path,
    chirps_dir: Path,
    scratch: Path,
) -> list[str]:
    """package_tile_meteo invocation for one tile (norm -> scratch)."""
    return [
        py,
        str(scripts / "data" / "package_tile_meteo.py"),
        "--tiles-zarr",
        str(interim / f"tiles_{tile}_{year}.zarr"),
        "--labeled-parquet",
        str(interim / f"tiles_{tile}_{year}_labeled.parquet"),
        "--era5-dir",
        str(era5_dir),
        "--chirps-dir",
        str(chirps_dir),
        "--norm-out",
        str(scratch / f"norm_{tile}_{year}.json"),
    ]


def aggregate_class_weights(scratch: Path, year: int, out: Path) -> dict:
    """Pool per-tile class histograms -> global weights (PRD §5.2 class-weighted loss)."""
    from nilevit.tiles import class_weights_from_counts

    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    n_tiles = 0
    for path in sorted(scratch.glob(f"weights_*_{year}.json")):
        report = json.loads(path.read_text())
        for cls, value in report.get("counts", {}).items():
            counts[int(cls)] += int(value)
        n_tiles += 1
    total = sum(counts.values())
    doc = {
        "scope": f"global over {n_tiles} packaged tiles ({year})",
        "n_tiles": n_tiles,
        "counts": counts,
        "selected_sample_compound_fraction": (counts[3] / total if total else 0.0),
        "note": (
            "selected_sample_compound_fraction is a DATASET statistic (D7 enrichment), "
            "NOT the §4.4 gate - run scripts/data/check_gate.py for that."
        ),
        "class_weights_median_freq": class_weights_from_counts(counts, scheme="median_freq"),
        "class_weights_inverse": class_weights_from_counts(counts, scheme="inverse"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


def aggregate_meteo_norm(interim: Path, year: int, out: Path) -> dict:
    """Refit z-score stats over the TRAIN series of every packaged tile."""
    import geopandas as gpd
    import xarray as xr

    from nilevit.meteo import METEO_CHANNELS, meteo_channel_stats

    series = []
    n_tiles = 0
    for parquet in sorted(interim.glob(f"tiles_*_{year}_labeled.parquet")):
        tile = parquet.stem.split("_")[1]
        store = interim / f"tiles_{tile}_{year}_meteo.zarr"
        if not store.exists():
            continue
        index = gpd.read_parquet(parquet).set_index("sample_id")
        dataset = xr.open_zarr(store, consolidated=False)
        order = [str(s) for s in dataset["sample"].values]
        values = dataset["meteo"].values
        for i, sample_id in enumerate(order):
            if sample_id in index.index and str(index.loc[sample_id, "split"]) == TRAIN_SPLIT:
                series.append(values[i])
        dataset.close()
        n_tiles += 1

    stats = meteo_channel_stats(series) if series else {}
    doc = {
        "scope": f"global over {n_tiles} packaged tiles ({year})",
        "fit_split": TRAIN_SPLIT,
        "n_train_samples": len(series),
        "channels": list(METEO_CHANNELS),
        "stats": stats,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


@app.command()
def main(
    year: int = typer.Option(2023, "--year", "-y", help="Year to package."),
    interim: Path = typer.Option(Path("data/interim"), help="Tiled-output dir."),
    labels_dir: Path | None = typer.Option(None, help="Defaults to <interim>/labels_<year>."),
    splits_json: Path = typer.Option(Path("configs/splits/v1.json"), help="§4.5 split map."),
    era5_dir: Path | None = typer.Option(None, help="Defaults to data/raw/era5/<year>."),
    chirps_dir: Path | None = typer.Option(None, help="Defaults to data/raw/chirps/<year>."),
    tiles: str | None = typer.Option(None, "--tiles", help="Comma-separated subset."),
    scratch: Path = typer.Option(Path("data/interim/_pkg_scratch"), help="Per-tile stats dir."),
    weights_out: Path = typer.Option(Path("configs/class_weights_v1.json")),
    norm_out: Path = typer.Option(Path("configs/meteo_norm_v1.json")),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan; run nothing."),
    aggregate_only: bool = typer.Option(
        False, "--aggregate-only", help="Skip per-tile runs; just recompute global stats."
    ),
    stop_on_fail: bool = typer.Option(False, "--stop-on-fail"),
) -> None:
    """Package all tiled tiles for a year, then recompute the global stats."""
    labels_dir = labels_dir or interim / f"labels_{year}"
    era5_dir = era5_dir or Path("data/raw/era5") / str(year)
    chirps_dir = chirps_dir or Path("data/raw/chirps") / str(year)
    scripts = Path(__file__).resolve().parents[1]

    subset = {t.strip() for t in tiles.split(",")} if tiles else None
    work = tiled_inputs(interim, year, subset)
    if not work and not aggregate_only:
        console.print(f"[red]ERROR[/red] no tiles_*_{year}.parquet in {interim}")
        raise typer.Exit(code=1)

    plan = Table(title="M4 packaging plan", show_header=False, title_style="bold cyan")
    plan.add_column(style="dim")
    plan.add_column(style="bold")
    plan.add_row("Year", str(year))
    plan.add_row("Tiles to package", str(len(work)))
    plan.add_row("Labels dir", str(labels_dir))
    plan.add_row("ERA5 / CHIRPS", f"{era5_dir} | {chirps_dir}")
    plan.add_row("Mode", "dry-run" if dry_run else ("aggregate-only" if aggregate_only else "run"))
    console.print(plan)
    console.print()

    if dry_run:
        for tile, _ in work[:15]:
            console.print(f"  [{year}] {tile}: labels -> meteo")
        if len(work) > 15:
            console.print(f"  ... and {len(work) - 15} more")
        console.print("\n[yellow]Dry run - nothing packaged.[/yellow]")
        raise typer.Exit(code=0)

    n_ok, n_fail = 0, 0
    if not aggregate_only:
        scratch.mkdir(parents=True, exist_ok=True)
        for tile, _ in work:
            console.rule(f"[bold]{year} {tile}")
            steps = [
                (
                    "labels",
                    labels_cmd(
                        sys.executable,
                        scripts,
                        tile,
                        year,
                        interim,
                        labels_dir,
                        splits_json,
                        scratch,
                    ),
                ),
                (
                    "meteo",
                    meteo_cmd(
                        sys.executable, scripts, tile, year, interim, era5_dir, chirps_dir, scratch
                    ),
                ),
            ]
            ok = True
            for name, cmd in steps:
                result = subprocess.run(cmd, check=False)
                if result.returncode != 0:
                    ok = False
                    console.print(f"[red]FAILED[/red] {tile} {name} (exit {result.returncode})")
                    break
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                if stop_on_fail:
                    console.print("[red]--stop-on-fail set; aborting.[/red]")
                    raise typer.Exit(code=1)

    console.rule("[bold]global aggregation")
    weights = aggregate_class_weights(scratch, year, weights_out)
    norm = aggregate_meteo_norm(interim, year, norm_out)
    console.print(f"wrote {weights_out}  (pooled over {weights['n_tiles']} tiles)")
    console.print(f"  counts: {weights['counts']}")
    console.print(
        f"  selected-sample compound fraction: "
        f"{weights['selected_sample_compound_fraction']:.2%}  "
        f"[dim](dataset stat - §4.4 gate is check_gate.py)[/dim]"
    )
    console.print(
        f"  class_weights median_freq: "
        f"{[round(w, 4) for w in weights['class_weights_median_freq']]}"
    )
    console.print(f"wrote {norm_out}  (z-score on {norm['n_train_samples']} train samples)")

    if not aggregate_only:
        console.print(f"\n[green]Packaging: {n_ok} ok, {n_fail} failed.[/green]")
        if n_fail:
            console.print("[yellow]Re-run to retry failed tiles.[/yellow]")
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
