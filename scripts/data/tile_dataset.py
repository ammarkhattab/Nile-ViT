# ruff: noqa: B008
"""Drive the M4 stream-tiling run: 05b over every land tile x year, resumably.

Calls ``05b_tile.py --source stac --all-dates --labels-dir ...`` for each
(land tile, year), applying the D7 selection (one scene per composite window +
per-class patch keep). Streams HLS from Planetary Computer - no raw scenes on disk.

Resumable: a manifest records completed (tile, year) pairs, and 05b itself skips
already-indexed samples, so a re-run picks up where it left off. A year whose
labels aren't built yet is skipped with a note (labels gate tiling).

Usage
-----
Preview the plan (no tiling):
  uv run python scripts/data/tile_dataset.py --years 2023 --dry-run

Tile all land tiles for 2023 (labels must exist at data/interim/labels_2023):
  uv run python scripts/data/tile_dataset.py --years 2023

Full run once all years' labels exist:
  uv run python scripts/data/tile_dataset.py --years 2017-2024

Subset of tiles / resume:
  uv run python scripts/data/tile_dataset.py --years 2023 --tiles T35SNS,T35SLT
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

from nilevit.select import DEFAULT_KEEP_PROB

with contextlib.suppress(ImportError):
    import nilevit  # noqa: F401

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


def parse_years(spec: str) -> list[int]:
    """'2023' -> [2023]; '2017-2024' -> [2017..2024]."""
    spec = spec.strip()
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-", 1))
        if lo > hi:
            raise ValueError("year range start must be <= end")
        return list(range(lo, hi + 1))
    return [int(spec)]


def build_command(
    py: str,
    script: Path,
    tile: str,
    year: int,
    labels_dir: Path,
    output_dir: Path,
    cloud_max: float,
    keep: dict[int, float],
) -> list[str]:
    """05b invocation for one (tile, year) with D7 selection."""
    return [
        py,
        str(script),
        "--tile",
        tile,
        "--year",
        str(year),
        "--source",
        "stac",
        "--all-dates",
        "--labels-dir",
        str(labels_dir),
        "--output-dir",
        str(output_dir),
        "--cloud-max",
        str(cloud_max),
        "--keep-none",
        str(keep[0]),
        "--keep-drought",
        str(keep[1]),
        "--keep-heat",
        str(keep[2]),
        "--keep-compound",
        str(keep[3]),
    ]


def _load_manifest(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    doc = json.loads(path.read_text())
    return {(t, int(y)) for t, y in doc.get("done", [])}


def _save_manifest(path: Path, done: set[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"done": sorted([t, y] for t, y in done)}, indent=2) + "\n",
        encoding="utf-8",
    )


@app.command()
def main(
    years: str = typer.Option("2023", "--years", help="Year or range, e.g. 2017-2024."),
    roi_tiles: Path = typer.Option(Path("configs/roi_tiles.json"), help="Tile coverage."),
    labels_root: Path = typer.Option(
        Path("data/interim"), help="Dir holding labels_<year>/ subdirs."
    ),
    output_dir: Path = typer.Option(Path("data/interim"), help="Tiling output dir."),
    tiles: str | None = typer.Option(None, "--tiles", help="Comma-separated subset."),
    cloud_max: float = typer.Option(50.0, "--cloud-max", help="Max eo:cloud_cover %."),
    keep_none: float = typer.Option(DEFAULT_KEEP_PROB[0], "--keep-none"),
    keep_drought: float = typer.Option(DEFAULT_KEEP_PROB[1], "--keep-drought"),
    keep_heat: float = typer.Option(DEFAULT_KEEP_PROB[2], "--keep-heat"),
    keep_compound: float = typer.Option(DEFAULT_KEEP_PROB[3], "--keep-compound"),
    manifest: Path = typer.Option(
        Path("data/interim/tiling_manifest.json"), help="Resume manifest."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan; tile nothing."),
    stop_on_fail: bool = typer.Option(False, "--stop-on-fail", help="Abort on first fail."),
) -> None:
    """Stream-tile every land tile x year with D7 selection, resumably."""
    try:
        year_list = parse_years(years)
    except ValueError as exc:
        console.print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=2) from exc

    cfg = json.loads(roi_tiles.read_text())
    land = cfg.get("land_roi_tiles")
    if not land:
        console.print("[red]ERROR[/red] roi_tiles.json has no 'land_roi_tiles'.")
        raise typer.Exit(code=1)
    if tiles:
        want = {t.strip() for t in tiles.split(",") if t.strip()}
        land = [t for t in land if t in want]
        if not land:
            console.print(f"[red]ERROR[/red] no land tiles matched {tiles!r}.")
            raise typer.Exit(code=1)

    script = Path(__file__).resolve().parent / "05b_tile.py"
    keep = {0: keep_none, 1: keep_drought, 2: keep_heat, 3: keep_compound}
    done = _load_manifest(manifest)

    # Build the work list, skipping years without labels and completed pairs.
    work: list[tuple[int, str, Path]] = []
    skipped_years: list[int] = []
    for year in year_list:
        labels_dir = labels_root / f"labels_{year}"
        if not labels_dir.exists():
            skipped_years.append(year)
            continue
        work.extend((year, tile, labels_dir) for tile in land if (tile, year) not in done)

    plan = Table(title="M4 stream-tiling plan", show_header=False, title_style="bold cyan")
    plan.add_column(style="dim")
    plan.add_column(style="bold")
    plan.add_row("Years", years)
    plan.add_row("Land tiles", str(len(land)))
    plan.add_row("Already done", str(len(done)))
    plan.add_row("To tile", str(len(work)))
    if skipped_years:
        plan.add_row("No labels (skipped)", ", ".join(str(y) for y in skipped_years))
    plan.add_row("keep n/d/h/c", f"{keep_none}/{keep_drought}/{keep_heat}/{keep_compound}")
    plan.add_row("Mode", "dry-run" if dry_run else "run")
    console.print(plan)
    console.print()

    if dry_run:
        for year, tile, _ in work[:20]:
            console.print(f"  [{year}] {tile}")
        if len(work) > 20:
            console.print(f"  ... and {len(work) - 20} more")
        console.print("\n[yellow]Dry run - nothing tiled.[/yellow]")
        raise typer.Exit(code=0)

    n_ok, n_fail = 0, 0
    for year, tile, labels_dir in work:
        console.rule(f"[bold]{year} {tile}")
        cmd = build_command(
            sys.executable, script, tile, year, labels_dir, output_dir, cloud_max, keep
        )
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            n_ok += 1
            done.add((tile, year))
            _save_manifest(manifest, done)
        else:
            n_fail += 1
            console.print(f"[red]FAILED[/red] {year} {tile} (exit {result.returncode})")
            if stop_on_fail:
                console.print("[red]--stop-on-fail set; aborting.[/red]")
                raise typer.Exit(code=1)

    console.print(f"\n[green]Tiling: {n_ok} ok, {n_fail} failed.[/green]")
    if n_fail:
        console.print("[yellow]Re-run to retry; completed pairs are skipped.[/yellow]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
