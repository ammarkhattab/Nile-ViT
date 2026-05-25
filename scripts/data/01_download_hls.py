"""Download HLS (Harmonized Landsat Sentinel-2) tiles via NASA earthaccess.

The first Phase-2 data-pipeline script: pulls one MGRS tile, one month at a time.
For the M2 milestone we just need to prove that:

  1. NASA Earthdata authentication works from this environment.
  2. earthaccess finds HLS granules over our Nile-Delta ROI.
  3. Files land on disk in the directory layout the rest of the pipeline expects.

Once those are confirmed for one tile/month, scaling up to the full ROI x 8 years
is just a loop over (tile, month) pairs - no new architecture needed.

Usage
-----
  uv run python scripts/data/01_download_hls.py --tile T36RUU --month 2023-08
  uv run python scripts/data/01_download_hls.py --tile T36RUU --month 2023-08 --dry-run
  uv run python scripts/data/01_download_hls.py \
      --bbox "29.5,30.0,32.5,31.5" --start 2023-08-01 --end 2023-08-31

Output layout
-------------
  <repo>/data/raw/hls/<tile>/<year>/<granule files>

References
----------
- HLS product description: https://lpdaac.usgs.gov/products/hlss30v002/
- earthaccess docs:        https://earthaccess.readthedocs.io/
- HLS MGRS tile grid:      https://hls.gsfc.nasa.gov/products-description/tiling-system/
"""

# typer's standard pattern uses function calls (typer.Option(...)) in argument
# defaults, which ruff's B008 flags. That's a typer-specific style, not a bug.
# ruff: noqa: B008

from __future__ import annotations

import contextlib
from datetime import date, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

# Set PROJ paths on Windows before any geospatial import (no-op elsewhere).
# nilevit/_environment.py runs as a side-effect of the import.
with contextlib.suppress(ImportError):
    import nilevit  # noqa: F401

app = typer.Typer(
    add_completion=False,
    help="Download HLS tiles via NASA earthaccess for a region and date range.",
)
console = Console()


# MGRS tiles covering the Nile Delta + Eastern Mediterranean ROI.
# Bounds are approximate (lon_min, lat_min, lon_max, lat_max).
# Source: https://hls.gsfc.nasa.gov/products-description/tiling-system/
KNOWN_TILES: dict[str, tuple[float, float, float, float]] = {
    "T36RUU": (31.0, 30.0, 32.2, 31.0),  # Central Delta - Cairo, Tanta
    "T36RUV": (31.0, 31.0, 32.2, 32.0),  # North-central - Damietta
    "T35RPN": (29.0, 30.0, 30.2, 31.0),  # Western Delta - Damanhur
    "T36RTU": (30.0, 30.0, 31.2, 31.0),  # Western-central Delta
    "T36RTV": (30.0, 31.0, 31.2, 32.0),  # NW Delta - Alexandria approach
}


def is_colab() -> bool:
    """Return True if running inside a Google Colab kernel."""
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def default_data_root() -> Path:
    """Repo's data/ dir locally; /content/data on Colab."""
    if is_colab():
        return Path("/content/data")
    return Path(__file__).resolve().parents[2] / "data"


def parse_month(month_str: str) -> tuple[date, date]:
    """'2023-08' -> (date(2023,8,1), date(2023,8,31))."""
    try:
        year_s, month_s = month_str.split("-")
        year, month = int(year_s), int(month_s)
        first = date(year, month, 1)
        next_first = date(year + (month // 12), (month % 12) + 1, 1)
        last = date.fromordinal(next_first.toordinal() - 1)
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid month {month_str!r}; expected YYYY-MM") from exc
    return first, last


def parse_bbox(bbox_str: str) -> tuple[float, float, float, float]:
    """'lon_min,lat_min,lon_max,lat_max' -> 4-tuple of floats."""
    try:
        parts = [float(p.strip()) for p in bbox_str.split(",")]
    except ValueError as exc:
        raise ValueError(
            f"Invalid bbox {bbox_str!r}; expected 'lon_min,lat_min,lon_max,lat_max'"
        ) from exc
    if len(parts) != 4:
        raise ValueError(f"Invalid bbox {bbox_str!r}; expected 4 comma-separated floats")
    lon_min, lat_min, lon_max, lat_max = parts
    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError(f"Invalid bbox {bbox_str!r}; need lon_min < lon_max and lat_min < lat_max")
    return lon_min, lat_min, lon_max, lat_max


@app.command()
def main(
    tile: str | None = typer.Option(
        None,
        "--tile",
        "-t",
        help="MGRS tile id (e.g. T36RUU). Mutually exclusive with --bbox.",
    ),
    bbox: str | None = typer.Option(
        None,
        "--bbox",
        "-b",
        help='Bounding box "lon_min,lat_min,lon_max,lat_max". Mutually exclusive with --tile.',
    ),
    month: str | None = typer.Option(
        None,
        "--month",
        "-m",
        help="Year-month YYYY-MM (e.g. 2023-08). Overrides --start/--end.",
    ),
    start: str | None = typer.Option(None, "--start", help="Start date YYYY-MM-DD."),
    end: str | None = typer.Option(None, "--end", help="End date YYYY-MM-DD (inclusive)."),
    cloud_max: float = typer.Option(
        50.0,
        "--cloud-max",
        help="Max cloud cover percent [0..100] (server-side filter).",
    ),
    product: str = typer.Option(
        "HLSS30",
        "--product",
        help="HLS short_name. HLSS30 = Sentinel-2-derived, HLSL30 = Landsat-derived.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Override default download directory.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Search and summarise but do not download.",
    ),
    max_granules: int | None = typer.Option(
        None,
        "--max-granules",
        help="Cap downloads for quick smoke tests.",
    ),
) -> None:
    """Search and download HLS granules."""
    # ---- Validate region inputs ----
    if (tile is None) == (bbox is None):
        console.print("[red]ERROR[/red] supply exactly one of --tile or --bbox.")
        raise typer.Exit(code=2)

    if tile is not None:
        if tile not in KNOWN_TILES:
            console.print(
                f"[red]ERROR[/red] tile {tile!r} not in known set: "
                f"{sorted(KNOWN_TILES)}. Use --bbox for ad-hoc regions."
            )
            raise typer.Exit(code=2)
        bbox_tuple = KNOWN_TILES[tile]
        region_label = tile
    else:
        try:
            bbox_tuple = parse_bbox(bbox)  # type: ignore[arg-type]
        except ValueError as exc:
            console.print(f"[red]ERROR[/red] {exc}")
            raise typer.Exit(code=2) from exc
        region_label = f"bbox{bbox_tuple}"

    # ---- Validate date inputs ----
    if month is not None:
        try:
            start_date, end_date = parse_month(month)
        except ValueError as exc:
            console.print(f"[red]ERROR[/red] {exc}")
            raise typer.Exit(code=2) from exc
    elif start is not None and end is not None:
        try:
            start_date = datetime.strptime(start, "%Y-%m-%d").date()
            end_date = datetime.strptime(end, "%Y-%m-%d").date()
        except ValueError as exc:
            console.print(f"[red]ERROR[/red] dates must be YYYY-MM-DD: {exc}")
            raise typer.Exit(code=2) from exc
        if start_date > end_date:
            console.print("[red]ERROR[/red] --start must be <= --end.")
            raise typer.Exit(code=2)
    else:
        console.print("[red]ERROR[/red] supply --month OR both --start and --end.")
        raise typer.Exit(code=2)

    # ---- Resolve output directory ----
    if output_dir is None:
        output_dir = default_data_root() / "raw" / "hls" / region_label / str(start_date.year)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Print plan ----
    plan = Table(title="HLS download plan", show_header=False, title_style="bold cyan")
    plan.add_column(style="dim")
    plan.add_column(style="bold")
    plan.add_row("Region", f"{region_label}  bbox={bbox_tuple}")
    plan.add_row("Date range", f"{start_date} -> {end_date}")
    plan.add_row("Product", product)
    plan.add_row("Max cloud cover", f"{cloud_max:.0f}%")
    plan.add_row("Output dir", str(output_dir))
    plan.add_row("Runtime", "Colab" if is_colab() else "Local")
    plan.add_row("Dry run", "yes" if dry_run else "no")
    if max_granules is not None:
        plan.add_row("Max granules", str(max_granules))
    console.print(plan)
    console.print()

    # ---- Lazy imports so --help works without these installed ----
    try:
        import earthaccess
    except ImportError:
        console.print(
            "[red]ERROR[/red] earthaccess not installed. "
            "Run `uv sync` (or `pip install earthaccess`)."
        )
        raise typer.Exit(code=1) from None

    # ---- Authenticate ----
    console.print("[bold]Authenticating with NASA Earthdata...[/bold]")
    auth = earthaccess.login(strategy="netrc")
    if not auth.authenticated:
        console.print(
            "[yellow]No ~/.netrc credentials found. "
            "Falling back to interactive login (will persist for next time).[/yellow]"
        )
        auth = earthaccess.login(strategy="interactive", persist=True)
    if not auth.authenticated:
        console.print("[red]ERROR[/red] Earthdata authentication failed.")
        raise typer.Exit(code=1)
    console.print("[green]Authenticated.[/green]\n")

    # ---- Search ----
    console.print(f"[bold]Searching {product} via CMR...[/bold]")
    granules = earthaccess.search_data(
        short_name=product,
        bounding_box=bbox_tuple,
        temporal=(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
        cloud_cover=(0, int(cloud_max)),
    )
    console.print(f"[green]{len(granules)} granule(s) found.[/green]")

    # CMR's bbox filter returns all tiles INTERSECTING the box - when --tile
    # is specified we want exactly that MGRS tile, so post-filter by GranuleUR.
    if tile is not None and granules:
        before = len(granules)
        granules = [g for g in granules if f".{tile}." in g.get("umm", {}).get("GranuleUR", "")]
        if len(granules) != before:
            console.print(f"  Filtered to {tile} only: {before} -> {len(granules)} granule(s).")
    console.print()

    if not granules:
        console.print(
            "[yellow]No granules matched. Try widening --cloud-max or the date range.[/yellow]"
        )
        raise typer.Exit(code=0)

    if max_granules is not None:
        granules = granules[:max_granules]
        console.print(f"  Capping to first {len(granules)} granule(s).\n")

    # ---- Summary table of what we found ----
    summary = Table(title=f"Granules to fetch ({len(granules)})", title_style="bold")
    summary.add_column("#", style="dim", justify="right")
    summary.add_column("Acquisition date", style="cyan")
    summary.add_column("Granule UR", style="green", overflow="fold")

    for i, g in enumerate(granules[:30], start=1):
        umm = g.get("umm", {}) if hasattr(g, "get") else getattr(g, "_data", {}).get("umm", {})
        granule_ur = umm.get("GranuleUR", "?")
        try:
            begin_dt = umm["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
            acq_date = begin_dt[:10]
        except (KeyError, TypeError):
            acq_date = "?"
        summary.add_row(str(i), acq_date, granule_ur)
    if len(granules) > 30:
        summary.add_row("...", "...", f"({len(granules) - 30} more)")
    console.print(summary)
    console.print()

    if dry_run:
        console.print("[bold yellow]Dry run - exiting before download.[/bold yellow]")
        raise typer.Exit(code=0)

    # ---- Download ----
    console.print(f"[bold]Downloading to {output_dir}...[/bold]")
    files = earthaccess.download(granules, str(output_dir))

    # ---- Report ----
    downloaded = [Path(f) for f in (files or []) if f and Path(f).exists()]
    total_mb = sum(p.stat().st_size for p in downloaded) / (1024**2)
    console.print()
    console.print(f"[green]Downloaded {len(downloaded)} file(s), {total_mb:,.1f} MB total.[/green]")
    console.print(f"[green]Output: {output_dir}[/green]")


if __name__ == "__main__":
    app()
