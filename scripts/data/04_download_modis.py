"""Download MODIS Terra products (MOD13Q1 NDVI, MOD11A2 LST) from Planetary Computer.

MODIS feeds the Vegetation Health Index (VHI) used in compound-event labels:

    VHI = 0.5 * VCI + 0.5 * TCI
    VCI from MOD13Q1 NDVI (16-day, 250 m)
    TCI from MOD11A2 LST  (8-day, 1 km)

Why Planetary Computer instead of NASA Earthdata?
-------------------------------------------------
NASA distributes MODIS as HDF4-EOS, which is unreadable on a pip/uv Windows
stack: the GDAL inside the rasterio wheel has no HDF4 driver, and pyhdf needs
HDF4 runtime DLLs that no wheel ships. Microsoft Planetary Computer (MPC)
re-publishes every MODIS subdataset as a Cloud-Optimized GeoTIFF (COG), readable
directly with rioxarray - the same stack as our Sentinel-2 fallback. As a bonus
we clip to the ROI on read, so files land small and analysis-ready.

Collections
-----------
  MOD13Q1 -> modis-13Q1-061  (Vegetation Indices 16-day, 250 m)  asset ~ "*NDVI*"
  MOD11A2 -> modis-11A2-061  (LST/Emissivity 8-day, 1 km)        asset ~ "LST_Day*"

We keep only Terra granules (item id starts with "MOD"); MPC mixes Terra (MOD)
and Aqua (MYD) in these collections and the PRD specifies Terra.

Usage
-----
Smoke-test (list items, download nothing):
  uv run python scripts/data/04_download_modis.py \
      --product MOD13Q1 --month 2023-08 --dry-run

Download + ROI-clip one item:
  uv run python scripts/data/04_download_modis.py \
      --product MOD13Q1 --month 2023-08 --max-granules 1

Full month:
  uv run python scripts/data/04_download_modis.py --product MOD11A2 --month 2023-08

Output layout
-------------
  <repo>/data/raw/modis/<product>/<year>/<item_id>.<band>.nile-em.tif
"""

# typer's standard pattern uses function calls in argument defaults.
# ruff: noqa: B008

from __future__ import annotations

import contextlib
from datetime import date, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

with contextlib.suppress(ImportError):
    import nilevit  # noqa: F401

app = typer.Typer(
    add_completion=False,
    help="Download MODIS Terra products (MOD13Q1, MOD11A2) from Planetary Computer.",
)
console = Console()


MPC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Project ROI (W, S, E, N) in degrees - same as HLS / CHIRPS / ERA5 defaults.
DEFAULT_BBOX: tuple[float, float, float, float] = (22.0, 30.0, 36.0, 37.0)
DEFAULT_ROI_TAG = "nile-em"

# Map our PRD product short-names to MPC collections + the band we extract.
SUPPORTED_PRODUCTS: dict[str, dict[str, str]] = {
    "MOD13Q1": {
        "collection": "modis-13Q1-061",
        "long_name": "Terra Vegetation Indices",
        "cadence": "16-day composite",
        "resolution": "250 m",
        "use": "NDVI for VCI / VHI",
        "band_pattern": "NDVI",  # asset key contains this (case-insensitive)
        "expected_asset": "250m_16_days_NDVI",
    },
    "MOD11A2": {
        "collection": "modis-11A2-061",
        "long_name": "Terra Land Surface Temperature",
        "cadence": "8-day composite",
        "resolution": "1 km",
        "use": "LST for TCI / VHI",
        "band_pattern": "LST_DAY",
        "expected_asset": "LST_Day_1km",
    },
}


def is_colab() -> bool:
    """Return True if running inside a Google Colab kernel."""
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def default_data_root() -> Path:
    """Repo data/ dir locally; /content/data on Colab."""
    if is_colab():
        return Path("/content/data")
    return Path(__file__).resolve().parents[2] / "data"


def parse_date(s: str) -> date:
    """'YYYY-MM-DD' -> date. Raises ValueError on bad input."""
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()


def year_range(year: int) -> tuple[date, date]:
    """(Jan 1, Dec 31) for `year`. Raises ValueError if year is implausible."""
    if year < 2000 or year > date.today().year + 1:
        raise ValueError(f"Year {year} outside plausible MODIS range")
    return date(year, 1, 1), date(year, 12, 31)


def month_range(year_month: str) -> tuple[date, date]:
    """'YYYY-MM' -> (first day, last day) of that month."""
    parts = year_month.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid year-month {year_month!r}; expected YYYY-MM")
    y, m = int(parts[0]), int(parts[1])
    if not (1 <= m <= 12):
        raise ValueError(f"Month must be 1-12, got {m}")
    first = date(y, m, 1)
    next_first = date(y + (m // 12), (m % 12) + 1, 1)
    last = date.fromordinal(next_first.toordinal() - 1)
    return first, last


def is_terra(item_id: str) -> bool:
    """True for Terra granules (MOD*), False for Aqua (MYD*) or anything else."""
    return item_id.upper().startswith("MOD")


def pick_band_asset(asset_keys: list[str], pattern: str) -> str | None:
    """Choose the COG asset key matching `pattern` (case-insensitive substring).

    Prefers an exact case-insensitive match, else the first substring hit.
    """
    pat = pattern.upper()
    upper = {k.upper(): k for k in asset_keys}
    if pat in upper:
        return upper[pat]
    for up, original in upper.items():
        if pat in up:
            return original
    return None


@app.command()
def main(
    product: str = typer.Option(
        ...,
        "--product",
        "-p",
        help="MODIS product short name: MOD13Q1 (NDVI) or MOD11A2 (LST).",
    ),
    year: int | None = typer.Option(
        None, "--year", "-y", help="Calendar year, e.g. 2023 (Jan 1 - Dec 31)."
    ),
    month: str | None = typer.Option(
        None, "--month", "-m", help="Single year-month YYYY-MM. Overrides start/end."
    ),
    start: str | None = typer.Option(None, "--start", help="Start date YYYY-MM-DD (inclusive)."),
    end: str | None = typer.Option(None, "--end", help="End date YYYY-MM-DD (inclusive)."),
    bbox: tuple[float, float, float, float] = typer.Option(
        DEFAULT_BBOX,
        "--bbox",
        "-b",
        help="Bounding box: W S E N in degrees (default: Nile + E. Mediterranean).",
    ),
    roi_tag: str = typer.Option(
        DEFAULT_ROI_TAG, "--roi-tag", help="Tag inserted into output filenames."
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", "-o", help="Override default download directory."
    ),
    max_granules: int | None = typer.Option(
        None, "--max-granules", help="Cap items processed (for smoke testing)."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="List items and exit; download nothing."),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Re-clip even if the output file exists."
    ),
) -> None:
    """Download + ROI-clip MODIS Terra bands (MOD13Q1 NDVI / MOD11A2 LST) from MPC."""
    # ---- Validate product ----
    if product not in SUPPORTED_PRODUCTS:
        console.print(
            f"[red]ERROR[/red] --product must be one of {list(SUPPORTED_PRODUCTS)}, "
            f"got {product!r}."
        )
        raise typer.Exit(code=2)
    meta = SUPPORTED_PRODUCTS[product]

    # ---- Resolve date range ----
    try:
        if month is not None:
            start_date, end_date = month_range(month)
        elif year is not None:
            start_date, end_date = year_range(year)
        elif start is not None and end is not None:
            start_date = parse_date(start)
            end_date = parse_date(end)
            if start_date > end_date:
                raise ValueError("--start must be <= --end")
        else:
            console.print(
                "[red]ERROR[/red] supply one of: --year, --month, " "OR both --start and --end."
            )
            raise typer.Exit(code=2)
    except ValueError as exc:
        console.print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=2) from exc

    # ---- Validate bbox ----
    w, s, e, n = bbox
    if not (w < e and s < n):
        console.print(f"[red]ERROR[/red] invalid bbox W={w} S={s} E={e} N={n}.")
        raise typer.Exit(code=2)

    # ---- Output dir ----
    if output_dir is None:
        output_dir = default_data_root() / "raw" / "modis" / product / str(start_date.year)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Plan ----
    plan = Table(title="MODIS download plan", show_header=False, title_style="bold cyan")
    plan.add_column(style="dim")
    plan.add_column(style="bold")
    plan.add_row("Product", f"{product} - {meta['long_name']}")
    plan.add_row("MPC collection", meta["collection"])
    plan.add_row("Band", meta["expected_asset"])
    plan.add_row("Cadence / res", f"{meta['cadence']}, {meta['resolution']}")
    plan.add_row("Date range", f"{start_date} -> {end_date}")
    plan.add_row("Bbox (W S E N)", f"{w}, {s}, {e}, {n}")
    plan.add_row("Output dir", str(output_dir))
    plan.add_row("Max granules", str(max_granules) if max_granules else "unlimited")
    plan.add_row("Runtime", "Colab" if is_colab() else "Local")
    plan.add_row("Dry run", "yes" if dry_run else "no")
    console.print(plan)
    console.print()

    # ---- Lazy imports ----
    try:
        import planetary_computer
        import pystac_client
    except ImportError:
        console.print(
            "[red]ERROR[/red] need pystac-client + planetary-computer. "
            "Run `uv add pystac-client planetary-computer`."
        )
        raise typer.Exit(code=1) from None

    # ---- Search ----
    console.print(f"[bold]Searching MPC STAC: {meta['collection']}...[/bold]")
    try:
        catalog = pystac_client.Client.open(MPC_STAC_URL, modifier=planetary_computer.sign_inplace)
        search = catalog.search(
            collections=[meta["collection"]],
            bbox=(w, s, e, n),
            datetime=f"{start_date.isoformat()}/{end_date.isoformat()}",
        )
        items = list(search.items())
    except Exception as exc:
        console.print(f"[red]ERROR[/red] STAC search failed: {exc}")
        raise typer.Exit(code=1) from exc

    # Terra-only filter.
    terra = [it for it in items if is_terra(it.id)]
    console.print(f"[green]Found {len(items)} item(s); {len(terra)} are Terra (MOD*).[/green]")
    if not terra:
        console.print("[yellow]No Terra items for this query.[/yellow]")
        raise typer.Exit(code=0)

    if max_granules and len(terra) > max_granules:
        console.print(f"[yellow]Capping {len(terra)} -> {max_granules}.[/yellow]")
        terra = terra[:max_granules]

    # ---- Item summary (first 5) ----
    summary = Table(title="First Terra items", title_style="bold cyan")
    summary.add_column("#", style="dim")
    summary.add_column("Item id", overflow="fold")
    for i, it in enumerate(terra[:5]):
        summary.add_row(str(i + 1), it.id)
    if len(terra) > 5:
        summary.add_row("...", f"+ {len(terra) - 5} more")
    console.print(summary)
    console.print()

    if dry_run:
        # Show the available assets on the first item so band keys are visible.
        first = terra[0]
        keys = list(first.assets.keys())
        chosen = pick_band_asset(keys, meta["band_pattern"])
        console.print(f"[bold]Assets on {first.id}:[/bold]")
        console.print(f"  {', '.join(keys)}")
        console.print(f"[bold]Band asset selected:[/bold] {chosen or '[red]NONE[/red]'}")
        console.print("\n[yellow]Dry run - not downloading.[/yellow]")
        raise typer.Exit(code=0)

    # ---- Download + clip ----
    try:
        import rioxarray
    except ImportError:
        console.print("[red]ERROR[/red] rioxarray not installed. Run `uv add rioxarray`.")
        raise typer.Exit(code=1) from None
    import rioxarray

    console.print(f"[bold]Clipping {len(terra)} item(s) to ROI...[/bold]")
    from rioxarray.exceptions import NoDataInBounds

    n_done, n_skip, n_outside, n_fail = 0, 0, 0, 0
    for it in terra:
        band_key = pick_band_asset(list(it.assets.keys()), meta["band_pattern"])
        if band_key is None:
            console.print(f"  [red]FAIL[/red] {it.id}: no band matching {meta['band_pattern']!r}")
            n_fail += 1
            continue

        out_path = output_dir / f"{it.id}.{band_key}.{roi_tag}.tif"
        if out_path.exists() and not overwrite:
            console.print(f"  [yellow]SKIP[/yellow] {out_path.name} (exists)")
            n_skip += 1
            continue

        href = it.assets[band_key].href  # already signed by the modifier
        try:
            da = rioxarray.open_rasterio(href, masked=True)
        except Exception as exc:
            console.print(f"  [red]FAIL[/red] {it.id}: open failed: {exc}")
            n_fail += 1
            continue

        try:
            # clip_box reprojects the lat/lon box into the raster's sinusoidal CRS.
            clipped = da.rio.clip_box(minx=w, miny=s, maxx=e, maxy=n, crs="EPSG:4326")
        except NoDataInBounds:
            # MODIS tile (e.g. a v06 tile) does not overlap the ROI - expected.
            da.close()
            console.print(f"  [dim]outside ROI[/dim] {it.id}")
            n_outside += 1
            continue
        except Exception as exc:
            da.close()
            console.print(f"  [red]FAIL[/red] {it.id}: clip failed: {exc}")
            n_fail += 1
            continue

        clipped.rio.to_raster(out_path, compress="deflate")
        da.close()
        mb = out_path.stat().st_size / 1e6
        console.print(f"  [green]OK[/green] {out_path.name} ({mb:.2f} MB)")
        n_done += 1

    console.print()
    console.print(
        f"[green]Done: {n_done} clipped, {n_skip} skipped, "
        f"{n_outside} outside ROI, {n_fail} failed.[/green]"
    )
    console.print(f"[green]Output: {output_dir}[/green]")
    if n_fail:
        raise typer.Exit(code=1)

    # ---- Verify one file ----
    console.print("\n[bold]Verifying one clipped GeoTIFF...[/bold]")
    tifs = sorted(output_dir.glob("*.tif"))
    if not tifs:
        console.print("[yellow]Nothing to verify.[/yellow]")
        return
    sample = tifs[0]
    try:
        import numpy as np

        da = rioxarray.open_rasterio(sample, masked=True)
        console.print(f"\n[cyan]{sample.name}[/cyan]")
        console.print(f"  Shape   : {tuple(da.shape)}")
        console.print(f"  CRS     : {da.rio.crs}")
        console.print(f"  Bounds  : {tuple(round(b, 1) for b in da.rio.bounds())}")
        console.print(f"  Pixel   : {tuple(round(r, 1) for r in da.rio.resolution())}")
        vals = da.values[np.isfinite(da.values)]
        if vals.size:
            console.print(
                f"  Data    : min {float(vals.min()):.1f}, "
                f"max {float(vals.max()):.1f}, mean {float(vals.mean()):.1f} (scaled)"
            )
        da.close()
        console.print("\n  [green]COG read OK.[/green]")
    except Exception as exc:
        console.print(f"[yellow]Could not verify: {exc}[/yellow]")


if __name__ == "__main__":
    app()
