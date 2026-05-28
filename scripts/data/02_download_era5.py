"""Download ERA5-Land hourly meteorological data via Copernicus CDS.

The second Phase-2 data-pipeline script: pulls hourly weather variables over
the Nile Delta + Eastern Mediterranean ROI for a date range.

For the M2 milestone we just need to prove that:

  1. CDS authentication works from this environment (via ~/.cdsapirc).
  2. cdsapi can submit a request, queue, and return a NetCDF.
  3. The downloaded file opens cleanly with xarray and has the expected shape.

Once those are confirmed for one day/month, scaling to the full 2017-2024
window is just a loop over (year, month) - no new architecture.

Usage
-----
Smoke test (smallest possible request - one day, one variable):
  uv run python scripts/data/02_download_era5.py \
      --start 2023-08-01 --end 2023-08-01 --variables 2m_temperature --dry-run

Real smoke run (small but real):
  uv run python scripts/data/02_download_era5.py \
      --start 2023-08-01 --end 2023-08-01 --variables 2m_temperature

Full month, all default vars:
  uv run python scripts/data/02_download_era5.py --month 2023-08

Output layout
-------------
  <repo>/data/raw/era5/<year>/era5_land_<year>-<mm>_d<dd>-d<dd>.nc

References
----------
- ERA5-Land documentation:
  https://confluence.ecmwf.int/display/CKB/ERA5-Land
- CDS API docs:
  https://cds.climate.copernicus.eu/how-to-api
- Variable names (CDS short names) used here:
  t2m   - 2 m air temperature (K)
  swvl1 - volumetric soil water layer 1 (m^3/m^3, 0-7 cm)
  e     - total evaporation (m of water equivalent, negative = evap)
  tp    - total precipitation (m)
"""

# typer's standard pattern uses function calls in argument defaults; ruff B008
# is not relevant for CLI scripts.
# ruff: noqa: B008

from __future__ import annotations

import contextlib
from datetime import date, datetime, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

# Set PROJ paths on Windows before any geospatial import (no-op elsewhere).
with contextlib.suppress(ImportError):
    import nilevit  # noqa: F401

app = typer.Typer(
    add_completion=False,
    help="Download ERA5-Land hourly data via Copernicus CDS.",
)
console = Console()


# Default ROI for the Nile Delta + Eastern Mediterranean.
# CDS area order is (N, W, S, E) - unusual, easy to get wrong.
DEFAULT_AREA: tuple[float, float, float, float] = (37.0, 22.0, 30.0, 36.0)


# Default variables for compound-event detection.
# Names match the CDS variable catalogue for "reanalysis-era5-land".
DEFAULT_VARIABLES: list[str] = [
    "2m_temperature",
    "volumetric_soil_water_layer_1",
    "total_evaporation",
    "total_precipitation",
]


# Map CDS variable names to expected NetCDF shortnames for verification.
VAR_TO_SHORTNAME: dict[str, str] = {
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "volumetric_soil_water_layer_1": "swvl1",
    "volumetric_soil_water_layer_2": "swvl2",
    "soil_temperature_level_1": "stl1",
    "total_evaporation": "e",
    "total_precipitation": "tp",
    "surface_pressure": "sp",
    "skin_temperature": "skt",
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


def parse_area(area_str: str) -> tuple[float, float, float, float]:
    """'N,W,S,E' -> 4-tuple of floats (CDS area order)."""
    try:
        parts = [float(p.strip()) for p in area_str.split(",")]
    except ValueError as exc:
        raise ValueError(f"Invalid area {area_str!r}; expected 'N,W,S,E' (CDS order)") from exc
    if len(parts) != 4:
        raise ValueError(f"Invalid area {area_str!r}; expected 4 comma-separated floats")
    n, w, s, e = parts
    if n <= s:
        raise ValueError(f"N ({n}) must be greater than S ({s})")
    if e <= w:
        raise ValueError(f"E ({e}) must be greater than W ({w})")
    if not (-90 <= s <= 90 and -90 <= n <= 90):
        raise ValueError("Latitudes must be in [-90, 90]")
    if not (-180 <= w <= 180 and -180 <= e <= 180):
        raise ValueError("Longitudes must be in [-180, 180]")
    return n, w, s, e


def daterange(start: date, end: date) -> list[date]:
    """Inclusive list of dates from start to end."""
    out = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def days_by_yearmonth(dates: list[date]) -> dict[tuple[int, int], list[int]]:
    """Group day-of-month numbers by (year, month) for CDS chunking."""
    out: dict[tuple[int, int], list[int]] = {}
    for d in dates:
        out.setdefault((d.year, d.month), []).append(d.day)
    return out


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


@app.command()
def main(
    month: str | None = typer.Option(
        None,
        "--month",
        "-m",
        help="Year-month YYYY-MM (e.g. 2023-08). Overrides --start/--end.",
    ),
    start: str | None = typer.Option(None, "--start", help="Start date YYYY-MM-DD."),
    end: str | None = typer.Option(None, "--end", help="End date YYYY-MM-DD (inclusive)."),
    area: str | None = typer.Option(
        None,
        "--area",
        "-a",
        help=(
            'Area "N,W,S,E" in lat-lon degrees. CDS order! '
            f"Default: {','.join(str(v) for v in DEFAULT_AREA)} (Nile + East Med)."
        ),
    ),
    variables: str = typer.Option(
        ",".join(DEFAULT_VARIABLES),
        "--variables",
        "-v",
        help="Comma-separated list of ERA5-Land variable names.",
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
        help="Build the CDS request and print it, but don't submit.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Re-download even if the output file already exists.",
    ),
) -> None:
    """Submit a CDS request for ERA5-Land hourly data."""
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

    # ERA5-Land has a 2-3 month latency vs realtime.
    today = date.today()
    if end_date >= today - timedelta(days=60):
        console.print(
            "[yellow]Warning: ERA5-Land has ~2-3 month latency; "
            "very recent dates may not be available.[/yellow]"
        )

    # ---- Validate area ----
    if area is None:
        n, w, s, e = DEFAULT_AREA
    else:
        try:
            n, w, s, e = parse_area(area)
        except ValueError as exc:
            console.print(f"[red]ERROR[/red] {exc}")
            raise typer.Exit(code=2) from exc

    # ---- Variables ----
    var_list = [v.strip() for v in variables.split(",") if v.strip()]
    if not var_list:
        console.print("[red]ERROR[/red] no variables specified.")
        raise typer.Exit(code=2)

    # ---- Output dir ----
    if output_dir is None:
        output_dir = default_data_root() / "raw" / "era5" / str(start_date.year)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Chunk by month ----
    all_dates = daterange(start_date, end_date)
    months = days_by_yearmonth(all_dates)

    # ---- Print plan ----
    plan = Table(title="ERA5-Land download plan", show_header=False, title_style="bold cyan")
    plan.add_column(style="dim")
    plan.add_column(style="bold")
    plan.add_row("Date range", f"{start_date} -> {end_date} ({len(all_dates)} day(s))")
    plan.add_row("Area (N,W,S,E)", f"{n}, {w}, {s}, {e}")
    plan.add_row("Variables", ", ".join(var_list))
    plan.add_row("Requests (per-month)", str(len(months)))
    plan.add_row("Output dir", str(output_dir))
    plan.add_row("Runtime", "Colab" if is_colab() else "Local")
    plan.add_row("Dry run", "yes" if dry_run else "no")
    if overwrite:
        plan.add_row("Overwrite", "yes")
    console.print(plan)
    console.print()

    # ---- Lazy imports ----
    try:
        import cdsapi
    except ImportError:
        console.print(
            "[red]ERROR[/red] cdsapi not installed. Run `uv sync` " "(or `pip install cdsapi`)."
        )
        raise typer.Exit(code=1) from None

    # ---- Build requests (one per month) ----
    times = [f"{h:02d}:00" for h in range(24)]
    requests: list[tuple[dict, Path]] = []
    for (yr, mn), days in sorted(months.items()):
        out_name = f"era5_land_{yr}-{mn:02d}_d{days[0]:02d}-d{days[-1]:02d}.nc"
        out_path = output_dir / out_name
        req = {
            "variable": var_list,
            "year": str(yr),
            "month": f"{mn:02d}",
            "day": [f"{d:02d}" for d in days],
            "time": times,
            "area": [n, w, s, e],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        requests.append((req, out_path))

    # ---- Show requests ----
    rtable = Table(title="Requests", title_style="bold")
    rtable.add_column("#", style="dim", justify="right")
    rtable.add_column("Year-Mo", style="cyan")
    rtable.add_column("Days", style="green", justify="right")
    rtable.add_column("Output file", style="white", overflow="fold")
    rtable.add_column("Exists?", style="yellow", justify="center")
    for i, (req, out_path) in enumerate(requests, 1):
        rtable.add_row(
            str(i),
            f"{req['year']}-{req['month']}",
            str(len(req["day"])),
            out_path.name,
            "yes" if out_path.exists() else "no",
        )
    console.print(rtable)
    console.print()

    if dry_run:
        console.print("[bold yellow]Dry run - not submitting.[/bold yellow]")
        raise typer.Exit(code=0)

    # ---- Submit requests ----
    client = cdsapi.Client()
    console.print(f"[green]CDS client ready: {client.url}[/green]\n")

    for i, (req, out_path) in enumerate(requests, 1):
        if out_path.exists() and not overwrite:
            console.print(
                f"[yellow]Request {i}/{len(requests)}: "
                f"{out_path.name} already exists, skipping. "
                "Pass --overwrite to force.[/yellow]"
            )
            continue

        console.print(f"[bold]Request {i}/{len(requests)}: submitting -> {out_path.name}[/bold]")
        console.print(
            "  Queue times vary from ~1 min to several hours; " "cdsapi will block until ready."
        )
        try:
            client.retrieve("reanalysis-era5-land", req, str(out_path))
        except Exception as exc:
            console.print(f"[red]Request {i} failed: {exc}[/red]")
            console.print(
                "[yellow]If this is a license error, accept the dataset "
                "license at https://cds.climate.copernicus.eu/datasets/"
                "reanalysis-era5-land?tab=download[/yellow]"
            )
            raise typer.Exit(code=1) from exc

        size_mb = out_path.stat().st_size / 1024 / 1024
        console.print(f"[green]Saved {out_path} ({size_mb:.1f} MB)[/green]\n")

    # ---- Verify ----
    console.print("[bold]Verifying downloaded NetCDF files...[/bold]")
    try:
        import xarray as xr
    except ImportError:
        console.print("[yellow]xarray not installed; skipping verification.[/yellow]")
        return

    for _, out_path in requests:
        if not out_path.exists():
            continue
        try:
            ds = xr.open_dataset(out_path)
        except Exception as exc:
            console.print(
                f"[red]Could not open {out_path.name}: {exc}[/red]\n"
                "[yellow]You may need a NetCDF engine: "
                "`uv add netCDF4` or `uv add h5netcdf`.[/yellow]"
            )
            continue
        console.print(f"\n[cyan]{out_path.name}[/cyan]")
        console.print(f"  Dimensions: {dict(ds.sizes)}")
        console.print(f"  Variables:  {list(ds.data_vars)}")
        # CDS-Beta renamed `time` to `valid_time`; handle both for compat.
        time_coord = (
            "valid_time" if "valid_time" in ds.coords else "time" if "time" in ds.coords else None
        )
        if time_coord:
            t = ds[time_coord]
            console.print(f"  {time_coord} range: {t.min().values} -> {t.max().values}")
        if "latitude" in ds.coords and "longitude" in ds.coords:
            console.print(
                f"  Lat range:  {float(ds.latitude.min()):.2f} -> "
                f"{float(ds.latitude.max()):.2f}"
            )
            console.print(
                f"  Lon range:  {float(ds.longitude.min()):.2f} -> "
                f"{float(ds.longitude.max()):.2f}"
            )
        ds.close()


if __name__ == "__main__":
    app()
