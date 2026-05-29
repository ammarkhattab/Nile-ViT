"""Download CHIRPS v3 daily precipitation as monthly NetCDF, ROI-subset.

CHIRPS v3 daily data is distributed as **monthly NetCDF files** (one per
year-month, ~300-400 MB each, GLOBAL extent at 0.05 deg). This script:

  1. Downloads the monthly file from the UCSB CHC data server (HTTPS, no auth).
  2. Subsets it to our ROI using xarray (defaults to the Nile-EM box).
  3. Saves the small ROI slice (~700 KB / month).
  4. Optionally deletes the giant global file (default: delete).

Why this design
---------------
- Storage: 96 months at 350 MB each is ~33 GB - way over our 3 GB CHIRPS budget.
- Our ROI is ~1/440 of the global grid, so subset files are ~1 MB instead.
- We still get full daily-resolution precip over the region we care about.

The file naming follows the CHC convention exactly:
  chirps-v3.0.YYYY.MM.days_p05.nc      (global, ~350 MB)

and our derived per-ROI file:
  chirps-v3.0.YYYY.MM.days_p05.nile-em.nc   (subset, ~1 MB)

Confirmed endpoint (browsed Apache autoindex 2026-05-28):
  https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/rnl/netcdf/byMonth/

Flavors
-------
  rnl   ERA5-based daily disaggregation (matches our ERA5 pipeline; DEFAULT)
  sat   IMERG-based daily disaggregation

Usage
-----
Probe (verify the file exists and you can reach the server):
  uv run python scripts/data/03_download_chirps.py --month 2023-08 --probe

Download August 2023, default ROI, subset, delete global:
  uv run python scripts/data/03_download_chirps.py --month 2023-08

Range of months, keep the global file too:
  uv run python scripts/data/03_download_chirps.py \
      --start 2017-01 --end 2024-12 --keep-global

Custom area (N W S E in degrees):
  uv run python scripts/data/03_download_chirps.py --month 2023-08 \
      --area 37 22 30 36
"""

# typer's standard pattern uses function calls in argument defaults.
# ruff: noqa: B008

from __future__ import annotations

import contextlib
from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

with contextlib.suppress(ImportError):
    import nilevit  # noqa: F401

app = typer.Typer(
    add_completion=False,
    help="Download CHIRPS v3 daily precipitation (monthly NetCDF) from UCSB CHC.",
)
console = Console()


CHC_BASE = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final"

# (north, west, south, east) - matches ERA5 conventions.
DEFAULT_AREA: tuple[float, float, float, float] = (37.0, 22.0, 30.0, 36.0)
DEFAULT_ROI_TAG = "nile-em"


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


def parse_year_month(s: str) -> tuple[int, int]:
    """'2023-08' -> (2023, 8). Raises ValueError on bad input."""
    parts = s.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid year-month {s!r}; expected YYYY-MM")
    year, month = int(parts[0]), int(parts[1])
    if not (1 <= month <= 12):
        raise ValueError(f"Month must be 1-12, got {month}")
    if year < 1981:
        raise ValueError(f"CHIRPS v3 starts in 1981; got {year}")
    return year, month


def months_in_range(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Inclusive list of (year, month) tuples between two endpoints."""
    if start > end:
        raise ValueError(f"start {start} must be <= end {end}")
    out: list[tuple[int, int]] = []
    y, m = start
    while (y, m) <= end:
        out.append((y, m))
        m += 1
        if m == 13:
            m = 1
            y += 1
    return out


def monthly_filename(year: int, month: int) -> str:
    """Canonical CHC filename for a year-month, e.g. chirps-v3.0.2023.08.days_p05.nc."""
    return f"chirps-v3.0.{year:04d}.{month:02d}.days_p05.nc"


def subset_filename(year: int, month: int, roi_tag: str = DEFAULT_ROI_TAG) -> str:
    """Our per-ROI derived filename."""
    return f"chirps-v3.0.{year:04d}.{month:02d}.days_p05.{roi_tag}.nc"


def monthly_url(year: int, month: int, flavor: str = "rnl") -> str:
    """Full URL of the monthly NetCDF on the CHC server."""
    return f"{CHC_BASE}/{flavor}/netcdf/byMonth/{monthly_filename(year, month)}"


def head_exists(url: str, timeout: float = 30.0) -> tuple[bool, int | None, int | None]:
    """HEAD probe. Returns (exists, status_code, content_length).

    Some servers don't allow HEAD; fall back to a 1-byte ranged GET.
    """
    import requests

    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        if resp.status_code == 200:
            cl = int(resp.headers.get("Content-Length", 0)) or None
            return True, 200, cl
        if resp.status_code == 405:  # HEAD not allowed
            resp = requests.get(url, timeout=timeout, stream=True, headers={"Range": "bytes=0-0"})
            if resp.status_code in (200, 206):
                cr = resp.headers.get("Content-Range")  # "bytes 0-0/12345678"
                cl = None
                if cr and "/" in cr:
                    with contextlib.suppress(ValueError):
                        cl = int(cr.split("/")[-1])
                return True, resp.status_code, cl
        return False, resp.status_code, None
    except requests.RequestException:
        return False, None, None


def download_to(url: str, dest: Path, expected_size: int | None = None) -> None:
    """Stream-download a URL to a file with a Rich progress bar."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        total = expected_size or int(resp.headers.get("Content-Length", 0)) or None
        with (
            Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=False,
            ) as progress,
            tmp.open("wb") as fh,
        ):
            task = progress.add_task(dest.name, total=total)
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                fh.write(chunk)
                progress.update(task, advance=len(chunk))
    tmp.replace(dest)


def subset_netcdf(src: Path, dst: Path, area: tuple[float, float, float, float]) -> None:
    """Open `src`, slice to (N, W, S, E) area, write `dst`. Uses xarray."""
    import xarray as xr

    north, west, south, east = area
    with xr.open_dataset(src) as ds:
        # CHIRPS uses lat ascending; sel with slice requires (low, high) i.e. (south, north).
        # Variable name in CHIRPS v3 NetCDF is 'precip' with coords latitude/longitude OR
        # latitude/longitude/time - we accept either lat/lon shorthand.
        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"

        # CHIRPS latitude can be ascending OR descending - handle both.
        lat_vals = ds[lat_name].values
        lat_slice = slice(south, north) if lat_vals[0] < lat_vals[-1] else slice(north, south)

        sub = ds.sel({lat_name: lat_slice, lon_name: slice(west, east)})
        # Encode small to save bytes.
        encoding = {
            v: {"zlib": True, "complevel": 4} for v in sub.data_vars if sub[v].dtype.kind == "f"
        }
        dst.parent.mkdir(parents=True, exist_ok=True)
        sub.to_netcdf(dst, encoding=encoding, engine="h5netcdf")


@app.command()
def main(
    month: str | None = typer.Option(
        None,
        "--month",
        "-m",
        help="Single year-month YYYY-MM (e.g. 2023-08).",
    ),
    start: str | None = typer.Option(None, "--start", help="Start year-month YYYY-MM (inclusive)."),
    end: str | None = typer.Option(None, "--end", help="End year-month YYYY-MM (inclusive)."),
    flavor: str = typer.Option(
        "rnl",
        "--flavor",
        "-f",
        help="Daily disaggregation flavor: 'rnl' (ERA5) or 'sat' (IMERG).",
    ),
    area: tuple[float, float, float, float] = typer.Option(
        DEFAULT_AREA,
        "--area",
        "-a",
        help="ROI as N W S E in degrees (default: Nile + Eastern Mediterranean).",
    ),
    roi_tag: str = typer.Option(
        DEFAULT_ROI_TAG,
        "--roi-tag",
        help="Short tag inserted into the subset filename.",
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", "-o", help="Override default download directory."
    ),
    keep_global: bool = typer.Option(
        False,
        "--keep-global/--no-keep-global",
        help="Keep the giant global NetCDF after subsetting (default: delete).",
    ),
    subset: bool = typer.Option(
        True,
        "--subset/--no-subset",
        help="Subset to ROI after download (default: yes).",
    ),
    probe: bool = typer.Option(
        False,
        "--probe",
        help="Only probe the URL(s); download nothing.",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Re-download even if the subset file exists."
    ),
) -> None:
    """Download CHIRPS v3 daily precipitation (monthly NetCDF), ROI-subset."""
    # ---- Validate flavor ----
    if flavor not in ("rnl", "sat"):
        console.print(f"[red]ERROR[/red] --flavor must be 'rnl' or 'sat', got {flavor!r}.")
        raise typer.Exit(code=2)

    # ---- Resolve month range ----
    try:
        if month is not None:
            ym = parse_year_month(month)
            month_list = [ym]
        elif start is not None and end is not None:
            month_list = months_in_range(parse_year_month(start), parse_year_month(end))
        else:
            console.print("[red]ERROR[/red] supply --month OR both --start and --end.")
            raise typer.Exit(code=2)
    except ValueError as exc:
        console.print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=2) from exc

    # ---- Output dir ----
    if output_dir is None:
        output_dir = default_data_root() / "raw" / "chirps" / str(month_list[0][0])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Plan ----
    plan = Table(title="CHIRPS v3 download plan", show_header=False, title_style="bold cyan")
    plan.add_column(style="dim")
    plan.add_column(style="bold")
    plan.add_row(
        "Months",
        f"{month_list[0][0]:04d}-{month_list[0][1]:02d}"
        f" -> {month_list[-1][0]:04d}-{month_list[-1][1]:02d}"
        f" ({len(month_list)} month(s))",
    )
    plan.add_row("Flavor", f"{flavor} ({'ERA5' if flavor == 'rnl' else 'IMERG'}-based)")
    plan.add_row("Area (N W S E)", f"{area[0]}, {area[1]}, {area[2]}, {area[3]}")
    plan.add_row("ROI tag", roi_tag)
    plan.add_row("Output dir", str(output_dir))
    plan.add_row("Subset", "yes" if subset else "no (keep global only)")
    plan.add_row("Keep global", "yes" if keep_global else "no (delete after subset)")
    plan.add_row("Mode", "probe only" if probe else "download")
    console.print(plan)
    console.print()

    # ---- Ensure requests + xarray ----
    try:
        import requests  # noqa: F401
    except ImportError:
        console.print("[red]ERROR[/red] requests not installed. Run `uv add requests`.")
        raise typer.Exit(code=1) from None
    if subset:
        try:
            import xarray  # noqa: F401
        except ImportError:
            console.print(
                "[red]ERROR[/red] xarray not installed but --subset is on. "
                "Run `uv add xarray` or pass --no-subset."
            )
            raise typer.Exit(code=1) from None

    # ---- Probe ----
    if probe:
        console.print("[bold]Probing CHIRPS v3 monthly URLs (HEAD)...[/bold]")
        any_ok = False
        for y, m in month_list:
            url = monthly_url(y, m, flavor)
            ok, code, size = head_exists(url)
            tag = "[green]OK[/green]" if ok else f"[red]{code}[/red]"
            sz = f" ({size / 1e6:.0f} MB)" if size else ""
            console.print(f"  {tag} {url}{sz}")
            any_ok = any_ok or ok
        console.print()
        if any_ok:
            console.print(
                "[green]At least one URL resolved.[/green] Re-run without --probe to download."
            )
            raise typer.Exit(code=0)
        console.print(
            "[red]No URLs resolved.[/red] Check "
            "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/rnl/netcdf/byMonth/ "
            "for current availability."
        )
        raise typer.Exit(code=1)

    # ---- Download loop ----
    n_done, n_skip, n_fail = 0, 0, 0
    for y, m in month_list:
        subset_path = output_dir / subset_filename(y, m, roi_tag)
        global_path = output_dir / monthly_filename(y, m)
        url = monthly_url(y, m, flavor)

        final_needed = subset_path if subset else global_path
        if final_needed.exists() and not overwrite:
            console.print(f"  [yellow]SKIP[/yellow] {final_needed.name} (exists)")
            n_skip += 1
            continue

        # Probe before download to surface bad URLs fast.
        ok, code, expected_size = head_exists(url)
        if not ok:
            console.print(f"  [red]FAIL[/red] {y}-{m:02d}: HEAD returned {code}: {url}")
            n_fail += 1
            continue

        # Download the global file (we may delete it afterwards).
        if global_path.exists() and not overwrite:
            console.print(f"  using cached global: {global_path.name}")
        else:
            try:
                download_to(url, global_path, expected_size=expected_size)
            except Exception as exc:
                console.print(f"  [red]FAIL[/red] download {y}-{m:02d}: {exc}")
                n_fail += 1
                continue

        # Subset.
        if subset:
            try:
                subset_netcdf(global_path, subset_path, area)
            except Exception as exc:
                console.print(f"  [red]FAIL[/red] subset {y}-{m:02d}: {exc}")
                n_fail += 1
                continue
            sub_mb = subset_path.stat().st_size / 1e6
            console.print(f"  [green]OK[/green] {subset_path.name} ({sub_mb:.2f} MB)")
            if not keep_global:
                with contextlib.suppress(FileNotFoundError):
                    global_path.unlink()

        n_done += 1

    console.print()
    console.print(f"[green]Done: {n_done} downloaded, {n_skip} skipped, {n_fail} failed.[/green]")
    console.print(f"[green]Output: {output_dir}[/green]")
    if n_fail:
        raise typer.Exit(code=1)

    # ---- Verify ----
    console.print("\n[bold]Verifying one subset file...[/bold]")
    candidates = (
        sorted(output_dir.glob("chirps-v3.0.*.days_p05.*.nc"))
        if subset
        else sorted(output_dir.glob("chirps-v3.0.*.days_p05.nc"))
    )
    if not candidates:
        console.print("[yellow]Nothing to verify.[/yellow]")
        return
    sample = candidates[0]
    try:
        import xarray as xr

        with xr.open_dataset(sample) as ds:
            console.print(f"\n[cyan]{sample.name}[/cyan]")
            console.print(f"  Dimensions  : {dict(ds.sizes)}")
            console.print(f"  Variables   : {list(ds.data_vars)}")
            console.print(f"  Coords      : {list(ds.coords)}")
            time_coord = (
                "time"
                if "time" in ds.coords
                else "valid_time"
                if "valid_time" in ds.coords
                else None
            )
            if time_coord:
                t = ds[time_coord]
                console.print(f"  {time_coord:11s} : {t.min().values} -> {t.max().values}")
            lat_name = "latitude" if "latitude" in ds.coords else "lat"
            lon_name = "longitude" if "longitude" in ds.coords else "lon"
            console.print(
                f"  {lat_name:11s} : {float(ds[lat_name].min()):.2f} -> {float(ds[lat_name].max()):.2f}"
            )
            console.print(
                f"  {lon_name:11s} : {float(ds[lon_name].min()):.2f} -> {float(ds[lon_name].max()):.2f}"
            )
            if "precip" in ds.data_vars:
                p = ds["precip"]
                console.print(f"  precip shape: {p.shape}")
                console.print(f"  precip mean : {float(p.mean()):.3f} mm/day")
                console.print(f"  precip max  : {float(p.max()):.2f} mm/day")
    except Exception as exc:
        console.print(f"[yellow]Could not verify: {exc}[/yellow]")


# Make `python 03_download_chirps.py` work as well as `python -m typer`.
def _cli() -> None:
    app()


if __name__ == "__main__":
    app()


# ---- Day-zero compatibility shims kept so the unit tests stay stable. ------
# Older test code referenced these names; we keep thin wrappers so tests do not
# need to be re-imported in a different order.
def parse_month(s: str) -> tuple[date, date]:
    """Backwards-compat: 'YYYY-MM' -> (first_of_month, last_of_month)."""
    y, m = parse_year_month(s)
    first = date(y, m, 1)
    next_first = date(y + (m // 12), (m % 12) + 1, 1)
    last = date.fromordinal(next_first.toordinal() - 1)
    return first, last
