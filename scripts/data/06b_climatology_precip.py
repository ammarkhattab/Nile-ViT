"""Script 06b (climatology) - CHIRPS monthly precip stack for the SPEI baseline.

Builds the precipitation half of the SPEI-3 baseline (PRD 4.4). SPEI works on
monthly accumulations, and CHIRPS is fundamentally a monthly product (daily is
derived from it), so we pull the consolidated CHIRPS monthly NetCDF once, subset
it to the ROI, and slice out the baseline years. The result is a compact
per-pixel monthly precip stack that later combines with B2c's PET into the
per-pixel / per-calendar-month water-balance mean+std the SPEI z-score needs.

Note: the daily CHIRPS pulled in B1 is a *separate* consumer - it is the model's
`chirps_p` meteo-input channel (PRD 4.3), not the SPEI label baseline. Monthly
here, daily there; no overlap.

Source: CHIRPS v2.0 consolidated monthly (verified single file, ~2 GB, 1981->
present). v2 monthly is an established climatology reference on the same 0.05
grid as v3, so the SPEI baseline aligns spatially with the v3 data already on
disk; the standardization is robust to the v2/v3 difference. A v3-monthly URL
can be supplied with --url once its layout is confirmed.

Usage:
  uv run python scripts/data/06b_climatology_precip.py --probe          # HEAD only
  uv run python scripts/data/06b_climatology_precip.py --dry-run
  uv run python scripts/data/06b_climatology_precip.py                  # 1991-2020

Output:
  data/climatology/precip_monthly_<start>_<end>.nc   (time, lat, lon) ROI subset
"""

# typer's standard pattern uses function calls in argument defaults.
# ruff: noqa: B008

from __future__ import annotations

import contextlib
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

with contextlib.suppress(ImportError):
    import nilevit  # noqa: F401

app = typer.Typer(
    add_completion=False,
    help="CHIRPS monthly precip climatology for the SPEI baseline (Script 06b).",
)
console = Console()

ROI_BBOX = (22.0, 30.0, 36.0, 37.0)  # W, S, E, N

SOURCES: dict[str, str] = {
    "v2-consolidated": (
        "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/"
        "netcdf/chirps-v2.0.monthly.nc"
    ),
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


def resolve_url(source: str, url_override: str | None) -> str:
    """Pick the download URL from --url or the named source."""
    if url_override:
        return url_override
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; choices: {list(SOURCES)}")
    return SOURCES[source]


def _clean_value(v):
    """Scrub characters that can't round-trip through UTF-8 (lone surrogates)."""
    if isinstance(v, str):
        return v.encode("utf-8", "replace").decode("utf-8")
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return v


def _sanitize_attrs(ds) -> None:
    """In place: scrub surrogate chars from dataset, variable, and coord attrs."""

    def scrub(attrs: dict) -> dict:
        return {_clean_value(k): _clean_value(v) for k, v in attrs.items()}

    ds.attrs = scrub(ds.attrs)
    for name in list(ds.variables):
        ds[name].attrs = scrub(ds[name].attrs)


def head_info(url: str) -> tuple[bool, int, int]:
    """HEAD the URL -> (ok, status_code, content_length_bytes)."""
    import requests

    r = requests.head(url, allow_redirects=True, timeout=60)
    size = int(r.headers.get("Content-Length", 0))
    return r.ok, r.status_code, size


def download_resumable(url: str, dst: Path, console: Console) -> None:
    """Stream `url` to `dst` with HTTP Range resume via a .part file."""
    import requests
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TransferSpeedColumn,
    )

    part = dst.with_name(dst.name + ".part")
    existing = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        # If the server ignored our Range (200 not 206), restart from scratch.
        if existing and r.status_code == 200:
            existing = 0
        total = int(r.headers.get("Content-Length", 0)) + existing
        mode = "ab" if existing else "wb"
        dst.parent.mkdir(parents=True, exist_ok=True)
        with (
            open(part, mode) as fh,
            Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                console=console,
            ) as prog,
        ):
            task = prog.add_task(dst.name, total=total or None, completed=existing)
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                prog.update(task, advance=len(chunk))
    part.replace(dst)


def subset_and_slice(
    src: Path,
    dst: Path,
    area: tuple[float, float, float, float],
    start_year: int,
    end_year: int,
):
    """Subset the global monthly NetCDF to the ROI + year range, write `dst`.

    Opens lazily (NetCDF backends read only the indexed region on .to_netcdf),
    so memory stays small even for a multi-GB source. Returns the subset.
    """
    import xarray as xr

    west, south, east, north = area
    with xr.open_dataset(src) as ds:
        lat = "latitude" if "latitude" in ds.coords else "lat"
        lon = "longitude" if "longitude" in ds.coords else "lon"
        lat_vals = ds[lat].values
        lat_slice = slice(south, north) if lat_vals[0] < lat_vals[-1] else slice(north, south)
        sub = ds.sel({lat: lat_slice, lon: slice(west, east)})
        sub = sub.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
        _sanitize_attrs(sub)
        sub = sub.load()  # pull the small ROI slice into memory
        enc = {v: {"zlib": True, "complevel": 4} for v in sub.data_vars if sub[v].dtype.kind == "f"}
        dst.parent.mkdir(parents=True, exist_ok=True)
        sub.to_netcdf(dst, encoding=enc, engine="h5netcdf")
    return sub


@app.command()
def main(
    source: str = typer.Option(
        "v2-consolidated", "--source", help=f"Named source: {list(SOURCES)}."
    ),
    url: str | None = typer.Option(
        None, "--url", help="Override the download URL (e.g. a v3 monthly file)."
    ),
    start_year: int = typer.Option(1991, "--start-year", help="Baseline first year."),
    end_year: int = typer.Option(2020, "--end-year", help="Baseline last year."),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", "-o", help="Override climatology output dir."
    ),
    keep_global: bool = typer.Option(
        False, "--keep-global", help="Keep the downloaded global file after subset."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Re-download / rebuild even if outputs exist."
    ),
    probe: bool = typer.Option(
        False, "--probe", help="HEAD the source URL and report size; download nothing."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the plan; download/write nothing."),
) -> None:
    """Build the CHIRPS monthly precip climatology stack for [start, end] years."""
    if start_year > end_year:
        console.print("[red]ERROR[/red] --start-year must be <= --end-year.")
        raise typer.Exit(code=2)
    try:
        src_url = resolve_url(source, url)
    except ValueError as exc:
        console.print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=2) from exc

    root = default_data_root()
    out_dir = output_dir or (root / "climatology")
    out_dir.mkdir(parents=True, exist_ok=True)
    global_path = out_dir / Path(src_url).name
    out_path = out_dir / f"precip_monthly_{start_year}_{end_year}.nc"

    plan = Table(title="06b precip climatology plan", show_header=False, title_style="bold cyan")
    plan.add_column(style="dim")
    plan.add_column(style="bold")
    plan.add_row("Source", source if not url else "url-override")
    plan.add_row("URL", src_url)
    plan.add_row("Baseline years", f"{start_year}-{end_year}")
    plan.add_row("ROI bbox (W S E N)", str(ROI_BBOX))
    plan.add_row("Global cache", str(global_path))
    plan.add_row("Output", str(out_path))
    plan.add_row("Keep global", "yes" if keep_global else "no (delete after subset)")
    plan.add_row("Mode", "probe" if probe else ("dry-run" if dry_run else "run"))
    console.print(plan)
    console.print()

    if probe:
        try:
            ok, code, size = head_info(src_url)
        except Exception as exc:
            console.print(f"[red]HEAD failed:[/red] {exc}")
            raise typer.Exit(code=1) from None
        tag = "[green]OK[/green]" if ok else f"[red]{code}[/red]"
        gb = size / 1e9
        console.print(f"{tag} status {code}, size {gb:.2f} GB")
        raise typer.Exit(code=0 if ok else 1)

    if dry_run:
        console.print("[yellow]Dry run - nothing downloaded or written.[/yellow]")
        raise typer.Exit(code=0)

    if out_path.exists() and not overwrite:
        console.print(f"[yellow]SKIP[/yellow] {out_path.name} exists (use --overwrite).")
        raise typer.Exit(code=0)

    try:
        import xarray  # noqa: F401
    except ImportError:
        console.print("[red]ERROR[/red] xarray not installed.")
        raise typer.Exit(code=1) from None

    # ---- Download global monthly (resumable) ----
    if global_path.exists() and not overwrite:
        console.print(f"using cached global: {global_path.name}")
    else:
        console.print(f"Downloading {src_url} ...")
        try:
            download_resumable(src_url, global_path, console)
        except Exception as exc:
            console.print(f"[red]FAIL[/red] download: {exc}")
            raise typer.Exit(code=1) from None

    # ---- Subset to ROI + baseline years ----
    console.print("Subsetting to ROI and baseline years...")
    try:
        sub = subset_and_slice(global_path, out_path, ROI_BBOX, start_year, end_year)
    except Exception as exc:
        console.print(f"[red]FAIL[/red] subset: {exc}")
        raise typer.Exit(code=1) from None

    if not keep_global:
        with contextlib.suppress(FileNotFoundError):
            global_path.unlink()
        console.print(f"deleted global cache {global_path.name}")

    # ---- Verify ----
    import numpy as np

    pvar = "precip" if "precip" in sub.data_vars else next(iter(sub.data_vars))
    n_months = sub.sizes.get("time", 0)
    expected = (end_year - start_year + 1) * 12
    p = sub[pvar]
    console.print(f"\n[cyan]{out_path.name}[/cyan]")
    console.print(f"  dims      : {dict(sub.sizes)}")
    console.print(f"  months    : {n_months} (expected {expected})")
    console.print(f"  precip mean: {float(np.nanmean(p.values)):.3f} mm/month")
    if n_months != expected:
        console.print(f"  [yellow]note: month count != {expected}; check source coverage[/yellow]")
    console.print("\n[green]Precip climatology stack complete.[/green]")


if __name__ == "__main__":
    app()
