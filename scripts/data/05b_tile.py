"""Script 05b - tile HLS imagery into 224x224 patches (Zarr + GeoParquet).

The second half of the gridding stage. It takes one HLS granule (6 Prithvi bands
on the native UTM 30 m grid) and cuts it into non-overlapping 224x224 patches,
writing:
  - the imagery cube to a Zarr store: image[sample, band, y, x], uint16
  - one metadata row per patch to a GeoParquet index (sample_id, tile, center
    lon/lat, date, region, cloud_pct, valid_pct, patch row/col)

This is the SPATIAL tiling primitive. The temporal T=3 cube (t-30d, t, t+15d
from PRD 4.3) is assembled later by stacking per-date patches that share a
(tile, row, col) - which needs the surrounding-month data (B1) anyway. Building
the per-date tiler first keeps it testable on the August data we have.

Bands written, in order (PRD 4.3): blue, green, red, nir_narrow, swir1, swir2.
S30 and L30 map these to different band codes; both are handled.

Quality flags
-------------
  cloud_pct : fraction of patch pixels flagged cloud/shadow/adjacent in Fmask
  valid_pct : fraction of patch pixels with valid (non-fill) reflectance
Patches below --min-valid are dropped (too much missing data).

Usage
-----
Preview the patch grid for one date, write nothing:
  uv run python scripts/data/05b_tile.py --tile T36RUU --date 2023-08-01 --dry-run

Tile one date:
  uv run python scripts/data/05b_tile.py --tile T36RUU --date 2023-08-01

Tile every available HLS date for the tile/year:
  uv run python scripts/data/05b_tile.py --tile T36RUU --all-dates

Output
------
  <repo>/data/interim/tiles_<tile>_<year>.zarr      (image cube)
  <repo>/data/interim/tiles_<tile>_<year>.parquet   (metadata index)
"""

# typer's standard pattern uses function calls in argument defaults.
# ruff: noqa: B008

from __future__ import annotations

import contextlib
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

with contextlib.suppress(ImportError):
    import nilevit  # noqa: F401

app = typer.Typer(
    add_completion=False,
    help="Tile HLS imagery into 224x224 patches -> Zarr + GeoParquet (Script 05b).",
)
console = Console()


PATCH = 224
BAND_NAMES = ["blue", "green", "red", "nir_narrow", "swir1", "swir2"]

HLS_BAND_MAP: dict[str, dict[str, str]] = {
    "S30": {
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "nir_narrow": "B8A",
        "swir1": "B11",
        "swir2": "B12",
    },
    "L30": {
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "nir_narrow": "B05",
        "swir1": "B06",
        "swir2": "B07",
    },
}

HLS_FILL = -9999  # HLS scaled-reflectance fill value
FMASK_FILL = 255  # HLS Fmask no-observation value
# Fmask cloud-related bits: bit1 cloud (2), bit2 adjacent (4), bit3 shadow (8).
FMASK_CLOUD_BITS = 0b1110  # = 14


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


def doy_to_date(year: int, doy: int) -> date:
    """(year, day-of-year) -> date. DOY is 1-based."""
    return date(year, 1, 1) + timedelta(days=doy - 1)


def hls_sensor_from_name(name: str) -> str | None:
    """'HLS.S30...' -> 'S30'; 'HLS.L30...' -> 'L30'; else None."""
    m = re.search(r"HLS\.(S30|L30)\.", name)
    return m.group(1) if m else None


def hls_date_from_name(name: str) -> date | None:
    """Parse acquisition date from the YYYYDDD token in an HLS filename."""
    m = re.search(r"\.(\d{4})(\d{3})T\d+", name)
    if not m:
        return None
    year, doy = int(m.group(1)), int(m.group(2))
    if not (1 <= doy <= 366):
        return None
    return doy_to_date(year, doy)


def patch_grid(height: int, width: int, patch: int = PATCH) -> list[tuple[int, int, int, int]]:
    """Non-overlapping patch origins as (row, col, y0, x0). Edge remainder dropped."""
    rows, cols = height // patch, width // patch
    return [(r, c, r * patch, c * patch) for r in range(rows) for c in range(cols)]


def make_sample_id(tile: str, d: date, row: int, col: int) -> str:
    """e.g. 'T36RUU_2023-08-01_r05c08'."""
    return f"{tile}_{d.isoformat()}_r{row:02d}c{col:02d}"


def classify_region(lon: float, lat: float) -> str:
    """Map a center coordinate to a PRD 4.1 sub-region label.

    Priority: delta core, then north coast, else Eastern-Med shelf.
    """
    if 29.5 <= lon <= 32.5 and 30.5 <= lat <= 31.5:
        return "delta"
    if 25.0 <= lon <= 35.0 and 31.0 <= lat <= 32.0:
        return "n_coast"
    return "em_shelf"


def cloud_fraction(fmask_patch) -> float:
    """Fraction of observed pixels flagged cloud/shadow/adjacent.

    Pixels equal to FMASK_FILL are treated as unobserved and excluded from the
    denominator. Returns 1.0 if the patch is entirely unobserved.
    """
    import numpy as np

    arr = np.asarray(fmask_patch)
    observed = arr != FMASK_FILL
    n_obs = int(observed.sum())
    if n_obs == 0:
        return 1.0
    cloudy = (arr & FMASK_CLOUD_BITS) != 0
    return float((cloudy & observed).sum()) / n_obs


def valid_fraction(red_patch, fill: int = HLS_FILL) -> float:
    """Fraction of patch pixels with a valid (non-fill, finite) reflectance."""
    import numpy as np

    arr = np.asarray(red_patch)
    finite = np.isfinite(arr)
    valid = finite & (arr != fill)
    return float(valid.sum()) / arr.size


def find_hls_b04(hls_dir: Path, want_date: date | None) -> tuple[Path, date, str] | None:
    """Find an HLS B04 (red) file; the granule prefix yields the other bands."""
    cands: list[tuple[date, str, Path]] = []
    for p in sorted(hls_dir.glob("*.B04.tif")):
        d, s = hls_date_from_name(p.name), hls_sensor_from_name(p.name)
        if d and s:
            cands.append((d, s, p))
    if not cands:
        return None
    if want_date is not None:
        for d, s, p in cands:
            if d == want_date:
                return p, d, s
        return None
    d, s, p = cands[0]
    return p, d, s


def band_path(b04_path: Path, band_code: str) -> Path:
    """Derive a sibling band file path from the B04 path by code substitution."""
    return b04_path.with_name(b04_path.name.replace(".B04.", f".{band_code}."))


@app.command()
def main(
    tile: str = typer.Option("T36RUU", "--tile", "-t", help="HLS MGRS tile id."),
    year: int = typer.Option(2023, "--year", "-y", help="Year to tile."),
    target_date: str | None = typer.Option(None, "--date", "-d", help="Single date YYYY-MM-DD."),
    all_dates: bool = typer.Option(
        False, "--all-dates", help="Tile every available HLS date for the tile/year."
    ),
    min_valid: float = typer.Option(
        0.5, "--min-valid", help="Drop patches with valid_pct below this (0..1)."
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", "-o", help="Override interim output directory."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview the patch grid; write nothing."),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Recreate the Zarr/Parquet instead of appending."
    ),
) -> None:
    """Tile HLS imagery into 224x224 patches for one date or all dates."""
    root = default_data_root()
    hls_dir = root / "raw" / "hls" / tile / str(year)
    if not hls_dir.exists():
        console.print(f"[red]ERROR[/red] HLS dir not found: {hls_dir}")
        raise typer.Exit(code=1)

    # ---- Resolve which dates to process ----
    want_date: date | None = None
    if target_date is not None:
        try:
            want_date = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError as exc:
            console.print(f"[red]ERROR[/red] --date must be YYYY-MM-DD: {exc}")
            raise typer.Exit(code=2) from exc

    all_b04 = sorted(hls_dir.glob("*.B04.tif"))
    avail = sorted({d for p in all_b04 if (d := hls_date_from_name(p.name))})
    if all_dates:
        dates = avail
    elif want_date is not None:
        dates = [want_date]
    elif avail:
        dates = [avail[0]]
    else:
        console.print(f"[red]ERROR[/red] no HLS B04 files in {hls_dir}")
        raise typer.Exit(code=1)

    if output_dir is None:
        output_dir = root / "interim"
    output_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = output_dir / f"tiles_{tile}_{year}.zarr"
    parquet_path = output_dir / f"tiles_{tile}_{year}.parquet"

    plan = Table(title="05b tiling plan", show_header=False, title_style="bold cyan")
    plan.add_column(style="dim")
    plan.add_column(style="bold")
    plan.add_row("Tile / year", f"{tile} / {year}")
    plan.add_row("Dates", f"{len(dates)} ({dates[0]} .. {dates[-1]})")
    plan.add_row("Patch size", f"{PATCH}x{PATCH}")
    plan.add_row("Bands", ", ".join(BAND_NAMES))
    plan.add_row("Min valid_pct", f"{min_valid}")
    plan.add_row("Zarr out", str(zarr_path))
    plan.add_row("Parquet out", str(parquet_path))
    plan.add_row("Mode", "dry-run" if dry_run else "write")
    console.print(plan)
    console.print()

    # ---- Heavy imports ----
    try:
        import geopandas as gpd
        import numpy as np
        import pandas as pd
        import rioxarray
        import xarray as xr
        from pyproj import Transformer
        from shapely.geometry import Point
    except ImportError as exc:
        console.print(f"[red]ERROR[/red] missing deps: {exc}")
        raise typer.Exit(code=1) from None

    if overwrite:
        import shutil

        if zarr_path.exists():
            shutil.rmtree(zarr_path)
        parquet_path.unlink(missing_ok=True)

    total_written, total_dropped = 0, 0
    for d in dates:
        ref = find_hls_b04(hls_dir, d)
        if ref is None:
            console.print(f"[yellow]SKIP {d}: no B04 file[/yellow]")
            continue
        b04_path, _, sensor = ref
        band_codes = HLS_BAND_MAP[sensor]

        # Load the 6 bands + Fmask onto the native HLS grid.
        red_da = rioxarray.open_rasterio(b04_path)  # for grid + valid mask
        h, w = red_da.shape[-2:]
        grid = patch_grid(h, w)

        if dry_run:
            console.print(
                f"[bold]{d}[/bold] sensor {sensor}: {h}x{w} -> "
                f"{len(grid)} patches ({h // PATCH}x{w // PATCH})"
            )
            red_da.close()
            continue

        console.print(f"[bold]{d}[/bold] sensor {sensor}: loading 6 bands + Fmask...")
        band_arrs = []
        for name in BAND_NAMES:
            bp = band_path(b04_path, band_codes[name])
            if not bp.exists():
                console.print(f"  [red]missing band {band_codes[name]} ({bp.name})[/red]")
                band_arrs = []
                break
            da = rioxarray.open_rasterio(bp)
            band_arrs.append(da.values[0])  # (H, W)
        if not band_arrs:
            red_da.close()
            total_dropped += len(grid)
            continue

        stack = np.stack(band_arrs, axis=0).astype(np.uint16)  # (6, H, W)
        fmask_path = band_path(b04_path, "Fmask")
        fmask = (
            rioxarray.open_rasterio(fmask_path).values[0]
            if fmask_path.exists()
            else np.zeros((h, w), dtype=np.uint8)
        )
        red = red_da.values[0]

        # UTM -> lon/lat transformer for patch centers.
        crs = red_da.rio.crs
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        xs, ys = red_da.x.values, red_da.y.values

        images, rows = [], []
        for r, c, y0, x0 in grid:
            sl = (slice(y0, y0 + PATCH), slice(x0, x0 + PATCH))
            vpct = valid_fraction(red[sl])
            if vpct < min_valid:
                total_dropped += 1
                continue
            cpct = cloud_fraction(fmask[sl])
            cy, cx = y0 + PATCH // 2, x0 + PATCH // 2
            lon, lat = transformer.transform(float(xs[cx]), float(ys[cy]))
            sid = make_sample_id(tile, d, r, c)
            images.append(stack[:, sl[0], sl[1]])
            rows.append(
                {
                    "sample_id": sid,
                    "mgrs_tile": tile,
                    "date": d.isoformat(),
                    "row": r,
                    "col": c,
                    "center_lon": round(lon, 6),
                    "center_lat": round(lat, 6),
                    "region": classify_region(lon, lat),
                    "cloud_pct": round(cpct, 4),
                    "valid_pct": round(vpct, 4),
                    "sensor": sensor,
                    "geometry": Point(lon, lat),
                }
            )
        red_da.close()

        if not images:
            console.print(f"  [yellow]{d}: all patches dropped (valid<{min_valid})[/yellow]")
            continue

        # ---- Write imagery to Zarr (append along sample) ----
        arr = np.stack(images, axis=0)  # (N, 6, 224, 224)
        ds = xr.Dataset(
            {"image": (("sample", "band", "y", "x"), arr)},
            coords={"sample": [row["sample_id"] for row in rows], "band": BAND_NAMES},
        )
        if zarr_path.exists():
            ds.to_zarr(zarr_path, append_dim="sample", mode="a")
        else:
            ds.to_zarr(
                zarr_path,
                mode="w",
                encoding={"image": {"chunks": (1, 6, PATCH, PATCH)}},
            )

        # ---- Append metadata to GeoParquet ----
        gdf_new = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
        if parquet_path.exists():
            gdf_old = gpd.read_parquet(parquet_path)
            gdf = gpd.GeoDataFrame(
                pd.concat([gdf_old, gdf_new], ignore_index=True),
                geometry="geometry",
                crs="EPSG:4326",
            )
        else:
            gdf = gdf_new
        gdf.to_parquet(parquet_path)

        total_written += len(images)
        console.print(f"  [green]{d}: wrote {len(images)} patches[/green]")

    console.print()
    if dry_run:
        console.print("[yellow]Dry run - nothing written.[/yellow]")
        raise typer.Exit(code=0)

    console.print(
        f"[green]Done: {total_written} patches written, " f"{total_dropped} dropped.[/green]"
    )

    # ---- Verify ----
    if zarr_path.exists():
        zds = xr.open_zarr(zarr_path)
        gdf = gpd.read_parquet(parquet_path)
        console.print(f"\n[cyan]{zarr_path.name}[/cyan]")
        console.print(f"  image dims: {dict(zds.sizes)}")
        console.print(f"  dtype     : {zds['image'].dtype}")
        console.print(f"  index rows: {len(gdf)}")
        console.print(f"  regions   : {gdf['region'].value_counts().to_dict()}")
        console.print(
            f"  cloud_pct : mean {gdf['cloud_pct'].mean():.3f}, "
            f"valid_pct mean {gdf['valid_pct'].mean():.3f}"
        )
        zds.close()


if __name__ == "__main__":
    app()
