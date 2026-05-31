"""Script 05a - spatial harmonization onto the HLS reference grid.

The first half of the gridding stage. It takes the four raw sources (which live
in three different CRSs and four resolutions) and reprojects them onto ONE
common grid: the HLS granule's UTM grid at 30 m (PRD decision Q1).

What it produces for a single (tile, date):
  - A multi-band GeoTIFF where every band shares the HLS grid exactly:
      band 1: HLS red (B04)            - the reference, for visual sanity
      band 2: MODIS NDVI  (resampled)  - vegetation, for VCI / VHI
      band 3: MODIS LST   (resampled)  - surface temp, for TCI / VHI
      band 4: ERA5 t2m daily mean      - air temp, for heat / PET
      band 5: CHIRPS precip daily      - rainfall, for SPEI
  - A printed alignment check: all layers share CRS + transform + shape.

This proves the warping machinery works and aligns before Script 05b tiles the
ROI into 224x224 patches and 05/06 build the label cube.

Reprojection uses rioxarray.reproject_match, which warps a DataArray to match a
reference DataArray's CRS, transform, and shape in one call. MODIS v05 tiles are
mosaicked first; meteo NetCDFs are tagged EPSG:4326 then matched.

Usage
-----
Inventory only (discover files + dates, warp nothing):
  uv run python scripts/data/05a_harmonize.py --inventory

Harmonize one tile + date (auto-picks first HLS date if --date omitted):
  uv run python scripts/data/05a_harmonize.py --tile T36RUU --date 2023-08-01

Output
------
  <repo>/data/interim/harmonized_<tile>_<YYYY-MM-DD>.tif
"""

# typer's standard pattern uses function calls in argument defaults.
# ruff: noqa: B008

from __future__ import annotations

import contextlib
import re
from datetime import date, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

with contextlib.suppress(ImportError):
    import nilevit  # noqa: F401

app = typer.Typer(
    add_completion=False,
    help="Harmonize HLS + MODIS + ERA5 + CHIRPS onto one 30 m HLS grid (Script 05a).",
)
console = Console()


# HLS Prithvi 6-band selection differs by sensor (S30=Sentinel, L30=Landsat).
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
    """'HLS.S30.T36RUU...' -> 'S30'; 'HLS.L30...' -> 'L30'; else None."""
    m = re.search(r"HLS\.(S30|L30)\.", name)
    return m.group(1) if m else None


def hls_date_from_name(name: str) -> date | None:
    """Parse acquisition date from an HLS filename.

    HLS files encode date as YYYYDDD before a 'T' time, e.g.
    'HLS.S30.T36RUU.2023213T083559.v2.0.B04.tif' -> 2023-08-01 (DOY 213).
    """
    m = re.search(r"\.(\d{4})(\d{3})T\d+", name)
    if not m:
        return None
    year, doy = int(m.group(1)), int(m.group(2))
    if not (1 <= doy <= 366):
        return None
    return doy_to_date(year, doy)


def modis_composite_from_name(name: str) -> tuple[int, int] | None:
    """Parse (year, doy) of a MODIS composite from its filename.

    e.g. 'MOD13Q1.A2023209.h20v05...' -> (2023, 209).
    """
    m = re.search(r"\.A(\d{4})(\d{3})\.", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def select_modis_composite(target: date, composite_doys: list[int], year: int) -> int | None:
    """Pick the composite whose start DOY is the latest one <= target's DOY.

    MODIS composites are labelled by their start DOY; a composite covers
    [start, next_start). We choose the most recent composite that has begun on
    or before `target`.
    """
    if not composite_doys:
        return None
    target_doy = (target - date(year, 1, 1)).days + 1
    eligible = sorted(d for d in composite_doys if d <= target_doy)
    if eligible:
        return eligible[-1]
    # target precedes all composites; fall back to the earliest available.
    return sorted(composite_doys)[0]


def find_hls_reference(hls_dir: Path, want_date: date | None) -> tuple[Path, date, str] | None:
    """Find an HLS red-band (B04) file to use as the reference grid.

    Returns (path, date, sensor) or None. If want_date is given, matches it;
    otherwise returns the earliest available date.
    """
    candidates: list[tuple[date, str, Path]] = []
    for p in sorted(hls_dir.glob("*.B04.tif")):
        d = hls_date_from_name(p.name)
        s = hls_sensor_from_name(p.name)
        if d and s:
            candidates.append((d, s, p))
    if not candidates:
        return None
    if want_date is not None:
        for d, s, p in candidates:
            if d == want_date:
                return p, d, s
        return None
    d, s, p = candidates[0]
    return p, d, s


@app.command()
def main(
    tile: str = typer.Option("T36RUU", "--tile", "-t", help="HLS MGRS tile id."),
    year: int = typer.Option(2023, "--year", "-y", help="Year to harmonize."),
    target_date: str | None = typer.Option(
        None, "--date", "-d", help="Target date YYYY-MM-DD (default: first HLS date)."
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", "-o", help="Override default interim output directory."
    ),
    inventory: bool = typer.Option(
        False, "--inventory", help="List discovered files + dates; warp nothing."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Re-create the stack even if it exists."
    ),
) -> None:
    """Harmonize all sources onto the HLS 30 m grid for one tile + date."""
    root = default_data_root()
    hls_dir = root / "raw" / "hls" / tile / str(year)
    ndvi_dir = root / "raw" / "modis" / "MOD13Q1" / str(year)
    lst_dir = root / "raw" / "modis" / "MOD11A2" / str(year)
    era5_dir = root / "raw" / "era5" / str(year)
    chirps_dir = root / "raw" / "chirps" / str(year)

    want_date: date | None = None
    if target_date is not None:
        from datetime import datetime

        try:
            want_date = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError as exc:
            console.print(f"[red]ERROR[/red] --date must be YYYY-MM-DD: {exc}")
            raise typer.Exit(code=2) from exc

    # ---- Inventory ----
    hls_b04 = sorted(hls_dir.glob("*.B04.tif"))
    hls_dates = sorted({d for p in hls_b04 if (d := hls_date_from_name(p.name))})
    ndvi_tiles = sorted(ndvi_dir.glob("*.tif"))
    lst_tiles = sorted(lst_dir.glob("*.tif"))
    era5_files = sorted(era5_dir.glob("*.nc"))
    chirps_files = sorted(chirps_dir.glob("*.nc"))

    inv = Table(title="05a harmonization inventory", show_header=False, title_style="bold cyan")
    inv.add_column(style="dim")
    inv.add_column(style="bold")
    inv.add_row("HLS dir", str(hls_dir))
    inv.add_row("HLS B04 files", str(len(hls_b04)))
    inv.add_row("HLS dates", ", ".join(d.isoformat() for d in hls_dates) or "[red]none[/red]")
    inv.add_row("MODIS NDVI tiles", str(len(ndvi_tiles)))
    inv.add_row("MODIS LST tiles", str(len(lst_tiles)))
    inv.add_row("ERA5 files", str(len(era5_files)))
    inv.add_row("CHIRPS files", str(len(chirps_files)))
    console.print(inv)
    console.print()

    if inventory:
        # Show MODIS composite DOYs discovered.
        ndvi_doys = sorted({c[1] for p in ndvi_tiles if (c := modis_composite_from_name(p.name))})
        lst_doys = sorted({c[1] for p in lst_tiles if (c := modis_composite_from_name(p.name))})
        console.print(f"[bold]NDVI composite DOYs:[/bold] {ndvi_doys}")
        console.print(f"[bold]LST composite DOYs:[/bold]  {lst_doys}")
        console.print("\n[yellow]Inventory mode - nothing warped.[/yellow]")
        raise typer.Exit(code=0)

    # ---- Resolve reference HLS grid ----
    ref = find_hls_reference(hls_dir, want_date)
    if ref is None:
        which = f"date {want_date.isoformat()}" if want_date else "any date"
        console.print(f"[red]ERROR[/red] no HLS B04 file found for {which} in {hls_dir}.")
        raise typer.Exit(code=1)
    ref_path, ref_date, sensor = ref
    console.print(
        f"[green]Reference grid:[/green] {ref_path.name}  "
        f"(date {ref_date.isoformat()}, sensor {sensor})\n"
    )

    if output_dir is None:
        output_dir = root / "interim"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"harmonized_{tile}_{ref_date.isoformat()}.tif"
    if out_path.exists() and not overwrite:
        console.print(f"[yellow]{out_path.name} exists; use --overwrite.[/yellow]")
        raise typer.Exit(code=0)

    # ---- Heavy imports ----
    try:
        import numpy as np
        import rioxarray
        import xarray as xr
        from rasterio.enums import Resampling
        from rioxarray.merge import merge_arrays
    except ImportError as exc:
        console.print(f"[red]ERROR[/red] missing geo deps: {exc}")
        raise typer.Exit(code=1) from None

    # ---- Reference DataArray ----
    ref_da = rioxarray.open_rasterio(ref_path, masked=True)
    console.print(
        f"  Reference CRS: {ref_da.rio.crs}, shape {tuple(ref_da.shape)}, "
        f"res {tuple(round(r, 1) for r in ref_da.rio.resolution())}"
    )

    def mosaic_and_match(tile_paths: list[Path], doys: list[int], label: str):
        """Mosaic the chosen MODIS composite's tiles and match the HLS grid."""
        chosen = select_modis_composite(ref_date, doys, year)
        sel = [p for p in tile_paths if (c := modis_composite_from_name(p.name)) and c[1] == chosen]
        if not sel:
            console.print(f"  [yellow]{label}: no tiles for composite DOY {chosen}[/yellow]")
            return None
        console.print(f"  {label}: composite DOY {chosen}, mosaicking {len(sel)} tile(s)")
        arrs = [rioxarray.open_rasterio(p, masked=True) for p in sel]
        mosaic = merge_arrays(arrs) if len(arrs) > 1 else arrs[0]
        matched = mosaic.rio.reproject_match(ref_da, resampling=Resampling.bilinear)
        for a in arrs:
            a.close()
        return matched

    def meteo_match(nc_files: list[Path], var: str, label: str):
        """Pick the NetCDF containing `var` + ref_date, daily-mean it, match grid.

        Iterates candidate files (e.g. a 1-day smoke file and a full-month file)
        and uses the first whose time coord actually covers ref_date.
        """
        if not nc_files:
            console.print(f"  [yellow]{label}: no NetCDF found[/yellow]")
            return None
        target = np.datetime64(ref_date)
        for f in nc_files:
            ds = xr.open_dataset(f)
            tname = "valid_time" if "valid_time" in ds.coords else "time"
            has_var = var in ds.data_vars
            has_date = (
                tname in ds.coords and (ds[tname].values.astype("datetime64[D]") == target).any()
            )
            if not (has_var and has_date):
                ds.close()
                continue
            day = ds[var].sel({tname: ref_date.isoformat()})
            if tname in day.dims:  # multiple sub-daily steps -> daily mean
                day = day.mean(dim=tname)
            lon = "longitude" if "longitude" in day.coords else "lon"
            lat = "latitude" if "latitude" in day.coords else "lat"
            day = day.rio.write_crs("EPSG:4326").rio.set_spatial_dims(x_dim=lon, y_dim=lat)
            matched = day.rio.reproject_match(ref_da, resampling=Resampling.bilinear)
            ds.close()
            console.print(f"  {label}: matched from {f.name}")
            return matched
        console.print(
            f"  [yellow]{label}: no file has {var!r} covering " f"{ref_date.isoformat()}[/yellow]"
        )
        return None

    # ---- Warp each source ----
    console.print("\n[bold]Reprojecting sources onto the HLS grid...[/bold]")
    ndvi_doys = sorted({c[1] for p in ndvi_tiles if (c := modis_composite_from_name(p.name))})
    lst_doys = sorted({c[1] for p in lst_tiles if (c := modis_composite_from_name(p.name))})
    ndvi_m = mosaic_and_match(ndvi_tiles, ndvi_doys, "NDVI")
    lst_m = mosaic_and_match(lst_tiles, lst_doys, "LST")
    t2m_m = meteo_match(era5_files, "t2m", "ERA5 t2m")
    precip_m = meteo_match(chirps_files, "precip", "CHIRPS precip")

    # ---- Assemble + verify ----
    layers = {
        "HLS_B04_red": ref_da,
        "MODIS_NDVI": ndvi_m,
        "MODIS_LST": lst_m,
        "ERA5_t2m": t2m_m,
        "CHIRPS_precip": precip_m,
    }
    present = {k: v for k, v in layers.items() if v is not None}

    console.print("\n[bold]Alignment check:[/bold]")
    ref_shape = ref_da.shape[-2:]
    all_aligned = True
    check = Table(show_header=True, header_style="bold")
    check.add_column("Layer")
    check.add_column("Shape (H,W)")
    check.add_column("CRS match")
    check.add_column("Aligned")
    for name, da in present.items():
        shp = tuple(da.shape[-2:])
        crs_ok = str(da.rio.crs) == str(ref_da.rio.crs)
        aligned = shp == tuple(ref_shape) and crs_ok
        all_aligned = all_aligned and aligned
        check.add_row(
            name,
            str(shp),
            "yes" if crs_ok else "[red]no[/red]",
            "[green]yes[/green]" if aligned else "[red]NO[/red]",
        )
    console.print(check)

    if not all_aligned:
        console.print("\n[red]Some layers are not aligned - not writing output.[/red]")
        raise typer.Exit(code=1)

    # ---- Stack + write ----
    band_names = list(present.keys())
    stacked = xr.concat([present[n].squeeze(drop=True) for n in band_names], dim="band")
    stacked = stacked.assign_coords(band=range(1, len(band_names) + 1))
    stacked.rio.write_crs(ref_da.rio.crs, inplace=True)
    stacked.rio.to_raster(out_path, compress="deflate")

    # Tag band descriptions for QGIS readability.
    with contextlib.suppress(Exception):
        import rasterio

        with rasterio.open(out_path, "r+") as dst:
            for i, nm in enumerate(band_names, start=1):
                dst.set_band_description(i, nm)

    size_mb = out_path.stat().st_size / 1e6
    console.print(f"\n[green]Wrote {out_path.name}[/green] ({size_mb:.1f} MB)")
    console.print(f"  Bands: {', '.join(f'{i + 1}={n}' for i, n in enumerate(band_names))}")

    # ---- Quick per-layer stats (scaled units) ----
    console.print("\n[bold]Per-layer stats (raw scaled values):[/bold]")
    for name, da in present.items():
        vals = da.values[np.isfinite(da.values)]
        if vals.size:
            console.print(
                f"  {name:16s} min {float(vals.min()):.1f}  "
                f"max {float(vals.max()):.1f}  mean {float(vals.mean()):.1f}"
            )
    ref_da.close()
    console.print("\n[green]Harmonization complete.[/green]")


if __name__ == "__main__":
    app()
