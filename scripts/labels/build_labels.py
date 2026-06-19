"""scripts/labels/build_labels.py - M3 compound-extreme labels (ROI grid).

Generates the 2023 4-class label set on the common ROI 0.05 grid (the CHIRPS /
SPEI grid), at the MODIS 16-day composite cadence, using the deterministic rule
in nilevit.labels and the climatology baselines from 06a-06d. This satisfies the
M3 gate (class-3 prevalence in [0.5%, 8%] + the Aug-2023 Eastern-Med sanity
check) from the cheap sources already on disk - no HLS. The per-tile 224x224
packaging (PRD 4.3) is the M4 step, once HLS yields the real tile index; the same
label function is simply sampled onto each tile there.

Per label date d, on the common grid:
  VHI    = 0.5*VCI(NDVI_d, ndvi_min, ndvi_max) + 0.5*TCI(LST_d, lst_min, lst_max)
  Tmax_z = (Tmax_d - tmax_doy_mean[doy]) / tmax_doy_std[doy]
  SPEI3  = (WB3_d - spei_wb_mean[m]) / spei_wb_std[m],  WB3_d = sum_{m-2..m}(P-PET)
  label  = classify(SPEI3, Tmax_z, VHI)          # nilevit.labels, PRD 4.4

SPEI-3 needs the two months before d, so labels start once a full 3-month window
of 2023 (or supplied 2022-11/12) is available; earlier dates are skipped + noted.

  uv run python scripts/labels/build_labels.py --dry-run
  uv run python scripts/labels/build_labels.py

Outputs:
  data/interim/labels_2023/label_<YYYY-MM-DD>.tif   (uint8 {0,1,2,3}, ROI grid)
  data/interim/labels_2023/prevalence.csv           (per-class pixel fractions)
"""

# typer's standard pattern uses function calls in argument defaults.
# ruff: noqa: B008

from __future__ import annotations

import contextlib
from datetime import date, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

with contextlib.suppress(ImportError):
    import nilevit  # noqa: F401

app = typer.Typer(
    add_completion=False,
    help="Generate 2023 compound-extreme labels on the ROI grid (Script 06/M3).",
)
console = Console()

ROI_TAG = "nile-em"
MODIS_TILES = ("h19v05", "h20v05", "h21v05")
NDVI_SCALE = 1e-4  # MOD13Q1 NDVI digital number -> [-1, 1]
LST_SCALE = 0.02  # MOD11A2 LST digital number -> Kelvin


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


# --------------------------- pure temporal/PET helpers ---------------------------
def modis_16day_dates(year: int) -> list[date]:
    """MOD13Q1 composite start dates: DOY 1, 17, 33, ... (every 16 days)."""
    out, doy = [], 1
    while doy <= 365:
        out.append(date(year, 1, 1) + timedelta(days=doy - 1))
        doy += 16
    return out


def doy_of(d: date) -> int:
    """Day-of-year (1..366)."""
    return d.timetuple().tm_yday


def wb3_window_months(target: date) -> list[tuple[int, int]]:
    """The three (year, month) keys whose P-PET sum forms WB3 ending at target."""
    out = []
    for back in (2, 1, 0):
        mm, yy = target.month - back, target.year
        if mm <= 0:
            mm, yy = mm + 12, yy - 1
        out.append((yy, mm))
    return out


def hargreaves_ra(lat_deg, doy: int):
    """Extraterrestrial radiation Ra (MJ/m^2/day), FAO-56 Eq. 21 (matches 06c)."""
    import numpy as np

    phi = np.deg2rad(lat_deg)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    decl = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(decl), -1.0, 1.0))
    return (
        (24.0 * 60.0 / np.pi)
        * 0.0820
        * dr
        * (ws * np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.sin(ws))
    )


def hargreaves_pet(tmax_c, tmin_c, ra_mj):
    """Daily Hargreaves PET (mm/day); Ra in MJ/m^2/day (->mm via 0.408)."""
    import numpy as np

    tmean = (tmax_c + tmin_c) / 2.0
    tr = np.clip(tmax_c - tmin_c, 0.0, None)
    return 0.0023 * (0.408 * ra_mj) * (tmean + 17.8) * np.sqrt(tr)


# --------------------------- regrid helpers ---------------------------
def _latlon_names(da):
    """Return this DataArray's (lat_dim, lon_dim) names."""
    lat = next((n for n in ("lat", "latitude", "y") if n in da.dims), "lat")
    lon = next((n for n in ("lon", "longitude", "x") if n in da.dims), "lon")
    return lat, lon


def _to_latlon(da):
    """Rename spatial dims to lat/lon and sort both ascending."""
    lat, lon = _latlon_names(da)
    ren = {}
    if lat != "lat":
        ren[lat] = "lat"
    if lon != "lon":
        ren[lon] = "lon"
    if ren:
        da = da.rename(ren)
    return da.sortby("lat").sortby("lon")


def interp_to(da, tlat, tlon):
    """Bilinear regrid a geographic DataArray onto (tlat, tlon)."""
    return _to_latlon(da).interp(lat=tlat, lon=tlon)


def reproject_sinusoidal(path: Path, tlat, tlon):
    """Open a MODIS sinusoidal raster, reproject to EPSG:4326, regrid to target."""
    import rioxarray

    da = rioxarray.open_rasterio(path).squeeze()
    da = da.rio.reproject("EPSG:4326").rename({"y": "lat", "x": "lon"})
    return _to_latlon(da).interp(lat=tlat, lon=tlon)


def modis_mosaic(paths, tlat, tlon):
    """Reproject+regrid each tile to target and coalesce (disjoint tiles)."""
    import xarray as xr

    parts = [reproject_sinusoidal(p, tlat, tlon) for p in paths]
    if len(parts) == 1:
        return parts[0]
    return xr.concat(parts, dim="t").max(dim="t", skipna=True)


# --------------------------- file locators ---------------------------
def _era5_month_file(root: Path, year: int, month: int) -> Path | None:
    hits = sorted((root / "raw" / "era5" / str(year)).glob(f"era5_land_{year}-{month:02d}_*.nc"))
    return hits[0] if hits else None


def _chirps_month_file(root: Path, year: int, month: int) -> Path | None:
    p = (
        root
        / "raw"
        / "chirps"
        / str(year)
        / f"chirps-v3.0.{year:04d}.{month:02d}.days_p05.{ROI_TAG}.nc"
    )
    return p if p.exists() else None


def _modis_files(root: Path, product: str, band_glob: str, year: int, tile: str):
    base = root / "raw" / "modis" / product / str(year)
    hits = sorted(base.glob(f"*{tile}*{band_glob}*{ROI_TAG}.tif"))
    return hits or sorted(base.glob(f"*{tile}*{ROI_TAG}.tif"))


def _nearest_by_name(paths, d: date):
    """Pick the composite whose embedded date (A{YYYYDDD}) is nearest to d."""
    import re

    best, best_gap = paths[0], 10**9
    for p in paths:
        mobj = re.search(r"A(\d{4})(\d{3})", p.name)
        if not mobj:
            continue
        cd = date(int(mobj.group(1)), 1, 1) + timedelta(days=int(mobj.group(2)) - 1)
        gap = abs((cd - d).days)
        if gap < best_gap:
            best, best_gap = p, gap
    return best


# --------------------------- 2023 indicator assembly ---------------------------
def _hourly_daily_minmax(ds):
    """From an hourly ERA5-Land t2m dataset -> (tmax, tmin) daily in degC."""
    tname = "valid_time" if "valid_time" in ds.coords else "time"
    var = "t2m" if "t2m" in ds.data_vars else next(iter(ds.data_vars))
    tc = ds[var] - 273.15
    return tc.resample({tname: "1D"}).max(), tc.resample({tname: "1D"}).min()


def monthly_precip(root: Path, yy: int, mm: int, tlat, tlon):
    """2023 monthly precip total (mm) on target grid, or None if absent."""
    import xarray as xr

    fp = _chirps_month_file(root, yy, mm)
    if fp is None:
        return None
    ds = xr.open_dataset(fp)
    v = ds[next(iter(ds.data_vars))]
    tname = "time" if "time" in v.dims else next(iter(v.dims))
    return interp_to(v.sum(dim=tname), tlat, tlon)


def monthly_pet(root: Path, yy: int, mm: int, tlat, tlon):
    """2023 monthly Hargreaves PET (mm) on target grid, or None if absent."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    fp = _era5_month_file(root, yy, mm)
    if fp is None:
        return None
    ds = xr.open_dataset(fp)
    tmax, tmin = _hourly_daily_minmax(ds)
    tname = "valid_time" if "valid_time" in tmax.coords else "time"
    lat_name, _ = _latlon_names(tmax)
    lat2d = tmax[lat_name].values[:, None] * np.ones(tmax.shape[1:])
    pet = np.zeros(tmax.shape[1:], dtype="float64")
    for i in range(tmax.sizes[tname]):
        d = pd.Timestamp(tmax[tname].values[i]).to_pydatetime().date()
        ra = hargreaves_ra(lat2d, doy_of(d))
        daily = hargreaves_pet(tmax.isel({tname: i}).values, tmin.isel({tname: i}).values, ra)
        pet += np.where(np.isfinite(daily), daily, 0.0)
    template = tmax.isel({tname: 0})
    return interp_to(template.copy(data=pet), tlat, tlon)


def daily_tmax(root: Path, d: date, tlat, tlon):
    """2023 daily Tmax (degC) for date d on target grid."""
    import pandas as pd
    import xarray as xr

    fp = _era5_month_file(root, d.year, d.month)
    if fp is None:
        raise FileNotFoundError(f"missing ERA5 for {d:%Y-%m}")
    ds = xr.open_dataset(fp)
    tmax, _ = _hourly_daily_minmax(ds)
    tname = "valid_time" if "valid_time" in tmax.coords else "time"
    return interp_to(tmax.sel({tname: pd.Timestamp(d)}, method="nearest"), tlat, tlon)


def modis_indicator(root: Path, product: str, band_glob: str, d: date, tlat, tlon):
    """Mosaic the MODIS composite nearest date d across tiles, on target grid."""
    chosen = []
    for tile in MODIS_TILES:
        hits = _modis_files(root, product, band_glob, d.year, tile)
        if hits:
            chosen.append(_nearest_by_name(hits, d))
    if not chosen:
        raise FileNotFoundError(f"no MODIS {product} tifs for {d}")
    return modis_mosaic(chosen, tlat, tlon)


# --------------------------- orchestration ---------------------------
def main_impl(root: Path, year: int, console: Console, *, dry_run: bool) -> int:
    """Core M3 label build; returns the number of label dates written."""
    import numpy as np
    import pandas as pd
    import rioxarray  # noqa: F401
    import xarray as xr

    import nilevit.labels as lab

    dates = modis_16day_dates(year)
    out_dir = root / "interim" / f"labels_{year}"
    if dry_run:
        console.print(
            f"[yellow]Dry run[/yellow]: {len(dates)} candidate dates "
            f"({dates[0]} .. {dates[-1]}); outputs -> {out_dir}"
        )
        return 0

    clim = root / "climatology"
    out_dir.mkdir(parents=True, exist_ok=True)

    spei_mean = xr.open_dataset(clim / "spei_wb_mean.nc")["wb_mean"]
    spei_std = xr.open_dataset(clim / "spei_wb_std.nc")["wb_std"]
    spei_mean = _to_latlon(spei_mean)
    spei_std = _to_latlon(spei_std)
    tlat, tlon = spei_mean["lat"], spei_mean["lon"]

    def clim_mosaic(name: str):
        paths = [clim / f"{name}_{t}.tif" for t in MODIS_TILES]
        paths = [p for p in paths if p.exists()]
        if not paths:
            raise FileNotFoundError(f"no climatology rasters {name}_*.tif in {clim}")
        return modis_mosaic(paths, tlat, tlon)

    ndvi_min = (clim_mosaic("ndvi_min") * NDVI_SCALE).values
    ndvi_max = (clim_mosaic("ndvi_max") * NDVI_SCALE).values
    lst_min = (clim_mosaic("lst_min") * LST_SCALE).values
    lst_max = (clim_mosaic("lst_max") * LST_SCALE).values
    tmax_mu = interp_to(xr.open_dataset(clim / "tmax_doy_mean.nc")["tmax"], tlat, tlon)
    tmax_sd = interp_to(xr.open_dataset(clim / "tmax_doy_std.nc")["tmax"], tlat, tlon)

    p_cache: dict[tuple[int, int], object] = {}
    e_cache: dict[tuple[int, int], object] = {}

    def wb3_on(target_date: date):
        total = None
        for yy, mm in wb3_window_months(target_date):
            if (yy, mm) not in p_cache:
                p_cache[(yy, mm)] = monthly_precip(root, yy, mm, tlat, tlon)
                e_cache[(yy, mm)] = monthly_pet(root, yy, mm, tlat, tlon)
            p, e = p_cache[(yy, mm)], e_cache[(yy, mm)]
            if p is None or e is None:
                return None
            wb = p - e
            total = wb if total is None else total + wb
        return total

    counts = np.zeros(4, dtype="int64")
    grid_px = valid_px = 0
    written, skipped = 0, []
    for d in dates:
        wb3 = wb3_on(d)
        if wb3 is None:
            skipped.append(d.isoformat())
            continue
        m, doy = d.month, doy_of(d)
        spei = lab.zscore(wb3.values, spei_mean.sel(month=m).values, spei_std.sel(month=m).values)
        tz = lab.zscore(
            daily_tmax(root, d, tlat, tlon).values,
            tmax_mu.isel(doy=doy - 1).values,
            tmax_sd.isel(doy=doy - 1).values,
        )
        nd = modis_indicator(root, "MOD13Q1", "NDVI", d, tlat, tlon).values * NDVI_SCALE
        ls = modis_indicator(root, "MOD11A2", "LST_Day", d, tlat, tlon).values * LST_SCALE
        vhi = lab.vhi(lab.vci(nd, ndvi_min, ndvi_max), lab.tci(ls, lst_min, lst_max))
        label = lab.classify(spei, tz, vhi)
        # Score only labelable land: sea / no-data (NaN in any indicator) -> 255.
        valid = np.isfinite(spei) & np.isfinite(tz) & np.isfinite(vhi)
        label = np.where(valid, label, 255).astype("uint8")
        for c in range(4):
            counts[c] += int((label == c).sum())  # 255 excluded automatically
        valid_px += int(valid.sum())
        grid_px += int(valid.size)
        out = xr.DataArray(
            label,
            coords={"lat": tlat, "lon": tlon},
            dims=["lat", "lon"],
        )
        out = (
            out.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
            .rio.write_crs("EPSG:4326")
            .rio.write_nodata(255)
        )
        out.rio.to_raster(out_dir / f"label_{d.isoformat()}.tif")
        written += 1

    total = int(counts.sum())
    frac = counts / total if total else counts.astype("float64")
    pd.DataFrame({"class": [0, 1, 2, 3], "pixels": counts, "fraction": frac}).to_csv(
        out_dir / "prevalence.csv", index=False
    )
    console.print(
        f"\n[cyan]labels_{year}[/cyan]  written: {written}, "
        f"skipped (incomplete SPEI window): {len(skipped)}"
    )
    cov = (valid_px / grid_px) if grid_px else 0.0
    console.print(
        f"  land coverage (valid px): {cov * 100:.1f}%  " f"(sea/no-data excluded from prevalence)"
    )
    console.print(
        f"  fractions  none {frac[0]:.3f}  drought {frac[1]:.3f}  "
        f"heat {frac[2]:.3f}  compound {frac[3]:.4f}"
    )
    gate = 0.005 <= frac[3] <= 0.08
    tag = "[green]PASS[/green]" if gate else "[red]CHECK[/red]"
    console.print(f"  M3 class-3 gate [0.5%, 8%]: {tag} ({frac[3] * 100:.2f}%)")
    return written


@app.command()
def main(
    data_root: Path | None = typer.Option(None, "--data-root", help="Override data/."),
    year: int = typer.Option(2023, "--year", help="Label year."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the plan only."),
) -> None:
    """Build the ROI-grid compound-extreme labels for year (M3)."""
    root = data_root or default_data_root()
    plan = Table(title="build_labels (M3)", show_header=False, title_style="bold cyan")
    plan.add_column(style="dim")
    plan.add_column(style="bold")
    plan.add_row("Data root", str(root))
    plan.add_row("Year", str(year))
    plan.add_row("Cadence", "MODIS 16-day")
    plan.add_row("Grid", "common ROI 0.05 (SPEI grid)")
    plan.add_row("Mode", "dry-run" if dry_run else "run")
    console.print(plan)
    console.print()
    try:
        main_impl(root, year, console, dry_run=dry_run)
    except ImportError as exc:
        console.print(f"[red]ERROR[/red] missing deps: {exc}")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
