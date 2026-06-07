"""Script 06c (climatology) - ERA5-Land temperature baselines.

Builds the two temperature-derived climatology references for the Script 06
labels (PRD 4.4), in a single streaming pass over 1991-2020:

  1. Heat-z baseline : per-day-of-year Tmax mean and std (with a centered
     +/-15-day window), used by  Tmax_z = (Tmax - mu_doy) / sigma_doy.
  2. PET (SPEI half) : monthly Hargreaves PET totals, combined with 06b's
     monthly precip in 06d to form the water-balance SPEI baseline.

Source: hourly ERA5-Land 2m temperature over the ROI via CDS
(`reanalysis-era5-land`, the PRD 4.2 dataset). Daily Tmax/Tmin are the max/min
of each day's 24 hourly values - computed locally, because the
`derived-era5-land-daily-statistics` endpoint has a known bug where
daily_maximum/daily_minimum return hourly arrays. See docs/B2_CLIMATOLOGY.md
(Decision 4) for why ERA5-Land stands in for CHIRTS here.

Reduce-don't-hoard: each year is downloaded, folded into running accumulators,
then (optionally) deleted. Accumulators + completed years are checkpointed after
each year, so a CDS drop / queue timeout just resumes.

  # smoke-test one year first, then the full baseline:
  uv run python scripts/data/06c_climatology_temp.py --start-year 2020 --end-year 2020
  uv run python scripts/data/06c_climatology_temp.py

Outputs (native ERA5-Land 0.1 grid; Script 06 reprojects to the HLS grid):
  data/climatology/tmax_doy_mean.nc   tmax_doy_std.nc      (doy, lat, lon)
  data/climatology/pet_monthly_1991_2020.nc                (time=360, lat, lon)
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
    help="ERA5-Land Tmax per-DOY climatology + monthly PET (Script 06c).",
)
console = Console()

DATASET = "reanalysis-era5-land"
# (N, W, S, E) - matches script 02's DEFAULT_AREA.
DEFAULT_AREA: tuple[float, float, float, float] = (37.0, 22.0, 30.0, 36.0)
TVAR = "2m_temperature"  # CDS name; stored as 't2m'
DOY_WINDOW = 31  # centered window (+/-15 days) for the heat-z baseline


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


def build_request(year: int, month: int, area: tuple[float, float, float, float]) -> dict:
    """CDS request for one month of hourly ERA5-Land 2m temperature over ROI.

    Chunked by month because a full year of hourly data exceeds the CDS
    per-request cost cap ("request is too large"). One month (~720 hourly
    steps, one variable, small area) is well within the limit.
    """
    n, w, s, e = area
    return {
        "variable": [TVAR],
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [n, w, s, e],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def hourly_to_daily(ds):
    """Return (tmax, tmin) daily DataArrays in degC from hourly ERA5-Land t2m.

    ERA5-Land 2m temperature is in Kelvin; we convert to Celsius. The time
    coordinate is `valid_time` on CDS-Beta, `time` on the legacy store.
    """
    tname = "valid_time" if "valid_time" in ds.coords else "time"
    var = "t2m" if "t2m" in ds.data_vars else next(iter(ds.data_vars))
    t_c = ds[var] - 273.15
    tmax = t_c.resample({tname: "1D"}).max()
    tmin = t_c.resample({tname: "1D"}).min()
    return tmax, tmin


def hargreaves_ra(lat_deg, doy):
    """Extraterrestrial radiation Ra (MJ/m^2/day) for latitude(s) and DOY.

    FAO-56 Eq. 21. `lat_deg` may be a scalar or array; `doy` is an int.
    """
    import numpy as np

    phi = np.deg2rad(lat_deg)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    decl = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)
    # Sunset hour angle; clip the cos argument to [-1, 1] for polar safety.
    cos_ws = np.clip(-np.tan(phi) * np.tan(decl), -1.0, 1.0)
    ws = np.arccos(cos_ws)
    gsc = 0.0820  # MJ/m^2/min
    return (
        (24.0 * 60.0 / np.pi)
        * gsc
        * dr
        * (ws * np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.sin(ws))
    )


def hargreaves_pet(tmax_c, tmin_c, ra_mj):
    """Daily Hargreaves PET (mm/day). Ra given in MJ/m^2/day (->mm via 0.408)."""
    import numpy as np

    tmean = (tmax_c + tmin_c) / 2.0
    tr = np.clip(tmax_c - tmin_c, 0.0, None)
    return 0.0023 * (0.408 * ra_mj) * (tmean + 17.8) * np.sqrt(tr)


def windowed_doy_stats(doy_sum, doy_sumsq, doy_count, window: int = DOY_WINDOW):
    """Pool raw per-DOY accumulators over a centered circular window -> mean,std.

    Implements PRD 4.4's "30-day rolling window centered on day-of-year". Arrays
    are (D, H, W); pooling is circular along axis 0 (Dec 31 wraps to Jan 1).
    """
    import numpy as np

    half = window // 2
    s = np.zeros_like(doy_sum)
    sq = np.zeros_like(doy_sumsq)
    c = np.zeros_like(doy_count)
    for shift in range(-half, half + 1):
        s += np.roll(doy_sum, shift, axis=0)
        sq += np.roll(doy_sumsq, shift, axis=0)
        c += np.roll(doy_count, shift, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(c > 0, s / c, np.nan)
        var = np.where(c > 0, sq / c - mean**2, np.nan)
    std = np.sqrt(np.clip(var, 0.0, None))
    return mean, std


def ckpt_path(out_dir: Path) -> Path:
    """Path of the resumability checkpoint."""
    return out_dir / ".ckpt_temp.npz"


def _load_checkpoint(path: Path):
    """Return (acc dict, set(done_years)) or empty accumulators."""
    import numpy as np

    if not path.exists():
        return {}, set()
    data = np.load(path, allow_pickle=True)
    acc = {k: data[k] for k in data.files if k != "done_years"}
    done = {int(y) for y in data["done_years"]} if "done_years" in data else set()
    return acc, done


def _save_checkpoint(path: Path, acc: dict, done_years: set[int]) -> None:
    """Atomically persist accumulators + completed years."""
    import numpy as np

    # np.savez appends '.npz' unless the name already ends in it, so the temp
    # name must end in '.npz' too or the subsequent replace() can't find it.
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, done_years=np.array(sorted(done_years)), **acc)
    tmp.replace(path)


@app.command()
def main(
    start_year: int = typer.Option(1991, "--start-year", help="Baseline first year."),
    end_year: int = typer.Option(2020, "--end-year", help="Baseline last year."),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", "-o", help="Override climatology output dir."
    ),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Resume from a checkpoint if present."
    ),
    keep_raw: bool = typer.Option(
        False, "--keep-raw", help="Keep each year's downloaded NetCDF after folding."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show the plan + first request; download nothing."
    ),
) -> None:
    """Build ERA5-Land Tmax per-DOY climatology + monthly PET over a year range."""
    if start_year > end_year:
        console.print("[red]ERROR[/red] --start-year must be <= --end-year.")
        raise typer.Exit(code=2)

    root = default_data_root()
    out_dir = output_dir or (root / "climatology")
    raw_dir = out_dir / "_era5land_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    years = list(range(start_year, end_year + 1))

    plan = Table(
        title="06c temperature climatology plan",
        show_header=False,
        title_style="bold cyan",
    )
    plan.add_column(style="dim")
    plan.add_column(style="bold")
    plan.add_row("Dataset", DATASET)
    plan.add_row("Variable", f"{TVAR} -> daily Tmax/Tmin (degC)")
    plan.add_row("Years", f"{start_year}-{end_year} ({len(years)})")
    plan.add_row("ROI area (N W S E)", str(DEFAULT_AREA))
    plan.add_row("DOY window", f"{DOY_WINDOW} days (centered)")
    plan.add_row("Output dir", str(out_dir))
    plan.add_row("Resume", "yes" if resume else "no")
    plan.add_row("Mode", "dry-run" if dry_run else "run")
    console.print(plan)
    console.print()

    if dry_run:
        console.print(f"First request ({start_year}-01):")
        console.print(build_request(start_year, 1, DEFAULT_AREA))
        console.print("\n[yellow]Dry run - nothing downloaded or written.[/yellow]")
        raise typer.Exit(code=0)

    try:
        import cdsapi
        import numpy as np
        import xarray as xr
    except ImportError as exc:
        console.print(f"[red]ERROR[/red] missing deps: {exc}")
        raise typer.Exit(code=1) from None

    acc: dict = {}
    done: set[int] = set()  # YYYYMM month-labels already folded
    if resume:
        acc, done = _load_checkpoint(ckpt_path(out_dir))
        if done:
            console.print(f"[dim]Resuming: {len(done)} month(s) already folded.[/dim]")

    raw_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    console.print(f"[green]CDS client ready: {client.url}[/green]\n")

    lat_grid = None  # captured on first fold, for Ra
    for year in years:
        for month in range(1, 13):
            label = year * 100 + month
            if label in done:
                continue
            raw_path = raw_dir / f"era5land_t2m_{year}-{month:02d}.nc"
            if not raw_path.exists():
                console.print(f"[bold]{year}-{month:02d}[/bold]: requesting (CDS will queue)...")
                try:
                    client.retrieve(
                        DATASET, build_request(year, month, DEFAULT_AREA), str(raw_path)
                    )
                except Exception as exc:
                    console.print(f"  [red]FAIL[/red] CDS retrieve: {exc}")
                    console.print("  [yellow]Re-run to resume from this month.[/yellow]")
                    raise typer.Exit(code=1) from None
            acc, lat_grid = _fold_month(raw_path, acc, lat_grid, year, month, np, xr)
            done.add(label)
            _save_checkpoint(ckpt_path(out_dir), acc, done)
            if not keep_raw:
                with contextlib.suppress(FileNotFoundError):
                    raw_path.unlink()
        console.print(f"[dim]  year {year} complete.[/dim]")

    _write_outputs(out_dir, acc, start_year, end_year, np, xr)


def _fold_month(raw_path: Path, acc: dict, lat_grid, year: int, month: int, np, xr):
    """Fold one month's hourly file into per-DOY Tmax accumulators + a PET slab."""
    with xr.open_dataset(raw_path) as ds:
        tmax, tmin = hourly_to_daily(ds)
        tmax = tmax.load()
        tmin = tmin.load()
        latname = "latitude" if "latitude" in tmax.coords else "lat"
        lats = tmax[latname].values
        if lat_grid is None:
            lat_grid = lats
        h, w = tmax.shape[-2], tmax.shape[-1]
        if "doy_sum" not in acc:
            acc["doy_sum"] = np.zeros((366, h, w), dtype="float64")
            acc["doy_sumsq"] = np.zeros((366, h, w), dtype="float64")
            acc["doy_count"] = np.zeros((366, h, w), dtype="float64")
            acc["pet_months"] = np.zeros((0, h, w), dtype="float32")
            acc["pet_labels"] = np.zeros((0,), dtype="int64")

        tname = "valid_time" if "valid_time" in tmax.coords else "time"
        doys = tmax[tname].dt.dayofyear.values
        lat2d = lats[:, None] * np.ones((h, w))
        tmax_v = tmax.values
        tmin_v = tmin.values
        valid = np.isfinite(tmax_v)

        pet_sum = np.zeros((h, w), dtype="float32")
        pet_valid = np.zeros((h, w), dtype=bool)
        for i, doy in enumerate(doys):
            k = int(doy) - 1
            v = np.where(valid[i], tmax_v[i], 0.0)
            acc["doy_sum"][k] += v
            acc["doy_sumsq"][k] += v * v
            acc["doy_count"][k] += valid[i]
            ra = hargreaves_ra(lat2d, int(doy))
            pet = hargreaves_pet(tmax_v[i], tmin_v[i], ra)
            finite = np.isfinite(pet)
            pet_sum += np.where(finite, pet, 0.0).astype("float32")
            pet_valid |= finite
        # Sea/never-valid pixels -> NaN (consistent with the count-masked Tmax).
        pet_month = np.where(pet_valid, pet_sum, np.nan).astype("float32")

        acc["pet_months"] = np.concatenate([acc["pet_months"], pet_month[None, :, :]], axis=0)
        acc["pet_labels"] = np.concatenate(
            [acc["pet_labels"], np.array([year * 100 + month], dtype="int64")]
        )
    return acc, lat_grid


def _write_outputs(out_dir: Path, acc: dict, start_year, end_year, np, xr):
    """Write the per-DOY Tmax mean/std and the monthly PET series."""
    if "doy_sum" not in acc:
        console.print("[yellow]No data accumulated; nothing written.[/yellow]")
        return
    mean, std = windowed_doy_stats(acc["doy_sum"], acc["doy_sumsq"], acc["doy_count"])
    doy = np.arange(1, 367)
    for name, arr in (("mean", mean), ("std", std)):
        da = xr.DataArray(arr.astype("float32"), dims=("doy", "lat", "lon"), coords={"doy": doy})
        dst = out_dir / f"tmax_doy_{name}.nc"
        da.to_dataset(name="tmax").to_netcdf(dst, engine="h5netcdf")

    # Monthly PET series, ordered by YYYYMM label.
    order = np.argsort(acc["pet_labels"])
    pet = acc["pet_months"][order]
    labels = acc["pet_labels"][order]
    # Land mask from the per-DOY counts (sea never accumulates a valid Tmax).
    # Applied here so any pre-fix slabs that stored sea as 0 are corrected too.
    land = acc["doy_count"].sum(axis=0) > 0
    pet = np.where(land[None, :, :], pet, np.nan).astype("float32")
    import pandas as pd

    times = pd.to_datetime([f"{lbl // 100}-{lbl % 100:02d}-01" for lbl in labels])
    da = xr.DataArray(pet, dims=("time", "lat", "lon"), coords={"time": times})
    da.to_dataset(name="pet").to_netcdf(
        out_dir / f"pet_monthly_{start_year}_{end_year}.nc", engine="h5netcdf"
    )
    console.print(
        f"\n[green]Wrote tmax_doy_mean/std.nc and pet_monthly " f"({pet.shape[0]} months).[/green]"
    )
    console.print("[green]Temperature climatology complete.[/green]")


if __name__ == "__main__":
    app()
