"""Script 06d (climatology) - SPEI-3 water-balance baseline.

The final climatology step (PRD 4.4, "SPEI-like (Gaussian)" per
docs/B2_CLIMATOLOGY.md Decision 2). Fully local - no downloads. It merges the
two monthly references already on disk:

  precip_monthly_1991_2020.nc   (06b, CHIRPS v2, 0.05 grid)   mm/month
  pet_monthly_1991_2020.nc      (06c, ERA5-Land Hargreaves)   mm/month

into the per-pixel / per-calendar-month statistics the SPEI z-score needs:

  WB         = precip - PET                      (monthly water balance)
  WB_3       = 3-month rolling sum of WB         (SPEI-3 accumulation)
  spei_wb_mean[m], spei_wb_std[m]  = mean/std of WB_3 over 1991-2020 for the
                                     3-month window ending in calendar month m

Script 06 then forms SPEI(p, target-month m) = (WB_3_target - mean[m]) / std[m].

PET (0.1) is bilinearly regridded onto the finer CHIRPS 0.05 grid so the water
balance is computed at the precip resolution; the result stays on the CHIRPS
grid and Script 06 reprojects to the HLS grid on demand.

  uv run python scripts/data/06d_climatology_spei.py
  uv run python scripts/data/06d_climatology_spei.py --window 3 --dry-run

Outputs:
  data/climatology/spei_wb_mean.nc   spei_wb_std.nc      (month=12, lat, lon)
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
    help="SPEI-3 water-balance climatology combine (Script 06d).",
)
console = Console()


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


def standardize_names(ds):
    """Rename latitude/longitude -> lat/lon so both inputs share coord names."""
    ren = {}
    if "latitude" in ds.coords:
        ren["latitude"] = "lat"
    if "longitude" in ds.coords:
        ren["longitude"] = "lon"
    return ds.rename(ren) if ren else ds


def _ym(da):
    """Return the YYYYMM integer key array for a time-coorded DataArray."""
    return (da["time"].dt.year * 100 + da["time"].dt.month).values


def align_time(precip, pet):
    """Sort both by time, verify identical month sequences, share one time axis.

    Both inputs are full 1991-2020 monthly series. CHIRPS and ERA5-Land may stamp
    months on different days, so after confirming the YYYYMM sequences match we
    overwrite PET's time with precip's, letting arithmetic align positionally.
    """
    import numpy as np

    precip = precip.sortby("time")
    pet = pet.sortby("time")
    if not np.array_equal(_ym(precip), _ym(pet)):
        raise ValueError(
            "precip and PET month sequences differ; cannot align "
            f"({precip.sizes['time']} vs {pet.sizes['time']} months)."
        )
    pet = pet.assign_coords(time=precip["time"])
    return precip, pet


def regrid_to(da, target_lat, target_lon):
    """Bilinearly interpolate `da` onto the target lat/lon (ascending coords)."""
    da = da.sortby("lat").sortby("lon")
    return da.interp(lat=target_lat, lon=target_lon, method="linear")


def water_balance_stats(precip, pet, window: int = 3):
    """Return (mean, std) of the `window`-month rolling water balance per month.

    precip, pet : DataArrays (time, lat, lon) on the same grid and time axis.
    Output dims: (month=12, lat, lon).
    """
    wb = precip - pet
    wb3 = wb.rolling(time=window, min_periods=window).sum()
    grouped = wb3.groupby(wb3["time"].dt.month)
    mean = grouped.mean("time")
    std = grouped.std("time")
    return mean.rename(month="month"), std.rename(month="month")


@app.command()
def main(
    data_root: Path | None = typer.Option(None, "--data-root", help="Override the data/ root."),
    precip_file: str = typer.Option(
        "precip_monthly_1991_2020.nc", "--precip-file", help="06b precip stack."
    ),
    pet_file: str = typer.Option("pet_monthly_1991_2020.nc", "--pet-file", help="06c PET stack."),
    window: int = typer.Option(3, "--window", help="Accumulation window (months)."),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Rewrite outputs if they already exist."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the plan; read/write nothing."),
) -> None:
    """Combine monthly precip + PET into the SPEI-3 water-balance baseline."""
    root = data_root or default_data_root()
    clim = root / "climatology"
    p_path = clim / precip_file
    e_path = clim / pet_file
    mean_path = clim / "spei_wb_mean.nc"
    std_path = clim / "spei_wb_std.nc"

    plan = Table(
        title="06d SPEI water-balance plan",
        show_header=False,
        title_style="bold cyan",
    )
    plan.add_column(style="dim")
    plan.add_column(style="bold")
    plan.add_row("Precip (06b)", str(p_path))
    plan.add_row("PET (06c)", str(e_path))
    plan.add_row("Accumulation", f"{window}-month rolling sum")
    plan.add_row("Outputs", f"{mean_path.name}, {std_path.name}")
    plan.add_row("Mode", "dry-run" if dry_run else "run")
    console.print(plan)
    console.print()

    if dry_run:
        console.print("[yellow]Dry run - nothing read or written.[/yellow]")
        raise typer.Exit(code=0)

    if mean_path.exists() and std_path.exists() and not overwrite:
        console.print("[yellow]SKIP[/yellow] outputs exist (use --overwrite).")
        raise typer.Exit(code=0)

    for path in (p_path, e_path):
        if not path.exists():
            console.print(f"[red]ERROR[/red] missing input: {path}")
            raise typer.Exit(code=1)

    try:
        import numpy as np  # noqa: F401
        import xarray as xr
    except ImportError as exc:
        console.print(f"[red]ERROR[/red] missing deps: {exc}")
        raise typer.Exit(code=1) from None

    precip_ds = standardize_names(xr.open_dataset(p_path))
    pet_ds = standardize_names(xr.open_dataset(e_path))
    pvar = "precip" if "precip" in precip_ds else next(iter(precip_ds.data_vars))
    evar = "pet" if "pet" in pet_ds else next(iter(pet_ds.data_vars))
    precip = precip_ds[pvar]
    pet = pet_ds[evar]

    console.print(
        f"precip grid {precip.sizes.get('lat')}x{precip.sizes.get('lon')}, "
        f"PET grid {pet.sizes.get('lat')}x{pet.sizes.get('lon')} -> regridding PET"
    )
    precip, pet = align_time(precip, pet)
    pet_on_grid = regrid_to(pet, precip["lat"], precip["lon"])

    mean, std = water_balance_stats(precip, pet_on_grid, window=window)

    mean.to_dataset(name="wb_mean").to_netcdf(mean_path, engine="h5netcdf")
    std.to_dataset(name="wb_std").to_netcdf(std_path, engine="h5netcdf")

    # Quick sanity report.
    valid = std.where(std > 0)
    console.print(f"\n[cyan]{mean_path.name} / {std_path.name}[/cyan]")
    console.print(f"  dims        : {dict(mean.sizes)}")
    console.print(f"  WB mean Jan : {float(mean.sel(month=1).mean(skipna=True)):.1f} mm/3mo")
    console.print(f"  WB mean Jul : {float(mean.sel(month=7).mean(skipna=True)):.1f} mm/3mo")
    console.print(f"  WB std mean : {float(valid.mean(skipna=True)):.1f} mm/3mo")
    console.print("\n[green]SPEI water-balance baseline complete.[/green]")

    precip_ds.close()
    pet_ds.close()


if __name__ == "__main__":
    app()
