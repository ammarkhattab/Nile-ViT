# ruff: noqa: B008
"""Package per-tile 90x7 meteo series into the tiled dataset (PRD section 4.3 / 5.1).

For each tile-time sample in the labelled 05b index, sample the 7 meteo channels
at the tile centre over the 90 days ending at the tile's date, assemble the
(90, 7) float32 series (nilevit.meteo), store it RAW in a Zarr `meteo
(sample, t, channel)` variable, and enrich the parquet with `meteo_path`. Then fit
per-channel z-score stats on the TRAIN split only (section 5.1, no leakage) and
write them to configs/meteo_norm_v1.json for the loader to apply.

Daily aggregation of the hourly ERA5-Land fields (grounded in the raw files):
  era5_t2m=mean, era5_swvl1=mean, era5_e=sum, era5_tp=sum (signed per-hour fluxes),
  chirts_tmax=max(t2m), chirts_tmin=min(t2m) (B2 Decision 4). chirps_p is the
  CHIRPS daily `precip` directly. Channels are sampled at the nearest land cell;
  sea/missing cells stay NaN (z-score and Time2Vec tolerate gaps).

Usage:
    uv run python scripts/data/package_tile_meteo.py \
        --tiles-zarr data/interim/tiles_T36RUU_2023.zarr \
        --labeled-parquet data/interim/tiles_T36RUU_2023_labeled.parquet
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
from collections import Counter
from pathlib import Path

import typer

from nilevit.meteo import (
    ERA5_DAILY_AGG,
    ERA5_SOURCE_VAR,
    METEO_CHANNELS,
    METEO_NUM_CHANNELS,
    METEO_WINDOW_DAYS,
    aggregate_hourly,
    assemble_meteo_series,
    meteo_channel_stats,
    meteo_window_dates,
)

with contextlib.suppress(ImportError):
    import nilevit  # noqa: F401

app = typer.Typer(add_completion=False, help=__doc__)

TRAIN_SPLIT = "train"


def _months_spanning(dates: list[dt.date]) -> set[tuple[int, int]]:
    """The (year, month) pairs the 90-day windows of all tile dates touch."""
    months: set[tuple[int, int]] = set()
    for end in dates:
        for day in meteo_window_dates(end):
            months.add((day.year, day.month))
    return months


def _load_era5(era5_dir: Path, months: set[tuple[int, int]]):
    """Concatenate the monthly ERA5-Land files covering ``months`` on valid_time."""
    import xarray as xr

    paths = []
    for year, month in sorted(months):
        matches = sorted(era5_dir.glob(f"era5_land_{year}-{month:02d}_*.nc"))
        paths.extend(matches)
    if not paths:
        return None
    parts = [xr.open_dataset(p) for p in paths]
    return xr.concat(parts, dim="valid_time").sortby("valid_time")


def _load_chirps(chirps_dir: Path, months: set[tuple[int, int]]):
    """Concatenate the monthly CHIRPS files covering ``months`` on time."""
    import xarray as xr

    paths = []
    for year, month in sorted(months):
        matches = sorted(chirps_dir.glob(f"chirps-v3.0.{year}.{month:02d}.*.nc"))
        paths.extend(matches)
    if not paths:
        return None
    parts = [xr.open_dataset(p) for p in paths]
    return xr.concat(parts, dim="time").sortby("time")


def _daily_era5_at_point(era5, lon: float, lat: float) -> dict[dt.date, dict[str, float]]:
    """Daily-aggregated ERA5 channels at the nearest cell, keyed by date."""
    import numpy as np
    import pandas as pd

    point = era5.sel(latitude=lat, longitude=lon, method="nearest")
    times = pd.to_datetime(point["valid_time"].values)
    days = np.array([t.date() for t in times])

    out: dict[dt.date, dict[str, float]] = {}
    source_cache = {var: np.asarray(point[var].values) for var in {"t2m", "swvl1", "e", "tp"}}
    for day in np.unique(days):
        mask = days == day
        row: dict[str, float] = {}
        for channel in ERA5_DAILY_AGG:
            hourly = source_cache[ERA5_SOURCE_VAR[channel]][mask]
            row[channel] = aggregate_hourly(hourly, ERA5_DAILY_AGG[channel])
        out[day] = row
    return out


def _daily_chirps_at_point(chirps, lon: float, lat: float) -> dict[dt.date, float]:
    """CHIRPS daily precip at the nearest cell, keyed by date."""
    import numpy as np
    import pandas as pd

    point = chirps.sel(latitude=lat, longitude=lon, method="nearest")
    times = pd.to_datetime(point["time"].values)
    values = np.asarray(point["precip"].values, dtype="float64")
    return {t.date(): float(v) for t, v in zip(times, values, strict=True)}


@app.command()
def main(
    tiles_zarr: Path = typer.Option(..., help="05b cube store (sample order)."),
    labeled_parquet: Path = typer.Option(
        ..., help="Labelled index from package_tile_labels.py (sample_id + split)."
    ),
    era5_dir: Path = typer.Option(
        Path("data/raw/era5/2023"), help="Directory of monthly ERA5-Land .nc files."
    ),
    chirps_dir: Path = typer.Option(
        Path("data/raw/chirps/2023"), help="Directory of monthly CHIRPS .nc files."
    ),
    meteo_zarr: Path | None = typer.Option(
        None, help="Output meteo store (default: <stem>_meteo.zarr)."
    ),
    out_parquet: Path | None = typer.Option(
        None, help="Index with meteo_path added (default: overwrite labelled parquet)."
    ),
    norm_out: Path = typer.Option(
        Path("configs/meteo_norm_v1.json"), help="Train-fit z-score stats path."
    ),
) -> None:
    """Sample, assemble, and store the per-tile meteo series; fit z-score stats."""
    import geopandas as gpd
    import numpy as np
    import xarray as xr

    meteo_zarr = meteo_zarr or tiles_zarr.with_name(f"{tiles_zarr.stem}_meteo.zarr")
    out_parquet = out_parquet or labeled_parquet

    gdf = gpd.read_parquet(labeled_parquet)
    dataset = xr.open_zarr(tiles_zarr, consolidated=False)
    zarr_order = [str(s) for s in dataset["sample"].values]
    # Follow the labelled index (already filtered to in-ROI tiles).
    sample_order = [s for s in zarr_order if s in set(gdf["sample_id"])]
    indexed = gdf.set_index("sample_id")

    tile_dates = [dt.date.fromisoformat(str(indexed.loc[s, "date"])) for s in sample_order]
    months = _months_spanning(tile_dates)
    era5 = _load_era5(era5_dir, months)
    chirps = _load_chirps(chirps_dir, months)
    if era5 is None or chirps is None:
        raise typer.BadParameter("missing ERA5 or CHIRPS files for the date window")

    # Cache per-centre daily series (many tiles share a coarse meteo cell).
    era5_cache: dict[tuple[float, float], dict] = {}
    chirps_cache: dict[tuple[float, float], dict] = {}

    n = len(sample_order)
    meteo = np.full((n, METEO_WINDOW_DAYS, METEO_NUM_CHANNELS), np.nan, dtype=np.float32)
    coverage: list[float] = []

    for index, sample_id in enumerate(sample_order):
        row = indexed.loc[sample_id]
        lon = round(float(row["center_lon"]), 2)
        lat = round(float(row["center_lat"]), 2)
        end = dt.date.fromisoformat(str(row["date"]))

        if (lon, lat) not in era5_cache:
            era5_cache[(lon, lat)] = _daily_era5_at_point(era5, lon, lat)
            chirps_cache[(lon, lat)] = _daily_chirps_at_point(chirps, lon, lat)
        era5_daily = era5_cache[(lon, lat)]
        chirps_daily = chirps_cache[(lon, lat)]

        merged: dict[dt.date, dict[str, float]] = {}
        for day in meteo_window_dates(end):
            entry = dict(era5_daily.get(day, {}))
            if day in chirps_daily:
                entry["chirps_p"] = chirps_daily[day]
            if entry:
                merged[day] = entry

        series = assemble_meteo_series(end, merged)
        meteo[index] = series
        coverage.append(float(np.isfinite(series).mean()))

    # --- write the meteo store (mode="w" -> idempotent) ---
    import shutil

    if meteo_zarr.exists():
        shutil.rmtree(meteo_zarr)
    xr.Dataset(
        {"meteo": (("sample", "t", "channel"), meteo)},
        coords={
            "sample": np.array(sample_order),
            "channel": np.array(METEO_CHANNELS),
        },
    ).to_zarr(meteo_zarr, mode="w", consolidated=False)

    # --- enrich the index with meteo_path ---
    enriched = indexed.loc[sample_order].reset_index()
    enriched["meteo_path"] = f"{meteo_zarr.name}::meteo"
    enriched["meteo_coverage"] = coverage
    enriched.to_parquet(out_parquet)

    # --- fit z-score stats on the TRAIN split only (section 5.1) ---
    train_series = [
        meteo[i]
        for i, sample_id in enumerate(sample_order)
        if str(indexed.loc[sample_id, "split"]) == TRAIN_SPLIT
    ]
    stats = meteo_channel_stats(train_series)
    norm_out.parent.mkdir(parents=True, exist_ok=True)
    norm_out.write_text(
        json.dumps(
            {
                "source_zarr": str(tiles_zarr),
                "fit_split": TRAIN_SPLIT,
                "n_train_samples": len(train_series),
                "channels": list(METEO_CHANNELS),
                "stats": stats,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # --- summary ---
    typer.echo(f"wrote {meteo_zarr}  (meteo: {n}x{METEO_WINDOW_DAYS}x{METEO_NUM_CHANNELS})")
    typer.echo(f"wrote {out_parquet}  (+meteo_path, +meteo_coverage)")
    typer.echo(f"wrote {norm_out}  (z-score fit on {len(train_series)} train tiles)")
    split_counts = dict(Counter(str(indexed.loc[s, "split"]) for s in sample_order))
    typer.echo(f"  split distribution: {split_counts}")
    typer.echo(f"  mean finite coverage: {float(np.mean(coverage)):.3f}")
    typer.echo("  train z-score means:")
    for channel in METEO_CHANNELS:
        mean = stats[channel]["mean"]
        std = stats[channel]["std"]
        typer.echo(f"    {channel:12s} mean={mean:.5g}  std={std:.5g}")


if __name__ == "__main__":
    app()
