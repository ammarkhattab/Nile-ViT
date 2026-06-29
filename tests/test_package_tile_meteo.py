"""End-to-end offline test for scripts/data/package_tile_meteo.py.

Builds a tiny synthetic 05b cube, a labelled index, and synthetic monthly
ERA5-Land (hourly) + CHIRPS (daily) files spanning the 90-day window, runs the
meteo packaging CLI, and checks the meteo store, enriched index, and train-fit
z-score stats.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "data" / "package_tile_meteo.py"

SAMPLES = [
    ("T36RUU_2023-08-01_r00c00", 31.0, 30.7, "train"),
    ("T36RUU_2023-08-01_r00c01", 31.1, 30.7, "buffer"),
]


def _build_inputs(tmp_path: Path):
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import xarray as xr
    from shapely.geometry import Point

    # Cube with the same 2 samples (image content irrelevant here).
    cube = xr.Dataset(
        {"image": (("sample", "band", "y", "x"), np.zeros((2, 6, 4, 4), "uint16"))},
        coords={"sample": np.array([s[0] for s in SAMPLES])},
    )
    tiles_zarr = tmp_path / "tiles_T36RUU_2023.zarr"
    cube.to_zarr(tiles_zarr, mode="w", consolidated=False)

    # Labelled index.
    records = [
        {
            "sample_id": sid,
            "mgrs_tile": "T36RUU",
            "date": "2023-08-01",
            "center_lon": lon,
            "center_lat": lat,
            "region": "delta",
            "split": split,
            "geometry": Point(lon, lat),
        }
        for sid, lon, lat, split in SAMPLES
    ]
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    labeled_parquet = tmp_path / "tiles_T36RUU_2023_labeled.parquet"
    gdf.to_parquet(labeled_parquet)

    # The 90-day window ending 2023-08-01 spans May, June, July, August.
    era5_dir = tmp_path / "era5"
    chirps_dir = tmp_path / "chirps"
    era5_dir.mkdir()
    chirps_dir.mkdir()
    lat_axis = np.array([30.6, 30.7, 30.8])
    lon_axis = np.array([30.9, 31.0, 31.1])

    for year, month in [(2023, 5), (2023, 6), (2023, 7), (2023, 8)]:
        days = pd.Period(f"{year}-{month:02d}").days_in_month
        hours = pd.date_range(f"{year}-{month:02d}-01", periods=days * 24, freq="h")
        shape = (len(hours), len(lat_axis), len(lon_axis))
        rng = np.random.default_rng(month)
        era5 = xr.Dataset(
            {
                "t2m": (
                    ("valid_time", "latitude", "longitude"),
                    300 + rng.normal(0, 5, shape),
                ),
                "swvl1": (
                    ("valid_time", "latitude", "longitude"),
                    rng.uniform(0.1, 0.2, shape),
                ),
                "e": (
                    ("valid_time", "latitude", "longitude"),
                    rng.normal(-1e-4, 1e-5, shape),
                ),
                "tp": (
                    ("valid_time", "latitude", "longitude"),
                    rng.uniform(0, 1e-5, shape),
                ),
            },
            coords={"valid_time": hours, "latitude": lat_axis, "longitude": lon_axis},
        )
        era5.to_netcdf(era5_dir / f"era5_land_{year}-{month:02d}_d01-d{days:02d}.nc")

        times = pd.date_range(f"{year}-{month:02d}-01", periods=days, freq="D")
        chirps = xr.Dataset(
            {
                "precip": (
                    ("time", "latitude", "longitude"),
                    rng.uniform(0, 3, (days, 3, 3)),
                )
            },
            coords={"time": times, "latitude": lat_axis, "longitude": lon_axis},
        )
        chirps.to_netcdf(chirps_dir / f"chirps-v3.0.{year}.{month:02d}.days_p05.nile-em.nc")

    return tiles_zarr, labeled_parquet, era5_dir, chirps_dir


def _load_cli():
    spec = importlib.util.spec_from_file_location("package_tile_meteo", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not SCRIPT_PATH.exists(), reason="meteo packaging script not found")
def test_package_tile_meteo_end_to_end(tmp_path: Path) -> None:
    import numpy as np
    import xarray as xr
    from typer.testing import CliRunner

    tiles_zarr, labeled_parquet, era5_dir, chirps_dir = _build_inputs(tmp_path)
    norm_out = tmp_path / "meteo_norm_v1.json"
    module = _load_cli()

    result = CliRunner().invoke(
        module.app,
        [
            "--tiles-zarr",
            str(tiles_zarr),
            "--labeled-parquet",
            str(labeled_parquet),
            "--era5-dir",
            str(era5_dir),
            "--chirps-dir",
            str(chirps_dir),
            "--norm-out",
            str(norm_out),
        ],
    )
    assert result.exit_code == 0, result.output

    # Meteo store: (sample, t=90, channel=7), float32, fully covered (no sea cells).
    meteo_zarr = tiles_zarr.with_name(f"{tiles_zarr.stem}_meteo.zarr")
    meteo_ds = xr.open_zarr(meteo_zarr, consolidated=False)
    assert meteo_ds["meteo"].dtype == "float32"
    assert meteo_ds["meteo"].sizes == {"sample": 2, "t": 90, "channel": 7}
    assert list(meteo_ds["channel"].values) == list(module.METEO_CHANNELS)
    assert bool(np.isfinite(meteo_ds["meteo"].values).all())

    # Index gained meteo_path; stats fit on the single TRAIN tile only.
    import geopandas as gpd

    enriched = gpd.read_parquet(labeled_parquet)
    assert {"meteo_path", "meteo_coverage"}.issubset(enriched.columns)

    norm = json.loads(norm_out.read_text())
    assert norm["fit_split"] == "train"
    assert norm["n_train_samples"] == 1  # only the train tile, not the buffer tile
    assert set(norm["stats"]) == set(module.METEO_CHANNELS)
    # t2m mean is in a plausible Kelvin range for the synthetic data.
    assert 280.0 < norm["stats"]["era5_t2m"]["mean"] < 320.0
