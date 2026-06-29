"""Offline tests for the per-tile meteo core (PRD section 4.3 / 5.1)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

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
    zscore_meteo,
)


def test_channel_contract() -> None:
    # Order and count must match the TileSample (T_m=90, V=7) block exactly.
    assert METEO_NUM_CHANNELS == 7
    assert METEO_WINDOW_DAYS == 90
    assert METEO_CHANNELS == (
        "era5_t2m",
        "era5_swvl1",
        "era5_e",
        "era5_tp",
        "chirps_p",
        "chirts_tmax",
        "chirts_tmin",
    )


def test_window_dates_end_inclusive_and_ordered() -> None:
    end = dt.date(2023, 8, 15)
    dates = meteo_window_dates(end)
    assert len(dates) == 90
    assert dates[-1] == end  # ends at `date`
    assert dates[0] == end - dt.timedelta(days=89)  # 90 days ending at date
    assert dates == sorted(dates)  # oldest-first
    with pytest.raises(ValueError, match="length"):
        meteo_window_dates(end, length=0)


def test_assemble_shape_order_and_gaps() -> None:
    end = dt.date(2023, 8, 15)
    daily = {
        end: {"era5_t2m": 305.0, "chirps_p": 0.0},
        end - dt.timedelta(days=1): {"chirts_tmax": 311.0},
    }
    series = assemble_meteo_series(end, daily)
    assert series.shape == (90, 7)
    assert series.dtype == np.float32
    # Last row carries the end-date values in the right columns.
    assert series[-1, 0] == pytest.approx(305.0)  # era5_t2m
    assert series[-1, 4] == pytest.approx(0.0)  # chirps_p
    # Second-to-last row has chirts_tmax (col 5); the rest is NaN.
    assert series[-2, 5] == pytest.approx(311.0)
    assert np.isnan(series[-2, 0])
    # A day with no data is entirely NaN.
    assert np.all(np.isnan(series[0]))


def test_channel_stats_skip_nan_and_empty() -> None:
    a = np.full((90, 7), np.nan, dtype=np.float32)
    a[:, 0] = 10.0  # era5_t2m constant -> std clamped to 1.0
    a[0, 1] = 5.0
    a[1, 1] = 7.0  # era5_swvl1 has two finite values
    stats = meteo_channel_stats([a])
    assert stats["era5_t2m"]["mean"] == pytest.approx(10.0)
    assert stats["era5_t2m"]["std"] == pytest.approx(1.0)  # zero variance -> 1.0
    assert stats["era5_swvl1"]["mean"] == pytest.approx(6.0)
    # A channel that is all-NaN falls back to mean 0 / std 1.
    assert stats["era5_e"] == {"mean": 0.0, "std": 1.0}
    # Empty input -> all defaults.
    defaults = meteo_channel_stats([])
    assert defaults["chirps_p"] == {"mean": 0.0, "std": 1.0}


def test_zscore_roundtrips_to_zero_mean_unit_std() -> None:
    rng = np.random.default_rng(0)
    raw = rng.normal(290.0, 8.0, size=(90, 7)).astype(np.float32)
    stats = meteo_channel_stats([raw])
    z = zscore_meteo(raw, stats)
    assert z.shape == (90, 7)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-3)


def test_zscore_preserves_nan() -> None:
    raw = np.full((90, 7), np.nan, dtype=np.float32)
    raw[:, 0] = 1.0
    stats = meteo_channel_stats([raw])
    z = zscore_meteo(raw, stats)
    assert np.all(np.isnan(z[:, 1]))  # NaN column stays NaN


def test_daily_aggregation_rules() -> None:
    # Fluxes summed, state vars meaned, heat extremes from t2m max/min.
    assert ERA5_DAILY_AGG == {
        "era5_t2m": "mean",
        "era5_swvl1": "mean",
        "era5_e": "sum",
        "era5_tp": "sum",
        "chirts_tmax": "max",
        "chirts_tmin": "min",
    }
    assert ERA5_SOURCE_VAR["chirts_tmax"] == "t2m"
    assert ERA5_SOURCE_VAR["chirts_tmin"] == "t2m"


def test_aggregate_hourly_methods_and_nan() -> None:
    hourly = [294.0, 300.0, np.nan, 312.0]
    assert aggregate_hourly(hourly, "max") == pytest.approx(312.0)
    assert aggregate_hourly(hourly, "min") == pytest.approx(294.0)
    assert aggregate_hourly(hourly, "mean") == pytest.approx(302.0)  # NaN skipped
    # Signed fluxes sum correctly (evaporation is negative here).
    assert aggregate_hourly([-1e-4, 2e-5, -3e-5], "sum") == pytest.approx(-1.1e-4)
    # All-NaN (sea cell) -> NaN.
    assert np.isnan(aggregate_hourly([np.nan, np.nan], "sum"))
    with pytest.raises(ValueError, match="method"):
        aggregate_hourly([1.0], "median")
