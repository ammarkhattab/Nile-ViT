"""Per-tile meteo time-series assembly for M4 (PRD section 4.3 / 5.1).

Each tile-time sample carries a ``(T_m=90, V=7)`` float32 meteo series: the 90
days ending at (and including) the tile's ``date``, 7 channels in fixed order,
sampled at the tile centre from the (coarse) ERA5-Land / CHIRPS fields.

Values are stored RAW. Per-channel z-scoring (section 5.1) is applied by the
loader using stats fit on the TRAIN split only (:func:`meteo_channel_stats`), so
normalisation never leaks val/test statistics -- the same anti-leakage discipline
as section 4.5. This module is pure (numpy only) and fully offline-testable.

Channel order (PRD section 4.3):
    era5_t2m, era5_swvl1, era5_e, era5_tp, chirps_p, chirts_tmax, chirts_tmin

Per B2 Decision 4, CHIRTS-ERA5 is not ROI-subsettable, so the chirts_tmax/tmin
channels are sourced from ERA5-Land daily max/min -- the same product used for the
heat-z label -- keeping the meteo input internally consistent. The PRD channel
names are kept; only the underlying product differs, and that is already tracked.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence

# Mirrors the (T_m=90, V=7) meteo block of TileSample in nilevit/schemas.py.
METEO_CHANNELS: tuple[str, ...] = (
    "era5_t2m",
    "era5_swvl1",
    "era5_e",
    "era5_tp",
    "chirps_p",
    "chirts_tmax",
    "chirts_tmin",
)
METEO_NUM_CHANNELS: int = len(METEO_CHANNELS)  # 7
METEO_WINDOW_DAYS: int = 90


def meteo_window_dates(end_date: dt.date, length: int = METEO_WINDOW_DAYS) -> list[dt.date]:
    """The ``length`` consecutive daily dates ending at (and including) ``end_date``."""
    if length < 1:
        raise ValueError("length must be >= 1")
    return [end_date - dt.timedelta(days=length - 1 - i) for i in range(length)]


def assemble_meteo_series(
    end_date: dt.date,
    daily_values: Mapping[dt.date, Mapping[str, float]],
    *,
    length: int = METEO_WINDOW_DAYS,
    channels: Sequence[str] = METEO_CHANNELS,
):
    """Build a ``(length, len(channels))`` float32 series ending at ``end_date``.

    ``daily_values`` maps a date to ``{channel: value}``. Missing dates or missing
    channels become NaN; the loader's Time2Vec/normalisation tolerates gaps and
    z-scoring skips NaN. Rows are ordered oldest-first, columns follow ``channels``.
    """
    import numpy as np

    dates = meteo_window_dates(end_date, length)
    out = np.full((length, len(channels)), np.nan, dtype=np.float32)
    for row_idx, day in enumerate(dates):
        row = daily_values.get(day)
        if not row:
            continue
        for col_idx, channel in enumerate(channels):
            value = row.get(channel)
            if value is not None:
                out[row_idx, col_idx] = value
    return out


def meteo_channel_stats(
    series: Sequence, *, channels: Sequence[str] = METEO_CHANNELS
) -> dict[str, dict[str, float]]:
    """Per-channel ``{mean, std}`` over a set of series, NaN-skipping.

    Fit on TRAIN samples only. A channel with no finite values (or zero variance)
    gets ``mean=0, std=1`` so :func:`zscore_meteo` is always a safe no-op there.
    """
    import numpy as np

    if len(series) == 0:
        return {ch: {"mean": 0.0, "std": 1.0} for ch in channels}

    stack = np.concatenate([np.asarray(item, dtype=np.float64) for item in series], axis=0)
    stats: dict[str, dict[str, float]] = {}
    for col_idx, channel in enumerate(channels):
        column = stack[:, col_idx]
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            stats[channel] = {"mean": 0.0, "std": 1.0}
            continue
        mean = float(finite.mean())
        std = float(finite.std())
        stats[channel] = {"mean": mean, "std": std if std > 0 else 1.0}
    return stats


def zscore_meteo(
    array,
    stats: Mapping[str, Mapping[str, float]],
    *,
    channels: Sequence[str] = METEO_CHANNELS,
):
    """Apply per-channel z-scoring with fitted ``stats``; NaN entries stay NaN."""
    import numpy as np

    out = np.asarray(array, dtype=np.float32).copy()
    for col_idx, channel in enumerate(channels):
        mean = stats[channel]["mean"]
        std = stats[channel]["std"] or 1.0
        out[:, col_idx] = (out[:, col_idx] - mean) / std
    return out


# --- ERA5-Land daily aggregation (grounded in the 2023 raw files) --------------
# The hourly fields were deaccumulated by cfgrib: t2m/swvl1 are state variables
# that vary hour-to-hour, and e/tp are signed per-hour fluxes (e is negative for
# evaporative loss). So fluxes are SUMMED to daily totals, state variables are
# MEAN-ed, and the heat extremes (chirts_*, per B2 Decision 4) are the daily
# max/min of t2m. Each meteo channel maps to its source ERA5-Land variable.
ERA5_SOURCE_VAR: dict[str, str] = {
    "era5_t2m": "t2m",
    "era5_swvl1": "swvl1",
    "era5_e": "e",
    "era5_tp": "tp",
    "chirts_tmax": "t2m",
    "chirts_tmin": "t2m",
}
ERA5_DAILY_AGG: dict[str, str] = {
    "era5_t2m": "mean",
    "era5_swvl1": "mean",
    "era5_e": "sum",
    "era5_tp": "sum",
    "chirts_tmax": "max",
    "chirts_tmin": "min",
}


def aggregate_hourly(values, method: str) -> float:
    """Reduce one day's hourly values to a daily scalar, NaN-skipping.

    ``method`` is one of ``mean``/``sum``/``max``/``min``. An all-NaN day (e.g. a
    sea cell) returns NaN. Used per ERA5-Land channel via :data:`ERA5_DAILY_AGG`.
    """
    import numpy as np

    finite = np.asarray(values, dtype="float64")
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    if method == "mean":
        return float(finite.mean())
    if method == "sum":
        return float(finite.sum())
    if method == "max":
        return float(finite.max())
    if method == "min":
        return float(finite.min())
    raise ValueError(f"unknown method {method!r}")
