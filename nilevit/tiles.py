"""Per-tile label packaging helpers for M4 (PRD section 4.3).

The label field is the ROI-grid raster M3 already produced from
``nilevit.labels.label_from_rasters`` (common 0.05deg grid). M4 packages it onto
each 224x224 tile by selecting the MODIS composite active at the tile's date and
resampling that ROI label onto the tile grid (nearest-neighbour, categorical),
filling no-data with 255. This module also holds the pure label statistics used
to recompute the focal-loss class weights from the real distribution.

The pure functions (``label_histogram``, ``valid_fraction``,
``class_weights_from_counts``, ``label_date_for``) have no geospatial
dependencies and are fully offline-testable. ``resample_label_to_tile`` lazily
imports rioxarray/rasterio inside the function, per the project convention.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

from nilevit.schemas import LABEL_NODATA, NUM_CLASSES

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    import xarray as xr


# --- pure label statistics -----------------------------------------------------
def label_histogram(
    array: np.ndarray, *, num_classes: int = NUM_CLASSES, nodata: int = LABEL_NODATA
) -> dict[int, int]:
    """Count pixels per class 0..num_classes-1, ignoring ``nodata``."""
    import numpy as np

    flat = np.asarray(array).ravel()
    valid = flat[flat != nodata]
    counts = np.bincount(valid.astype(np.int64), minlength=num_classes)
    return {cls: int(counts[cls]) for cls in range(num_classes)}


def valid_fraction(array: np.ndarray, *, nodata: int = LABEL_NODATA) -> float:
    """Fraction of pixels that are not ``nodata`` (the schema's ``valid_pct``)."""
    import numpy as np

    flat = np.asarray(array).ravel()
    if flat.size == 0:
        return 0.0
    return float((flat != nodata).sum() / flat.size)


def class_weights_from_counts(
    counts: Mapping[int, int] | Sequence[int],
    *,
    scheme: str = "median_freq",
    num_classes: int = NUM_CLASSES,
) -> list[float]:
    """Recompute focal-loss class weights from a label histogram.

    ``median_freq`` (median-frequency balancing, the seg default):
    ``w_c = median(freq) / freq_c``. ``inverse``: ``w_c = total / (K * count_c)``.
    Classes absent from the data get weight 0.0. Replaces the PRD placeholder
    ``[0.1, 1.0, 1.0, 3.0]`` once the real distribution is known.
    """
    import numpy as np

    if isinstance(counts, Mapping):
        vector = np.array([counts.get(cls, 0) for cls in range(num_classes)], dtype=np.float64)
    else:
        vector = np.asarray(counts, dtype=np.float64)

    total = vector.sum()
    if total <= 0:
        return [0.0] * num_classes

    present = vector > 0
    if scheme == "inverse":
        weights = np.zeros(num_classes, dtype=np.float64)
        weights[present] = total / (num_classes * vector[present])
        return [float(w) for w in weights]

    if scheme == "median_freq":
        freq = vector / total
        median = float(np.median(freq[present]))
        weights = np.zeros(num_classes, dtype=np.float64)
        weights[present] = median / freq[present]
        return [float(w) for w in weights]

    raise ValueError(f"unknown scheme {scheme!r}; use 'median_freq' or 'inverse'")


def aggregate_counts(
    per_tile: Iterable[Mapping[int, int]], *, num_classes: int = NUM_CLASSES
) -> dict[int, int]:
    """Sum per-tile histograms into one dataset-wide histogram."""
    totals = dict.fromkeys(range(num_classes), 0)
    for hist in per_tile:
        for cls, count in hist.items():
            if cls in totals:
                totals[cls] += count
    return totals


# --- date selection ------------------------------------------------------------
def label_date_for(tile_date: dt.date, label_dates: Sequence[dt.date]) -> dt.date:
    """ROI label date active at ``tile_date`` (latest composite start <= date).

    MODIS composites refresh every 16 days, so the label valid at a tile's
    acquisition date is the most recent composite on or before it. Falls back to
    the earliest available date if the tile predates all label dates.
    """
    if not label_dates:
        raise ValueError("label_dates is empty")
    ordered = sorted(label_dates)
    on_or_before = [d for d in ordered if d <= tile_date]
    return on_or_before[-1] if on_or_before else ordered[0]


# --- geospatial resampler (design A: resample the ROI label onto the tile) -----
def mgrs_to_epsg(mgrs_tile: str) -> int:
    """EPSG code of the UTM CRS for an MGRS tile id (e.g. ``"T36RUU" -> 32636``).

    MGRS latitude bands C..M are southern, N..X northern, so the band letter
    selects the 327xx (south) vs 326xx (north) UTM family for the zone number.
    """
    tile = mgrs_tile.upper().removeprefix("T")
    cut = 0
    while cut < len(tile) and tile[cut].isdigit():
        cut += 1
    zone = int(tile[:cut])
    band = tile[cut]
    if not 1 <= zone <= 60:
        raise ValueError(f"invalid UTM zone {zone} in MGRS tile {mgrs_tile!r}")
    northern = band >= "N"
    return (32600 if northern else 32700) + zone


def tile_grid_template(
    center_lon: float,
    center_lat: float,
    mgrs_tile: str,
    *,
    size: int = 224,
    res: float = 30.0,
) -> xr.DataArray:
    """Empty destination grid for a tile, reconstructed from its centre point.

    The 05b cube stores no per-tile transform, but tiles are a regular ``res``-m
    north-up grid in the MGRS tile's UTM CRS, and ``center_lon/lat`` is the window
    centroid. So the grid is the centre projected to UTM, expanded by
    ``size/2 * res`` to the upper-left. Returns a CRS- and transform-aware
    DataArray to pass as the ``tile_template`` of :func:`resample_label_to_tile`.
    """
    import numpy as np
    import rioxarray  # noqa: F401  (registers the .rio accessor)
    import xarray as xr
    from affine import Affine
    from pyproj import Transformer

    epsg = mgrs_to_epsg(mgrs_tile)
    transformer = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    center_x, center_y = transformer.transform(center_lon, center_lat)

    half = (size / 2.0) * res
    x_ul, y_ul = center_x - half, center_y + half
    transform = Affine(res, 0.0, x_ul, 0.0, -res, y_ul)
    xs = x_ul + (np.arange(size) + 0.5) * res
    ys = y_ul - (np.arange(size) + 0.5) * res

    template = xr.DataArray(
        np.zeros((size, size), dtype="float32"),
        coords={"y": ys, "x": xs},
        dims=("y", "x"),
    )
    return template.rio.write_crs(epsg).rio.write_transform(transform)


def resample_label_to_tile(
    roi_label: xr.DataArray,
    tile_template: xr.DataArray,
    *,
    nodata: int = LABEL_NODATA,
) -> xr.DataArray:
    """Resample an ROI-grid label onto a tile grid (nearest, categorical).

    ``roi_label`` is the 0.05deg label raster for the tile's date;
    ``tile_template`` is any band of the destination 224x224 tile (defines the
    grid/CRS/transform). Uses ``rioxarray.reproject_match`` with nearest
    resampling so class codes are never interpolated, and writes ``nodata`` (255)
    for pixels with no source coverage. Returns a uint8 DataArray; take
    ``.values`` for the array written to the Zarr ``label_path``.
    """
    import rioxarray  # noqa: F401  (registers the .rio accessor)
    from rasterio.enums import Resampling

    src = roi_label
    if src.rio.nodata is None:
        src = src.rio.write_nodata(nodata)

    matched = src.rio.reproject_match(tile_template, resampling=Resampling.nearest, nodata=nodata)
    return matched.fillna(nodata).astype("uint8")
