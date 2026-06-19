"""Deterministic compound-extreme label logic (PRD 4.4).

No model is involved: the label of a pixel is a fixed function of three
indicators computed against the 1991-2020 / 2001-2020 climatology baselines
(built by scripts/data/06a-06d). This module holds the *pure* functions so they
are unit-testable in isolation; scripts/labels/build_labels.py wires them to the
2023 rasters and writes the per-tile label maps (TileSample.label_path, PRD 4.3).

Indicators (all per-pixel):
  VCI = 100 * (NDVI - NDVI_min) / (NDVI_max - NDVI_min)        [MOD13Q1, 2001-2020]
  TCI = 100 * (LST_max - LST) / (LST_max - LST_min)            [MOD11A2, 2001-2020]
  VHI = 0.5 * VCI + 0.5 * TCI
  Tmax_z = (Tmax - mu_doy) / sigma_doy                          [ERA5-Land, 1991-2020]
  SPEI3  = (WB3 - mu_month) / sigma_month                       [Gaussian, 1991-2020]

Rule (PRD 4.4, precedence compound > heat > drought > none):
  3 compound      if SPEI3 < -1.0  AND  Tmax_z > 2.0  AND  VHI < 35
  2 heat-only     elif Tmax_z > 2.0
  1 drought-only  elif SPEI3 < -1.0
  0 none          otherwise

NaN handling: comparisons against NaN evaluate False, so a pixel missing an
indicator simply cannot satisfy the conditions that need it (e.g. a pixel with
no SPEI can still be heat-only, but never compound). This degrades safely
without special-casing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# --- PRD 4.4 thresholds (single source of truth; build_labels imports these) ---
SPEI3_THRESHOLD = -1.0  # SPEI-3 below this => drought signal
TMAX_Z_THRESHOLD = 2.0  # Tmax z-score above this => heat signal
VHI_THRESHOLD = 35.0  # VHI below this => vegetation stress

NONE, DROUGHT, HEAT, COMPOUND = 0, 1, 2, 3


def _ratio_index(num, den):
    """100 * num/den, guarding den<=0 (flat min==max) -> NaN, clipped to [0,100]."""
    import numpy as np

    num = np.asarray(num, dtype="float64")
    den = np.asarray(den, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = 100.0 * num / den
    out = np.where(den > 0, out, np.nan)
    return np.clip(out, 0.0, 100.0)


def vci(ndvi, ndvi_min, ndvi_max):
    """Vegetation Condition Index in [0,100] (NDVI vs its 2001-2020 min/max)."""
    import numpy as np

    return _ratio_index(
        np.asarray(ndvi) - np.asarray(ndvi_min),
        np.asarray(ndvi_max) - np.asarray(ndvi_min),
    )


def tci(lst, lst_min, lst_max):
    """Temperature Condition Index in [0,100] (inverted: hotter LST -> lower TCI)."""
    import numpy as np

    return _ratio_index(
        np.asarray(lst_max) - np.asarray(lst), np.asarray(lst_max) - np.asarray(lst_min)
    )


def vhi(vci_arr, tci_arr):
    """Vegetation Health Index = 0.5*VCI + 0.5*TCI."""
    import numpy as np

    return 0.5 * np.asarray(vci_arr, dtype="float64") + 0.5 * np.asarray(tci_arr, dtype="float64")


def zscore(value, mu, sigma):
    """(value - mu)/sigma, guarding sigma<=0 -> NaN. Used for Tmax_z and SPEI3."""
    import numpy as np

    value = np.asarray(value, dtype="float64")
    mu = np.asarray(mu, dtype="float64")
    sigma = np.asarray(sigma, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (value - mu) / sigma
    return np.where(sigma > 0, out, np.nan)


def classify(spei3, tmax_z, vhi_arr) -> np.ndarray:
    """Apply the PRD 4.4 4-class rule to broadcastable indicator arrays.

    Returns a uint8 array in {0,1,2,3} with precedence compound>heat>drought>none.
    """
    import numpy as np

    spei3 = np.asarray(spei3, dtype="float64")
    tmax_z = np.asarray(tmax_z, dtype="float64")
    vhi_arr = np.asarray(vhi_arr, dtype="float64")

    is_drought = spei3 < SPEI3_THRESHOLD
    is_heat = tmax_z > TMAX_Z_THRESHOLD
    is_veg_stress = vhi_arr < VHI_THRESHOLD
    is_compound = is_drought & is_heat & is_veg_stress

    shape = np.broadcast_shapes(spei3.shape, tmax_z.shape, vhi_arr.shape)
    out = np.zeros(shape, dtype="uint8")
    # Layer in precedence order; later writes win where their condition holds.
    out = np.where(is_drought, DROUGHT, out)
    out = np.where(is_heat, HEAT, out)  # heat outranks drought-only (PRD elif)
    out = np.where(is_compound, COMPOUND, out)
    return out.astype("uint8")


def label_from_rasters(
    ndvi,
    ndvi_min,
    ndvi_max,
    lst,
    lst_min,
    lst_max,
    tmax,
    tmax_mu,
    tmax_sigma,
    wb3,
    wb_mu,
    wb_sigma,
) -> np.ndarray:
    """Convenience: compute all three indicators from rasters, then classify.

    All inputs are broadcastable arrays already on a common grid. Returns the
    uint8 label array (PRD 4.4).
    """
    v = vhi(vci(ndvi, ndvi_min, ndvi_max), tci(lst, lst_min, lst_max))
    tz = zscore(tmax, tmax_mu, tmax_sigma)
    sp = zscore(wb3, wb_mu, wb_sigma)
    return classify(sp, tz, v)
