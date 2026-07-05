"""Label-driven patch selection (bounds the dataset; preserves the rare classes).

Exhaustive tiling of 81 land tiles x 8 warm seasons yields ~1M+ patches / multiple
TB - ~100-200x the PRD's ~6,000-sample target (§6.1 embedding budget) and far past
the §10.3 80 GB cap. The PRD fixes the *size* but not the *selection rule*; this
module is that rule, decided deliberately and documented in docs/M4_PACKAGING.md.

Rule: **keep every patch that contains a rare-class pixel (compound/heat/drought);
subsample the class-0 background at a tunable ratio.** This preserves the rare
compound class by construction (so §4.4 prevalence stays in [0.5%, 8%]) and the
background ratio is the single knob that tunes total size to the ~6,000 target.

Selection runs on the coarse M3 label rasters *before* HLS streaming, so only kept
patches are ever tiled - no multi-TB intermediate. The keep-decision is a
deterministic hash of (sample_id, seed), so it is order-independent and regenerates
identically from public labels: the keep-list publishes as part of the dataset.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

EVENT_CLASSES: tuple[int, ...] = (1, 2, 3)  # drought, heat, compound
LABEL_NODATA = 255
DEFAULT_SEED = 20260519  # PRD §6.2 seed

# Per-class keep probability. Compound (3) is the rare §4.4/R7 target -> kept in
# full; drought (1) is broad-area and dominates raw counts -> subsampled hard; heat
# (2) is moderately common; none (0) is background. Tuned so 81 tiles x ~19
# composite dates x 8 yr lands under the §10.3 80 GB embedding cap (~100k samples).
DEFAULT_KEEP_PROB: dict[int, float] = {0: 0.015, 1: 0.04, 2: 0.4, 3: 1.0}


def patch_class(values: Any, nodata: int = LABEL_NODATA) -> int | None:
    """Representative class of a patch: compound>heat>drought>none.

    Returns the highest-priority event class present, else 0, or None if the patch
    is entirely nodata (sea / no-data — not a dataset member).
    """
    import numpy as np

    arr = np.asarray(values).ravel()
    valid = arr[arr != nodata]
    if valid.size == 0:
        return None
    for cls in (3, 2, 1):  # compound > heat > drought priority
        if (valid == cls).any():
            return cls
    return 0


def label_window(patch_bbox: Sequence[float], label_da: Any, nodata: int = LABEL_NODATA) -> Any:
    """Label pixel values within a patch's geographic bbox (order-agnostic).

    Falls back to the single nearest pixel if the bbox spans no cell centre (patch
    smaller than the ~0.05° label grid), so every patch gets a class.
    """
    import numpy as np

    west, south, east, north = patch_bbox
    xs = np.asarray(label_da["x"].values)
    ys = np.asarray(label_da["y"].values)
    values = np.asarray(label_da.values)

    x_in = (xs >= west) & (xs <= east)
    y_in = (ys >= south) & (ys <= north)
    if x_in.any() and y_in.any():
        return values[np.ix_(y_in, x_in)]
    # Patch smaller than a label cell -> nearest single pixel at the centre.
    cx, cy = (west + east) / 2.0, (south + north) / 2.0
    xi = int(np.abs(xs - cx).argmin())
    yi = int(np.abs(ys - cy).argmin())
    return values[yi, xi]


def _unit_hash(sample_id: str, seed: int) -> float:
    """Deterministic value in [0, 1) from (sample_id, seed) — order-independent."""
    digest = hashlib.blake2b(f"{seed}:{sample_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def keep_by_prob(sample_id: str, prob: float, seed: int = DEFAULT_SEED) -> bool:
    """Seeded, reproducible keep-decision at a given probability (order-independent)."""
    if prob >= 1.0:
        return True
    if prob <= 0.0:
        return False
    return _unit_hash(sample_id, seed) < prob


def decide_patch(
    sample_id: str,
    patch_bbox: Sequence[float],
    label_da: Any,
    *,
    keep_prob: dict[int, float] | None = None,
    seed: int = DEFAULT_SEED,
    nodata: int = LABEL_NODATA,
) -> tuple[int | None, bool]:
    """Return ``(patch_class, keep)`` for one patch under per-class keep rates.

    The patch class is compound>heat>drought>none over its label window; the keep
    decision draws against ``keep_prob[class]`` (compound defaults to 1.0). All-nodata
    patches (sea) are dropped.
    """
    if keep_prob is None:
        keep_prob = DEFAULT_KEEP_PROB
    cls = patch_class(label_window(patch_bbox, label_da, nodata), nodata)
    if cls is None:
        return None, False
    return cls, keep_by_prob(sample_id, keep_prob.get(cls, 0.0), seed)


def grid_cells(
    tile_bbox: Sequence[float], n: int = 16
) -> list[tuple[int, int, tuple[float, float, float, float]]]:
    """Approximate 05b's nxn patch grid as geographic cells (for the size preview).

    05b makes 16x16 = 256 patches per HLS tile (3660px / 224). This grids the tile's
    geographic bbox into the same layout so the label-only preview counts match.
    """
    west, south, east, north = tile_bbox
    dx, dy = (east - west) / n, (north - south) / n
    cells: list[tuple[int, int, tuple[float, float, float, float]]] = []
    for r in range(n):
        for c in range(n):
            cw = west + c * dx
            cs = south + r * dy
            cells.append((r, c, (cw, cs, cw + dx, cs + dy)))
    return cells


def selection_report(
    tile: str,
    date_iso: str,
    tile_bbox: Sequence[float],
    label_da: Any,
    *,
    keep_prob: dict[int, float] | None = None,
    seed: int = DEFAULT_SEED,
    n: int = 16,
    nodata: int = LABEL_NODATA,
) -> dict[str, Any]:
    """Label-only size/prevalence preview for one (tile, date) — no HLS needed.

    Returns per-class patch counts, kept counts, and the kept-patch compound
    fraction, so the full-run size and class mix can be judged from labels alone
    before any streaming.
    """
    if keep_prob is None:
        keep_prob = DEFAULT_KEEP_PROB
    by_class = {0: 0, 1: 0, 2: 0, 3: 0}
    kept = {0: 0, 1: 0, 2: 0, 3: 0}
    kept_patches: list[tuple[int, int, int]] = []
    n_nodata = 0
    for r, c, bbox in grid_cells(tile_bbox, n):
        sid = f"{tile}_{date_iso}_r{r:02d}c{c:02d}"
        cls, keep = decide_patch(sid, bbox, label_da, keep_prob=keep_prob, seed=seed, nodata=nodata)
        if cls is None:
            n_nodata += 1
            continue
        by_class[cls] += 1
        if keep:
            kept[cls] += 1
            kept_patches.append((r, c, cls))
    n_kept = sum(kept.values())
    return {
        "tile": tile,
        "date": date_iso,
        "n_patches": n * n,
        "n_nodata": n_nodata,
        "by_class": by_class,
        "kept_by_class": kept,
        "n_kept": n_kept,
        "kept_compound_frac": (kept[3] / n_kept) if n_kept else 0.0,
        "kept_patches": kept_patches,
    }
