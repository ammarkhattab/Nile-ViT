# ruff: noqa: B008
"""Package per-tile 224x224 labels into the tiled dataset (PRD section 4.3, M4).

Design A: consume the existing ROI label rasters (M3 output of
``nilevit.labels.label_from_rasters``) and resample them onto each real tile grid
-- no label-logic change, only the grid they are written onto.

For each tile-time sample in the 05b index:
  1. pick the ROI label raster active at the tile's date (label_date_for),
  2. reconstruct the tile's UTM grid from its centre (tile_grid_template),
  3. resample the ROI label onto it (nearest, categorical, nodata 255),
  4. record the class histogram, label valid-fraction, the section 4.5 split
     (split_for_sample on the v1 cell map), and the label date used.

Outputs (idempotent; safe to re-run):
  <stem>_labels.zarr     a `label (sample, y, x)` uint8 store, sample-aligned
  <stem>_labeled.parquet the 05b index + label_path, split, label_date, label_valid_pct
  class_weights_v1.json   aggregated histogram + recomputed focal-loss weights

Buffer/none tiles (section 4.5 leakage guard) are kept in the parquet with their
split label but excluded from the class-weight aggregation.

Usage:
    uv run python scripts/labels/package_tile_labels.py \
        --tiles-zarr data/interim/tiles_T36RUU_2023.zarr \
        --tiles-parquet data/interim/tiles_T36RUU_2023.parquet
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import shutil
from collections import Counter
from pathlib import Path

import typer

from nilevit.schemas import LABEL_NODATA
from nilevit.splits import split_for_sample
from nilevit.tiles import (
    aggregate_counts,
    class_weights_from_counts,
    label_date_for,
    label_histogram,
    resample_label_to_tile,
    tile_grid_template,
    valid_fraction,
)

with contextlib.suppress(ImportError):
    import nilevit  # noqa: F401

app = typer.Typer(add_completion=False, help=__doc__)

USABLE_SPLITS = frozenset({"train", "val", "test", "ood"})


def _discover_label_rasters(labels_dir: Path) -> dict[dt.date, Path]:
    """Map each ``label_<YYYY-MM-DD>.tif`` to its date."""
    rasters: dict[dt.date, Path] = {}
    for tif in sorted(labels_dir.glob("label_*.tif")):
        rasters[dt.date.fromisoformat(tif.stem.removeprefix("label_"))] = tif
    return rasters


@app.command()
def main(
    tiles_zarr: Path = typer.Option(..., help="05b cube store (the `image` variable)."),
    tiles_parquet: Path = typer.Option(..., help="05b GeoParquet index."),
    labels_dir: Path = typer.Option(
        Path("data/interim/labels_2023"), help="Directory of ROI label_*.tif rasters."
    ),
    splits_json: Path = typer.Option(
        Path("configs/splits/v1.json"), help="v1 spatial-block split map."
    ),
    label_zarr: Path | None = typer.Option(
        None, help="Output label store (default: <stem>_labels.zarr)."
    ),
    out_parquet: Path | None = typer.Option(
        None, help="Enriched index (default: <stem>_labeled.parquet)."
    ),
    weights_out: Path = typer.Option(
        Path("configs/class_weights_v1.json"), help="Class-weights report path."
    ),
    drop_out_of_roi: bool = typer.Option(
        True,
        "--drop-out-of-roi/--keep-out-of-roi",
        help="Drop tiles whose centre falls outside every ROI/OOD cell (split 'none').",
    ),
) -> None:
    """Resample ROI labels onto every tile and recompute class weights."""
    import geopandas as gpd
    import numpy as np
    import rioxarray
    import xarray as xr

    label_zarr = label_zarr or tiles_zarr.with_name(f"{tiles_zarr.stem}_labels.zarr")
    out_parquet = out_parquet or tiles_parquet.with_name(f"{tiles_parquet.stem}_labeled.parquet")

    split_doc = json.loads(splits_json.read_text(encoding="utf-8"))
    cell_map = split_doc["cells"]
    buffer_km = float(split_doc.get("buffer_km", 25.0))

    label_rasters = _discover_label_rasters(labels_dir)
    if not label_rasters:
        raise typer.BadParameter(f"no label_*.tif found in {labels_dir}")
    available_dates = sorted(label_rasters)

    gdf = gpd.read_parquet(tiles_parquet).set_index("sample_id")
    dataset = xr.open_zarr(tiles_zarr, consolidated=False)
    sample_order = [str(s) for s in dataset["sample"].values]
    size = int(dataset.sizes["y"])
    n_samples = len(sample_order)

    labels = np.full((n_samples, size, size), LABEL_NODATA, dtype=np.uint8)
    splits: list[str] = []
    label_valid: list[float] = []
    label_dates_used: list[str] = []
    histograms: list[dict[int, int]] = []
    raster_cache: dict[dt.date, object] = {}

    for index, sample_id in enumerate(sample_order):
        row = gdf.loc[sample_id]
        lon, lat = float(row["center_lon"]), float(row["center_lat"])
        tile_date = dt.date.fromisoformat(str(row["date"]))

        active_date = label_date_for(tile_date, available_dates)
        if active_date not in raster_cache:
            raster_cache[active_date] = rioxarray.open_rasterio(
                label_rasters[active_date]
            ).squeeze()

        template = tile_grid_template(lon, lat, str(row["mgrs_tile"]), size=size)
        tile_label = resample_label_to_tile(raster_cache[active_date], template).values

        labels[index] = tile_label
        histograms.append(label_histogram(tile_label))
        label_valid.append(valid_fraction(tile_label))
        splits.append(split_for_sample(lon, lat, cell_map, buffer_km=buffer_km))
        label_dates_used.append(active_date.isoformat())

    # --- filter out-of-ROI tiles (split 'none' -> outside every ROI/OOD cell) ---
    keep = [i for i, split in enumerate(splits) if not (drop_out_of_roi and split == "none")]
    n_dropped = n_samples - len(keep)
    kept_samples = [sample_order[i] for i in keep]
    labels = labels[keep]
    splits = [splits[i] for i in keep]
    label_valid = [label_valid[i] for i in keep]
    label_dates_used = [label_dates_used[i] for i in keep]
    histograms = [histograms[i] for i in keep]

    # --- write the label store (mode="w" -> idempotent) ---
    if label_zarr.exists():
        shutil.rmtree(label_zarr)
    xr.Dataset(
        {"label": (("sample", "y", "x"), labels)},
        coords={"sample": np.array(kept_samples)},
    ).to_zarr(label_zarr, mode="w", consolidated=False)

    # --- enrich the index (follow the kept zarr sample order) ---
    enriched = gdf.loc[kept_samples].reset_index()
    enriched["label_path"] = f"{label_zarr.name}::label"
    enriched["split"] = splits
    enriched["label_valid_pct"] = label_valid
    enriched["label_date"] = label_dates_used
    enriched.to_parquet(out_parquet)

    # --- recompute class weights over usable tiles ---
    usable = [h for h, s in zip(histograms, splits, strict=True) if s in USABLE_SPLITS]
    counts = aggregate_counts(usable)
    total = sum(counts.values())
    compound_prevalence = counts[3] / total if total else 0.0
    report = {
        "source_zarr": str(tiles_zarr),
        "n_samples": n_samples,
        "n_dropped_out_of_roi": n_dropped,
        "n_kept": len(kept_samples),
        "n_usable_tiles": len(usable),
        "counts": counts,
        "compound_prevalence": compound_prevalence,
        "class_weights_median_freq": class_weights_from_counts(counts, scheme="median_freq"),
        "class_weights_inverse": class_weights_from_counts(counts, scheme="inverse"),
        "placeholder_weights": [0.1, 1.0, 1.0, 3.0],
    }
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    weights_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # --- summary ---
    typer.echo(f"wrote {label_zarr}  (label: {len(kept_samples)}x{size}x{size} uint8)")
    typer.echo(f"wrote {out_parquet}  (+label_path, +split, +label_date)")
    typer.echo(f"wrote {weights_out}")
    typer.echo(
        f"  samples: {n_samples} total, {n_dropped} dropped out-of-ROI, "
        f"{len(kept_samples)} kept"
    )
    typer.echo(f"  split distribution: {dict(Counter(splits))}")
    typer.echo(f"  label dates used:   {dict(Counter(label_dates_used))}")
    typer.echo(f"  class counts (usable): {counts}")
    typer.echo(f"  compound prevalence:   {compound_prevalence:.4f}")
    typer.echo(
        "  class_weights median_freq: "
        f"{[round(w, 4) for w in report['class_weights_median_freq']]}"
    )


if __name__ == "__main__":
    app()
