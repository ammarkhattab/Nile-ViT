# ruff: noqa: B008
"""Build the dataset keep-list + a size/prevalence report from the M3 labels.

Implements the M4 sampling model (documented in docs/M4_PACKAGING.md, decision D7):

  Temporal  - one central date per (tile, MODIS-composite window). Each label_*.tif
              is one composite window; 05b realises it as the least-cloudy HLS
              acquisition in that window. This stops the HLS ~2-3 day revisit from
              multiplying the sample count ~6x per window.
  Spatial   - within a (tile, date), keep every event patch (compound/heat/drought)
              and subsample class-0 background at --keep-bg-prob (seeded, reproducible).

Runs on labels alone (no HLS), so the full-run size and the compound fraction are
known before any streaming. Writes:
  * a keep-list  (tile -> composite-date -> [[row, col, class], ...]), and
  * a report     (per-date + aggregate counts, prevalence, 8-year extrapolation,
                  and the §6.1 embedding-size estimate).

Usage:
  uv run python scripts/data/select_samples.py --labels-dir data/interim/labels_2023
  uv run python scripts/data/select_samples.py --labels-dir data/interim/labels_2023 \
      --keep-bg-prob 0.05 --years 8
"""

from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path

import typer

from nilevit.select import DEFAULT_SEED, selection_report

with contextlib.suppress(ImportError):
    import nilevit  # noqa: F401

app = typer.Typer(add_completion=False, help=__doc__)

EMBED_KB_PER_SAMPLE = 770  # §6.1: h_v (196,1024) fp16 ~= 770 KB
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _date_from_label_name(name: str) -> str | None:
    m = _DATE_RE.search(name)
    return m.group(1) if m else None


@app.command()
def main(
    labels_dir: Path = typer.Option(..., help="Dir of M3 label_YYYY-MM-DD.tif rasters."),
    roi_tiles: Path = typer.Option(Path("configs/roi_tiles.json"), help="ROI tile coverage JSON."),
    keep_none: float = typer.Option(
        0.015, "--keep-none", help="Keep fraction for class-0 (background) patches."
    ),
    keep_drought: float = typer.Option(
        0.04, "--keep-drought", help="Keep fraction for drought-only (class-1) patches."
    ),
    keep_heat: float = typer.Option(
        0.4, "--keep-heat", help="Keep fraction for heat-only (class-2) patches."
    ),
    keep_compound: float = typer.Option(
        1.0, "--keep-compound", help="Keep fraction for compound (class-3) patches."
    ),
    years: int = typer.Option(
        8, "--years", help="Year count for the full-dataset size extrapolation."
    ),
    seed: int = typer.Option(DEFAULT_SEED, help="Seed for the background subsample."),
    out: Path = typer.Option(Path("configs/keeplist_v1.json"), help="Keep-list output path."),
    report_out: Path = typer.Option(
        Path("configs/keeplist_v1_report.json"), help="Report output path."
    ),
) -> None:
    """Select samples from labels; write the keep-list + size/prevalence report."""
    import rioxarray

    keep_prob = {0: keep_none, 1: keep_drought, 2: keep_heat, 3: keep_compound}

    cfg = json.loads(roi_tiles.read_text())
    land = cfg.get("land_roi_tiles")
    if not land:
        typer.echo(
            "ERROR: roi_tiles.json has no 'land_roi_tiles' (run make_roi_tiles "
            "with --label-raster first)."
        )
        raise typer.Exit(code=1)
    bboxes = cfg["roi_tiles"]

    label_files = sorted(labels_dir.glob("label_*.tif"))
    dates = [(p, _date_from_label_name(p.name)) for p in label_files]
    dates = [(p, d) for p, d in dates if d]
    if not dates:
        typer.echo(f"ERROR: no label_*.tif in {labels_dir}")
        raise typer.Exit(code=1)

    keeplist: dict[str, dict[str, list[list[int]]]] = {}
    agg_by_class = {0: 0, 1: 0, 2: 0, 3: 0}
    agg_kept = {0: 0, 1: 0, 2: 0, 3: 0}
    per_date: list[dict] = []

    for raster_path, date_iso in dates:
        da = rioxarray.open_rasterio(raster_path).squeeze()
        date_kept = {0: 0, 1: 0, 2: 0, 3: 0}
        for tile in land:
            rep = selection_report(tile, date_iso, bboxes[tile], da, keep_prob=keep_prob, seed=seed)
            for cls in (0, 1, 2, 3):
                agg_by_class[cls] += rep["by_class"][cls]
                agg_kept[cls] += rep["kept_by_class"][cls]
                date_kept[cls] += rep["kept_by_class"][cls]
            if rep["kept_patches"]:
                keeplist.setdefault(tile, {})[date_iso] = [
                    [r, c, cls] for r, c, cls in rep["kept_patches"]
                ]
        per_date.append(
            {"date": date_iso, "kept_by_class": date_kept, "n_kept": sum(date_kept.values())}
        )

    n_kept = sum(agg_kept.values())
    n_dates = len(dates)
    # 2023 labels -> full dataset extrapolation across `years`.
    est_full = round(n_kept * years)
    est_embed_gb = est_full * EMBED_KB_PER_SAMPLE / 1e6
    compound_frac = agg_kept[3] / n_kept if n_kept else 0.0

    report = {
        "keep_prob": keep_prob,
        "seed": seed,
        "n_land_tiles": len(land),
        "n_composite_dates": n_dates,
        "kept_by_class": agg_kept,
        "by_class_all": agg_by_class,
        "n_kept_this_year": n_kept,
        "kept_compound_frac": round(compound_frac, 4),
        "years_extrapolated": years,
        "est_full_samples": est_full,
        "est_embedding_gb": round(est_embed_gb, 1),
        "per_date": per_date,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(keeplist) + "\n", encoding="utf-8")
    report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    typer.echo(f"wrote {out}  and  {report_out}")
    typer.echo(f"  land tiles: {len(land)}   composite dates: {n_dates}")
    typer.echo(
        f"  keep none/drought/heat/compound: {keep_none}/{keep_drought}/{keep_heat}/{keep_compound}"
    )
    typer.echo(f"  kept this year: {n_kept:,}  by class {agg_kept}")
    typer.echo(f"  compound frac of kept: {compound_frac:.2%}")
    typer.echo(f"  -> est. full dataset (x{years} yr): {est_full:,} samples")
    typer.echo(f"  -> est. embeddings: {est_embed_gb:.1f} GB (§10.3 cap 80 GB)")
    if est_embed_gb > 80:
        typer.echo("  [!] over the 80 GB cap - lower --keep-drought / --keep-none.")


if __name__ == "__main__":
    app()
