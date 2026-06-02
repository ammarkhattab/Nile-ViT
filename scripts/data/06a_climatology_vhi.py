"""Script 06a (climatology) - per-pixel min/max of MODIS NDVI or LST.

Builds the VHI baseline from PRD 4.4: "VCI from MOD13Q1 NDVI, TCI from MOD11A2
LST, both vs 2001-2020 pixel-wise min/max." For each MODIS v05 tile over the
ROI, this streams every composite in [start-year, end-year] from Planetary
Computer, masks fill values, and folds each into a running per-pixel min and
max. The decades of raw composites are never kept - only the reduced min/max
reference rasters are written, so 20 years costs a few MB on disk.

Run once per product (NDVI then LST), like the B1 driver did:

  uv run python scripts/data/06a_climatology_vhi.py --product MOD13Q1
  uv run python scripts/data/06a_climatology_vhi.py --product MOD11A2

Outputs (sinusoidal CRS, per tile - Script 06 mosaics + reprojects to the HLS
grid the same way 05a does):

  data/climatology/ndvi_min_h19v05.tif  ndvi_max_h19v05.tif  (MOD13Q1)
  data/climatology/lst_min_h19v05.tif   lst_max_h19v05.tif   (MOD11A2)
  ... for each of h19v05, h20v05, h21v05

Resumability: after each year the running accumulators + the set of completed
years are checkpointed to data/climatology/.ckpt_<var>.npz. Re-run with
--resume (default) to pick up where a crash / connection drop left off. This
matters because a full 2001-2020 pass is thousands of COG reads and can take
hours - exactly the kind of long job that hits a network blip.

Tip: validate the whole label pipeline on a short range first
(--start-year 2018 --end-year 2022, a few minutes), then do the full
2001-2020 once Script 06 is known-good.
"""

# typer's standard pattern uses function calls in argument defaults.
# ruff: noqa: B008

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import NamedTuple

import typer
from rich.console import Console
from rich.table import Table

with contextlib.suppress(ImportError):
    import nilevit  # noqa: F401

app = typer.Typer(
    add_completion=False,
    help="MODIS NDVI/LST per-pixel min/max climatology (Script 06a, VHI baseline).",
)
console = Console()

MPC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
ROI_BBOX = (22.0, 30.0, 36.0, 37.0)  # W, S, E, N
V05_TILES = ("h19v05", "h20v05", "h21v05")


class Product(NamedTuple):
    """A MODIS product: STAC collection, band asset, output var, fill rule."""

    key: str  # MOD13Q1 / MOD11A2
    collection: str
    asset: str
    var: str  # ndvi / lst (used in output filenames)


PRODUCTS: dict[str, Product] = {
    "MOD13Q1": Product("MOD13Q1", "modis-13Q1-061", "250m_16_days_NDVI", "ndvi"),
    "MOD11A2": Product("MOD11A2", "modis-11A2-061", "LST_Day_1km", "lst"),
}


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


def tile_of(item_id: str) -> str | None:
    """Extract the MODIS tile id (e.g. 'h19v05') from a granule id."""
    import re

    m = re.search(r"h\d{2}v\d{2}", item_id)
    return m.group(0) if m else None


def valid_mask_for(product_key: str, arr):
    """Boolean mask of valid (non-fill) pixels for a product's raw values.

    MOD13Q1 NDVI: valid range [-2000, 10000]; fill -3000 (and below) excluded.
    MOD11A2 LST_Day: fill 0; valid values are strictly positive.
    """
    if product_key == "MOD13Q1":
        return (arr >= -2000) & (arr <= 10000)
    return arr > 0


def update_minmax(acc_min, acc_max, values, valid):
    """Fold one composite into running per-pixel min/max (NaN = unseen).

    Invalid pixels become NaN so np.fmin/np.fmax ignore them. The first call
    (acc_min/acc_max is None) seeds the accumulators.
    """
    import numpy as np

    masked = np.where(valid, values.astype("float32"), np.nan)
    if acc_min is None:
        return masked.copy(), masked.copy()
    return np.fmin(acc_min, masked), np.fmax(acc_max, masked)


def configure_gdal_http(timeout: int = 60) -> None:
    """Bound /vsicurl reads so a stalled MPC connection can't hang forever.

    GDAL's curl layer has no read timeout by default, so one dead socket wedges
    the whole run (the symptom: a year that searched fine but never finishes).
    These caps make a stalled read abort within ~timeout seconds and retry a
    few times before giving up, so the per-item retry loop can skip it.
    """
    os.environ["GDAL_HTTP_TIMEOUT"] = str(timeout)
    os.environ["GDAL_HTTP_CONNECTTIMEOUT"] = "20"
    os.environ["GDAL_HTTP_MAX_RETRY"] = "3"
    os.environ["GDAL_HTTP_RETRY_DELAY"] = "2"
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
    os.environ.setdefault("VSI_CACHE", "TRUE")


def _open_clip(href: str, bbox: tuple[float, float, float, float]):
    """Open a COG href and read the ROI window into memory (the I/O step)."""
    import rioxarray

    da = rioxarray.open_rasterio(href).rio.clip_box(*bbox, crs="EPSG:4326")
    return da.load()


def read_clipped(
    href: str,
    bbox: tuple[float, float, float, float],
    retries: int = 3,
    delay: float = 3.0,
):
    """Read a clipped COG with retry; raise RuntimeError after `retries` fails.

    Any failure (including a GDAL timeout from a stalled socket) is retried;
    after the last attempt it raises so the caller can skip the composite
    rather than hang or abort the whole run.
    """
    import time

    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return _open_clip(href, bbox)
        except Exception as exc:
            last = exc
            if attempt < retries:
                time.sleep(delay)
    raise RuntimeError(f"read failed after {retries} attempts: {last}")


def ckpt_path(out_dir: Path, var: str) -> Path:
    """Path of the resumability checkpoint for a variable."""
    return out_dir / f".ckpt_{var}.npz"


def _load_checkpoint(path: Path):
    """Return (acc dict keyed 'min_<tile>'/'max_<tile>', set(done_years)) or empty."""
    import numpy as np

    if not path.exists():
        return {}, set()
    data = np.load(path, allow_pickle=True)
    acc = {k: data[k] for k in data.files if k != "done_years"}
    done = {int(y) for y in data["done_years"]} if "done_years" in data else set()
    return acc, done


def _save_checkpoint(path: Path, acc: dict, done_years: set[int]) -> None:
    """Atomically persist accumulators + completed years."""
    import numpy as np

    # np.savez appends '.npz' unless the name already ends in it, so the temp
    # name must end in '.npz' too or the subsequent replace() can't find it.
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, done_years=np.array(sorted(done_years)), **acc)
    tmp.replace(path)


@app.command()
def main(
    product: str = typer.Option(
        "MOD13Q1", "--product", "-p", help="MOD13Q1 (NDVI) or MOD11A2 (LST)."
    ),
    start_year: int = typer.Option(2001, "--start-year", help="First year (incl.)."),
    end_year: int = typer.Option(2020, "--end-year", help="Last year (incl.)."),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", "-o", help="Override climatology output dir."
    ),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Resume from a checkpoint if present."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show the plan and per-year item counts; write nothing.",
    ),
) -> None:
    """Compute per-pixel min/max of a MODIS product over a year range."""
    product = product.upper()
    if product not in PRODUCTS:
        console.print(f"[red]ERROR[/red] --product must be one of {list(PRODUCTS)}.")
        raise typer.Exit(code=2)
    if start_year > end_year:
        console.print("[red]ERROR[/red] --start-year must be <= --end-year.")
        raise typer.Exit(code=2)

    prod = PRODUCTS[product]
    root = default_data_root()
    out_dir = output_dir or (root / "climatology")
    out_dir.mkdir(parents=True, exist_ok=True)
    years = list(range(start_year, end_year + 1))

    plan = Table(title="06a VHI climatology plan", show_header=False, title_style="bold cyan")
    plan.add_column(style="dim")
    plan.add_column(style="bold")
    plan.add_row("Product", f"{prod.key} -> {prod.var}")
    plan.add_row("Collection", prod.collection)
    plan.add_row("Years", f"{start_year}-{end_year} ({len(years)})")
    plan.add_row("Tiles", ", ".join(V05_TILES))
    plan.add_row("ROI bbox (W S E N)", str(ROI_BBOX))
    plan.add_row("Output dir", str(out_dir))
    plan.add_row("Resume", "yes" if resume else "no")
    plan.add_row("Mode", "dry-run" if dry_run else "run")
    console.print(plan)
    console.print()

    try:
        import numpy as np
        import planetary_computer as pc
        import pystac_client
        import rioxarray  # noqa: F401
    except ImportError as exc:
        console.print(f"[red]ERROR[/red] missing deps: {exc}")
        raise typer.Exit(code=1) from None

    configure_gdal_http()
    client = pystac_client.Client.open(MPC_STAC_URL)

    # Per-tile accumulators + transform/crs captured from the first read.
    acc: dict = {}
    done_years: set[int] = set()
    if resume and not dry_run:
        acc, done_years = _load_checkpoint(ckpt_path(out_dir, prod.var))
        if done_years:
            console.print(
                f"[dim]Resuming: {len(done_years)} year(s) already done: "
                f"{sorted(done_years)}[/dim]"
            )
    geo: dict[str, object] = {}  # tile -> (crs, transform) for output writing

    for year in years:
        if year in done_years:
            console.print(f"[dim]{year}: already done, skipping.[/dim]")
            continue
        search = client.search(
            collections=[prod.collection],
            bbox=ROI_BBOX,
            datetime=f"{year}-01-01/{year}-12-31",
        )
        items = [it for it in search.items() if it.id.startswith("MOD")]
        terra_v05 = [it for it in items if tile_of(it.id) in V05_TILES]
        console.print(
            f"[bold]{year}[/bold]: {len(items)} Terra item(s), " f"{len(terra_v05)} in v05 tiles"
        )
        if dry_run:
            continue

        n_used, n_skip, n_fail = 0, 0, 0
        for it in terra_v05:
            tile = tile_of(it.id)
            href = pc.sign(it.assets[prod.asset].href)
            try:
                da = read_clipped(href, ROI_BBOX)
            except RuntimeError as exc:
                n_fail += 1
                console.print(f"  [yellow]skip {it.id}: {exc}[/yellow]")
                continue
            arr = da.values[0]
            valid = valid_mask_for(prod.key, arr)
            kmin, kmax = f"min_{tile}", f"max_{tile}"
            existing_min = acc.get(kmin)
            existing_max = acc.get(kmax)
            new_min, new_max = update_minmax(existing_min, existing_max, arr, valid)
            # Guard against an off-by-one clip changing shape mid-stream.
            if existing_min is not None and new_min.shape != existing_min.shape:
                n_skip += 1
                continue
            acc[kmin], acc[kmax] = new_min, new_max
            if tile not in geo:
                geo[tile] = (da.rio.crs, da.rio.transform())
            n_used += 1
        console.print(f"  used {n_used}, skipped {n_skip}, failed {n_fail}")

        done_years.add(year)
        _save_checkpoint(ckpt_path(out_dir, prod.var), acc, done_years)

    if dry_run:
        console.print("\n[yellow]Dry run - nothing written.[/yellow]")
        raise typer.Exit(code=0)

    # ---- Write per-tile min/max reference rasters ----
    import xarray as xr

    written = []
    for tile in V05_TILES:
        kmin, kmax = f"min_{tile}", f"max_{tile}"
        if kmin not in acc or tile not in geo:
            console.print(f"[yellow]{tile}: no data accumulated, skipping.[/yellow]")
            continue
        crs, transform = geo[tile]
        for kind, key in (("min", kmin), ("max", kmax)):
            arr = np.nan_to_num(acc[key], nan=0).astype("float32")
            da = xr.DataArray(arr, dims=("y", "x"))
            da = da.rio.write_crs(crs).rio.write_transform(transform)
            dst = out_dir / f"{prod.var}_{kind}_{tile}.tif"
            da.rio.to_raster(dst)
            written.append(dst.name)

    console.print(f"\n[green]Wrote {len(written)} reference raster(s):[/green]")
    for name in written:
        console.print(f"  {name}")
    console.print(f"\n[green]VHI {prod.var} climatology complete.[/green]")


if __name__ == "__main__":
    app()
