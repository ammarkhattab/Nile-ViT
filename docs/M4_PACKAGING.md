# M4 — Per-Tile Dataset Packaging (decisions, gate, layout)

Status: **packaging + assembly core COMPLETE & validated on 2023 T36RUU**; full
multi-year run pending the network-bound inputs (HLS streaming + 8-year pulls).
This file records the design decisions, the per-tile packaging of the §4.3
`TileSample` record (`label_path`, `meteo_path`, `split`), the dataset assembly +
§4.5 leak-check gate, and the validation evidence on the August-2023 proof store.

## What M4 packaging produced

- `nilevit/schemas.py` — the §4.3 `TileSample` pydantic contract + region map
  (`region_for_point`, §4.1 bboxes) + `CLASS_NAMES`, `LABEL_NODATA=255`,
  `NUM_CLASSES=4`. `Split = Literal["train","val","test","ood"]` (no "buffer").
- `nilevit/tiles.py` — per-tile **label** packaging (design A: resample M3 ROI
  labels onto real tile grids); class-weight recompute.
- `nilevit/meteo.py` — per-tile **meteo** packaging (90×7, ERA5 daily aggregation
  grounded in the raw files, train-only z-score stats).
- `nilevit/dataset.py` — assembly + acceptance: `consolidate_index`,
  `filter_members`, `validate_tilesamples`, `spatial_leak_violations`,
  `temporal_leak_violations`, `split_region_year_counts`.
- Scripts: `scripts/labels/package_tile_labels.py`,
  `scripts/data/package_tile_meteo.py`, `scripts/data/assemble_dataset.py`.
- 51 offline tests green; lint/format clean.
- Artifacts (2023 proof): `tiles_T36RUU_2023_labels.zarr`,
  `tiles_T36RUU_2023_meteo.zarr`, `tiles_T36RUU_2023_labeled.parquet`,
  `data/processed/dataset_v1.parquet`, `configs/class_weights_v1.json`,
  `configs/meteo_norm_v1.json`.

## Decisions

**D1 — Labels: resample the ROI raster onto each tile (design A), not recompute.**
The §4.4 label field is the M3 ROI raster at 0.05° (~5.5 km). `package_tile_labels`
selects the MODIS composite active at the tile's date (`label_date_for` = latest
composite ≤ date) and resamples it onto the tile's UTM grid with nearest-neighbour
(categorical, nodata 255). Recomputing `label_from_rasters` at 30 m gives **no**
finer detail (every indicator is coarse), so A is faithful and efficient.
**Consequence:** tiles (6.7 km) < a label cell, so per-tile maps are
near-homogeneous (~1–2 classes/tile) — inherent to the M3/D2 label resolution; the
scientific hook is prevalence/lead-time, not fine segmentation IoU.

**D2 — Tile grid reconstructed from the centre point.**
The 05b Zarr stores only `image (sample, band, y, x)` + a string `sample` coord —
no transform/CRS. Tiles are a regular 30 m north-up grid in the MGRS tile's UTM CRS
(`mgrs_to_epsg`: T36RUU → EPSG:32636), and `center_lon/lat` is the window centroid
(verified: adjacent-column spacing 0.0701° = 224×30 m at lat 30.69). `tile_grid_
template` projects the centre to UTM and expands ±112×30 m. Round-trip error is
sub-metre — negligible vs the 5.5 km label.

**D3 — Out-of-ROI tiles dropped at label packaging (default `--drop-out-of-roi`).**
05b does **not** clip to the ROI: 414/1806 T36RUU tiles centre at lat < 30.0°, so
`split_for_sample` returns `"none"`. The dataset is ROI-defined (§4.1), so these are
dropped (report records `n_dropped_out_of_roi`); `--keep-out-of-roi` retains them.

**D4 — Meteo: point series at the tile centre, RAW storage, train-only z-score.**
§4.3 meteo is `(90, 7)` — a point series, not a spatial cube. Channels are sampled
at the nearest land cell (sea/missing → NaN). Values stored RAW; z-score stats fit
on the **TRAIN split only** → `configs/meteo_norm_v1.json`, applied by the loader.

**D5 — ERA5-Land daily aggregation: sum fluxes, mean state vars, max/min for heat.**
Grounded in the 2023 raw files (cfgrib deaccumulated them): `e`/`tp` are signed
per-hour fluxes → **summed** to daily totals; `t2m`/`swvl1` → **mean**;
`chirts_tmax`/`chirts_tmin` → daily **max**/**min** of `t2m` (B2 Decision 4);
`chirps_p` = CHIRPS daily `precip`. The Aug-tile window spans May–Aug, so the CLI
concatenates the monthly files it touches and leaves absent days NaN.

**D6 — Buffer/none tiles are NOT dataset members; excluded at assembly.**
`split_for_sample` returns `"buffer"` for tiles within the leakage buffer of a
differently-assigned cell, and `"none"` outside all cells. These are §4.5 *spacing*,
not samples, and `TileSample.split` has no such members. `assemble_dataset.py`
filters to `MEMBER_SPLITS = get_args(Split)` before validating/writing
`dataset_v1.parquet`; excluded counts are reported, never silently dropped. (This
supersedes an earlier draft that kept buffer rows in the index — that produced 888
schema-validation failures on the proof, which is how the contradiction surfaced.)

## Deviations from PRD (tracked, reversible)

| # | PRD | M4 reality | Why it's safe |
|---|-----|-----------|---------------|
| 1 | §4.3 single Zarr store w/ explicit path columns | `image` in 05b store; `label`/`meteo` in **sidecar** Zarrs; rows keyed by `sample_id`; `image_path` derived at assembly | Non-destructive; loader resolves `sample_id`→variable. Merge vs. keep-sidecars decided at full-dataset assembly (affects M5 HF shards). |
| 2 | §4.3 `chirts_tmax/tmin` from CHIRTS | ERA5-Land daily max/min | B2 Decision 4; same product as heat-z label, internally consistent. |

## Validation evidence (2023 T36RUU proof)

- **Labels:** 1806 → 414 dropped (sub-30°N) → **1392 kept**; `label_date` resolved
  to active composites (`2023-07-28/08-13/08-29`). Train tile spot-checked clean
  (heat-only, valid_pct 1.0, no 255 leakage). Compound = 0 on this southern
  footprint is **correct geography**; full-ROI rasters carry compound (9.8% on
  `08-13`). The §4.4 [0.5%, 8%] gate is judged on the full dataset, not one tile.
- **Meteo:** 1392 × 90 × 7, **coverage 1.000**. Train z-score (504 tiles, buffer
  excluded) physically sane: `chirts_tmax` 309.3 K > `era5_t2m` 301.5 K >
  `chirts_tmin` 294.0 K; `era5_e` negative; dry-August precip near zero.
- **Assembly:** 1392 → **504 members** (`{train: 504}`; regions delta 168, em_shelf
  336), **888 buffer excluded**. `schema=PASS, spatial_leak=PASS,
  temporal_leak=PASS`. No test tiles yet (one MGRS footprint), so the leak-check is
  trivially clean — correct; it becomes load-bearing once test cells are tiled.

## On-disk layout (proof; one MGRS tile-year)

```
data/interim/
  tiles_T36RUU_2023.zarr            image (sample, band, y, x) uint16        [05b]
  tiles_T36RUU_2023.parquet         05b index                               [05b]
  tiles_T36RUU_2023_labels.zarr     label (sample, y, x) uint8 {0,1,2,3}/255 [M4]
  tiles_T36RUU_2023_meteo.zarr      meteo (sample, t=90, channel=7) float32  [M4]
  tiles_T36RUU_2023_labeled.parquet 05b index + label_path, split, label_date,
                                    label_valid_pct, meteo_path, meteo_coverage [M4]
data/processed/
  dataset_v1.parquet                consolidated MEMBER index (+image_path)  [M4]
  dataset_v1_report.json            acceptance + split/region/year counts    [M4]
configs/
  splits/v1.json, splits/v1_temporal.json   spatial-block CV + temporal (§4.5) [07]
  class_weights_v1.json             aggregated histogram + median_freq/inverse  [M4]
  meteo_norm_v1.json                train-fit per-channel z-score stats         [M4]
```

## What remains in M4 (network-bound, feeds this proven pipeline)

1. Cheap-source pulls 2017–2022 + 2024 (`pull_year` × 7).
2. HLS streaming (`01_download_hls`) → 05a → 05b for all ROI MGRS tiles × 2017–2024.
3. Re-run `package_tile_labels` + `package_tile_meteo`; recompute class weights +
   meteo norm at scale.
4. `assemble_dataset.py` over all sidecars; **judge the §4.4 compound gate** on the
   full dataset (the leak-check becomes load-bearing once test cells are populated).
5. Optional: Nov/Dec 2022 backfill (4 skipped winter-2023 label dates).
6. Decide sidecar-vs-merged canonical store (affects M5 HF WebDataset shards).
```
