# M4 — Per-Tile Dataset Packaging (decisions, gate, layout)

Status: **packaging core COMPLETE & validated on 2023 T36RUU**; full multi-year
run pending the network-bound inputs (HLS streaming + 8-year cheap-source pulls).
This file records the design decisions, the per-tile packaging of the §4.3
`TileSample` record (`label_path`, `meteo_path`, `split`), and the validation
evidence on the August-2023 proof store.

## What M4 packaging produced

- `nilevit/schemas.py` — the §4.3 `TileSample` pydantic contract + region map
  (`region_for_point`, §4.1 bboxes) + `CLASS_NAMES`, `LABEL_NODATA=255`,
  `NUM_CLASSES=4`.
- `nilevit/tiles.py` — per-tile **label** packaging (pure stats + lazy geo):
  `label_histogram`, `valid_fraction`, `class_weights_from_counts`
  (`median_freq`/`inverse`), `aggregate_counts`, `label_date_for`,
  `mgrs_to_epsg`, `tile_grid_template`, `resample_label_to_tile`.
- `nilevit/meteo.py` — per-tile **meteo** packaging (pure):
  `METEO_CHANNELS` (7, fixed order), `METEO_WINDOW_DAYS=90`,
  `meteo_window_dates`, `assemble_meteo_series`, `meteo_channel_stats`
  (train-fit), `zscore_meteo`, `ERA5_SOURCE_VAR`, `ERA5_DAILY_AGG`,
  `aggregate_hourly`.
- `scripts/labels/package_tile_labels.py` — labels CLI (design A).
- `scripts/data/package_tile_meteo.py` — meteo CLI.
- Tests (all green): `test_schemas.py` (7), `test_tiles.py` (11),
  `test_meteo.py` (8), `test_package_tile_labels.py` (2),
  `test_package_tile_meteo.py` (1).
- Artifacts on disk (2023 proof): `tiles_T36RUU_2023_labels.zarr`,
  `tiles_T36RUU_2023_meteo.zarr`, `tiles_T36RUU_2023_labeled.parquet`,
  `configs/class_weights_v1.json`, `configs/meteo_norm_v1.json`.

## Decisions

**D1 — Labels: resample the ROI raster onto each tile (design A), not recompute.**
The §4.4 label field is the M3 ROI raster at 0.05° (~5.5 km). `package_tile_labels`
selects the MODIS composite active at the tile's date (`label_date_for` = latest
composite ≤ date) and resamples it onto the tile's UTM grid with nearest-neighbour
(categorical, no interpolation), nodata 255. Recomputing `label_from_rasters` at
30 m per tile gives **no** finer detail — every indicator is coarse — so A is the
faithful, efficient choice. **Consequence:** tiles (6.7 km) are smaller than a
label cell, so per-tile label maps are near-homogeneous (~1–2 classes/tile). This
is inherent to the M3/D2 label resolution; the scientific hook is prevalence/lead
-time, not fine segmentation IoU.

**D2 — Tile grid reconstructed from the centre point.**
The 05b Zarr stores only `image (sample, band, y, x)` + a string `sample` coord —
no per-tile transform/CRS. Tiles are a regular 30 m north-up grid in the MGRS
tile's UTM CRS (`mgrs_to_epsg`: T36RUU → EPSG:32636), and `center_lon/lat` is the
window centroid (verified: adjacent-column spacing 0.0701° = 224×30 m at lat
30.69). So `tile_grid_template` projects the centre to UTM and expands ±112×30 m
to the upper-left. Round-trip error is sub-metre — negligible vs the 5.5 km label.

**D3 — Out-of-ROI tiles dropped at packaging (default `--drop-out-of-roi`).**
05b does **not** clip to the ROI: 414/1806 T36RUU tiles have centres at lat
< 30.0° (south of the ROI's edge), so `split_for_sample` returns `"none"`. The
dataset is defined over the ROI (§4.1) and §4.5 splits only cover ROI cells, so a
`none` tile is not a dataset member. The flag drops them and the report records
`n_dropped_out_of_roi`; `--keep-out-of-roi` retains them as `split="none"`. Either
way they never enter the class-weight aggregation.

**D4 — Meteo: point series at the tile centre, RAW storage, train-only z-score.**
§4.3 meteo is `(T_m=90, V=7)` — a point series, not a spatial cube. Channels are
sampled at the nearest land cell (sea/missing → NaN, tolerated by z-score and
Time2Vec). Values are stored RAW; per-channel z-score stats are fit on the
**TRAIN split only** (`meteo_channel_stats`) and written to
`configs/meteo_norm_v1.json` for the loader to apply — the §4.5 anti-leakage
discipline extended to normalisation.

**D5 — ERA5-Land daily aggregation: sum fluxes, mean state vars, max/min for heat.**
Grounded in the 2023 raw files (cfgrib deaccumulated them): `t2m`/`swvl1` are
state variables varying hour-to-hour; `e`/`tp` are signed per-hour fluxes (`e` is
negative for evaporative loss). So `era5_e`/`era5_tp` are **summed** to daily
totals (not max-of-accumulation), `era5_t2m`/`era5_swvl1` are **meaned**, and
`chirts_tmax`/`chirts_tmin` are the daily **max**/**min** of `t2m`. `chirps_p` is
CHIRPS daily `precip` directly. The window for an Aug tile spans May–Aug, so the
CLI concatenates the monthly files it touches and leaves absent days NaN.

## Deviations from PRD (tracked, reversible)

| # | PRD | M4 reality | Why it's safe |
|---|-----|-----------|---------------|
| 1 | §4.3 single Zarr store w/ `image_path`/`meteo_path`/`label_path` | `image` in the 05b store; `label`/`meteo` in **sidecar** Zarrs; parquet keys by `sample_id` | Non-destructive during validation; loader resolves `sample_id` → variable. Merge into one store (or keep sidecars) at full-dataset assembly. |
| 2 | §4.3 `chirts_tmax/tmin` from CHIRTS | sourced from **ERA5-Land** daily max/min | B2 Decision 4 (CHIRTS-ERA5 not ROI-subsettable); same product as heat-z label, internally consistent. |

## Validation evidence (2023 T36RUU proof)

- **Labels:** 1806 → 414 dropped (sub-30°N) → **1392 kept**; split `{buffer 888,
  train 504}`; `label_date` resolved to `2023-07-28 / 08-13 / 08-29` (active
  composites). A train tile spot-checked clean: homogeneous heat-only (class 2,
  50176 px), `valid_pct 1.0`, no 255 leakage. Compound = 0 on this southern
  footprint is **correct geography** (compound lives on the E-Med shelf/coast, not
  the irrigated delta); the full-ROI rasters carry compound (9.8% on `08-13`).
  **The §4.4 [0.5%, 8%] gate is judged on the full multi-year dataset, not one
  tile.**
- **Meteo:** 1392 × 90 × 7, **coverage 1.000** (all land, no window gaps). Train
  z-score (504 tiles, buffer excluded) physically sane: `chirts_tmax` 309.3 K >
  `era5_t2m` 301.5 K > `chirts_tmin` 294.0 K; `era5_e` negative (−9.0e-4);
  `era5_tp`/`chirps_p` near zero (dry August delta); `swvl1` 0.068 m³/m³.
- **Class weights** (`median_freq`, from real histogram) replace the PRD
  placeholder `[0.1, 1.0, 1.0, 3.0]` — recomputed at full scale once compound
  -bearing regions are tiled.

## On-disk layout (proof; one MGRS tile-year)

```
data/interim/
  tiles_T36RUU_2023.zarr            image (sample, band, y, x) uint16        [05b]
  tiles_T36RUU_2023.parquet         05b index (sample_id, mgrs_tile, date, row,
                                    col, center_lon/lat, region, cloud_pct,
                                    valid_pct, sensor, geometry)             [05b]
  tiles_T36RUU_2023_labels.zarr     label (sample, y, x) uint8 {0,1,2,3}/255 [M4]
  tiles_T36RUU_2023_meteo.zarr      meteo (sample, t=90, channel=7) float32  [M4]
  tiles_T36RUU_2023_labeled.parquet 05b index + label_path, split, label_date,
                                    label_valid_pct, meteo_path, meteo_coverage [M4]
configs/
  splits/v1.json, splits/v1_temporal.json   spatial-block CV + temporal (§4.5) [07]
  class_weights_v1.json             aggregated histogram + median_freq/inverse  [M4]
  meteo_norm_v1.json                train-fit per-channel z-score stats         [M4]
```

## What remains in M4 (network-bound, feeds this proven pipeline)

1. Cheap-source pulls for 2017–2022 + 2024 (`pull_year` × 7) — ERA5/CHIRPS/MODIS.
2. HLS streaming (`01_download_hls`) → 05a harmonize → 05b tile at scale, for all
   ROI MGRS tiles × 2017–2024.
3. Re-run `package_tile_labels` + `package_tile_meteo` over the full tile set;
   recompute `class_weights_v1.json` and `meteo_norm_v1.json` at scale.
4. Assemble the canonical store (merge sidecars or finalise the loader contract);
   judge the §4.4 compound-prevalence gate on the full dataset.
5. Optional: Nov/Dec 2022 backfill to recover the 4 skipped winter-2023 label
   dates (M3 D5).
```
