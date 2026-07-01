# HLS via Planetary-Computer STAC streaming (M4)

Why the dataset streams HLS instead of shipping pixels, how it's wired, the
evidence it works, and the exact recipe to re-tile from public data. This is the
reproducibility contract for the NeurIPS Datasets & Benchmarks release.

## Decision (F2): stream, don't bulk-download

PRD §10.3 caps raw HLS at ~80 GB and says, twice, *"do not re-download raw HLS more
than once… cache via stackstac or use Microsoft PC's STAC streaming directly."*

Bulk download is infeasible at ROI scale: **81 land tiles × 8 years × any season =
0.9–3.7 TB** of raw HLS (measured cadence ~12 cloud-filtered acquisitions/month,
~50–80 MB per tile-date), i.e. **12–45× over budget**. Streaming is the only path
that satisfies full coverage + the 80 GB budget + the §4.4 compound gate at once —
and it is what a D&B reviewer wants: "re-tile from public STAC + manifest" is far
more reproducible than a multi-TB personal drive. So raw scenes are **never
persisted**; the ~80 GB budget becomes the tiled-Zarr cache.

## Wiring

- **`nilevit/hls_bands.py`** — canonical band map (Prithvi 6 bands → HLS asset codes,
  S30 vs L30). Single source of truth; both the disk and STAC paths import it.
- **`nilevit/hls_stac.py`** — `open_catalog` (anonymous PC signing), `search_hls_items`
  (both collections, cloud-filtered, filtered to the exact tile id parsed from the
  granule name), `item_band_hrefs` (assets → signed COG hrefs), `hrefs_by_date`
  (one scene per date), `manifest_rows_for_items` (the publishable manifest).
- **`scripts/data/05b_tile.py`** — `--source {disk,stac}` + `--cloud-max`. A "scene"
  normalises either source to `(sensor, {band: source_str}, fmask_source)`; a source
  string is a local path or a signed COG href, and `rioxarray.open_rasterio` reads
  both identically. The entire patch loop (grid, cloud/valid fractions, Zarr/Parquet
  append, `known_ids` idempotency) is **source-agnostic and unchanged**.

Confirmed catalog facts (live probe): collections `hls2-s30` / `hls2-l30`; anonymous
`planetary_computer.sign` works (`PC_SDK_SUBSCRIPTION_KEY` optional, only lifts the
rate limit for the full pull); asset keys are raw band codes + `Fmask`; item ids are
`HLS.<S30|L30>.<TILE>.<YYYYDDD>T…` (sensor + tile parse).

## Validation evidence (real, on-machine)

- **Offline:** 11 unit tests (`test_hls_stac.py` 8, `test_05b_stac.py` 3) — sensor/tile
  parsing, S30/L30 band-href mapping, missing-band rejection, exact-tile filtering,
  one-scene-per-date grouping, canonical-map drift guard, disk-scene building, STAC
  dispatch. Full suite 67 green.
- **Live search:** T36RUU 2023-08 → 20 streamable scenes (S30+L30, <50% cloud), each
  resolving all 6 bands + Fmask to signed hrefs.
- **Live tile (the end-to-end proof):**
  `05b --tile T36RUU --date 2023-08-01 --source stac` streamed the scene and wrote
  256 patches (16×16, 224², 6 bands, uint16) — **identical grid to the disk build** —
  with `valid_pct 1.000`, `cloud_pct 0.008`, regions `em_shelf 192 / delta 64`, and
  **no raw scene on disk**. Re-running reported `256 already indexed; skipping` —
  idempotent resume holds across the streaming source.

## Reproducibility recipe (for the dataset card)

1. `configs/roi_tiles.json` — 160 ROI tiles (81 land) + 238 OOD, from the MGRS grid.
2. For each (land tile, year, warm season): `05b --source stac --all-dates` streams
   HLS from PC and writes the tiled Zarr. No credentials required (anonymous).
3. Labels/meteo/assembly run identically on the tiled output (M4 packaging).
4. The `(tile, date, sensor, {band: href})` manifest pins exact provenance.

No raw HLS is redistributed; anyone reproduces the pixels from public PC STAC with
the code + configs above.

## Remaining (execution, no new code)

- Cheap-source pulls 2017–2022 + 2024 (`pull_year`, full ROI, idempotent).
- Season-bounded stream-tiling over 81 land tiles × 8 yr (`05b --source stac`,
  resumable via `known_ids`); a thin loop-driver is the only optional scaffolding.
- Package → assemble → judge the §4.4 compound gate on the full dataset → M4 closes.
