# Nile-ViT — Project Status & Gap Audit

Snapshot of every milestone (PRD §13), what is **done / in-progress / blocked /
delayed / not-started**, and the dependency gating each open item. PRD rule holds:
each milestone has one **binary** acceptance and **no concurrent forward motion**
past an unmet gate.

## Milestone status

| # | Milestone | Acceptance (PRD §13) | Status | Blocking dependency |
|---|---|---|---|---|
| M0 | Repo bootstrapped | `uv sync && pytest` | ✅ done | — |
| M1 | Prithvi smoke test | burn-scars F1 within 5 pp | ✅ done | — |
| M2 | ROI data pulled | 4 sources present 2023 | ✅ done | — |
| M3 | 2023 labels | class-3 prevalence ∈ [0.5%,8%] | ✅ done (`96e2dd5`) | — |
| — | Script 07 splits | leak-check pytest passes | ✅ done (`1ef8b75`) | — |
| **M4** | **Full dataset built** | 8 yr × ROI tiled + leak-check | 🟡 **partial** | HLS 05b STAC seam + 8-yr pulls |
| M5 | HF published | `load_dataset()` works | ⛔ not started | M4 |
| M6 | Baselines | 5 baselines + W&B | ⛔ not started | M4/M5 |
| M7 | Nile-ViT v0 | macro-F1 ≥ 0.55 | ⛔ not started | M6; fusion §5.1/5.2 |
| M8 | Ablations | 4 fusion × 2 bb × 3 seeds | ⛔ not started | M7 |
| M9 | OOD + temporal | R5+R6 or documented | ⛔ not started | M7/M8; OOD tiles |
| M10 | Lead-time | R7 or pivot | ⛔ not started | M9 |
| M11 | Demo | Gradio ≤ 15 s | ⛔ not started | M7+ |
| M12 | Paper v1 | 4-page CCAI | ⛔ not started | M9/M10 |
| M13 | Submission | camera-ready + public | ⛔ not started | M12 |

## M4 — done vs. outstanding

**Done & validated (committed):**
- §4.3 schema, label packaging (design A), meteo packaging, dataset assembly +
  §4.5 leak-check gate (buffer/none excluded). Proof: 2023 T36RUU, 504 members.
- **ROI tile coverage** — `configs/roi_tiles.json`, 160 ROI + 238 OOD from the MGRS
  grid; land-pruned to **81 land ROI tiles** (79 ocean/no-data) via the M3 mask (F1).
- **HLS STAC streaming layer** — `nilevit/hls_stac.py` + `nilevit/hls_bands.py`:
  search both PC collections (`hls2-s30`/`hls2-l30`, anonymous), map bands→signed
  COG hrefs, group one-scene-per-date, build the reproducibility manifest.
  Offline-tested (8) + **live smoke test passed** (T36RUU 2023-08 → 20 streamable
  scenes). See F2.
- 62 offline tests green.

**Outstanding (network-bound):**
1. **Cheap-source pulls 2017–2022 + 2024** — `scripts/pull_year.py` × 7 (idempotent,
   full ROI). *Runnable today, no code needed.*
2. **05b `--source stac` read-path** — the one remaining pipeline change: 05b tiles
   from STAC hrefs instead of disk. Design ready (see below); needs the full 05b to
   edit the seam without disturbing the validated disk path.
3. **Full-scale tiling + packaging** over 81 tiles × 8 yr via streaming; recompute
   class weights + meteo norm.
4. **Assemble + judge the §4.4 gate** on the full dataset (compound ∈ [0.5%,8%]).

## Found issues

- **F1 — ROI coverage was Delta-only (FIXED).** 5 hand-typed tiles → 160 ROI (81
  land) from the MGRS grid; E-Med shelf (compound region) was entirely missing;
  `T35RPN` (below 30°N) wrongly included. Dep: `uv add mgrs`.
- **F2 — HLS strategy resolved to STAC streaming (§10.3).** Bulk download rejected:
  81 land tiles × 8 yr × any season = **0.9–3.7 TB raw, 12–45× over the 80 GB
  budget**. §10.3 mandates fetch-once + stream via Planetary-Computer STAC. 05b will
  read bands on demand (`--source stac`); raw scenes never persisted; the ~80 GB
  becomes the tiled-Zarr cache. The `(tile,date,sensor,{band:href})` manifest is the
  NeurIPS reproducibility artifact. Collections + asset codes confirmed via the live
  probe; anonymous signing works (PC_SDK_SUBSCRIPTION_KEY optional, lifts rate limit
  for the full pull).

## 05b `--source stac` — design (ready to build once full 05b in hand)

Disk-dependence is localized to 3 seam points; the patch loop (grid, cloud/valid
fractions, transformer, Zarr/Parquet append, `known_ids` idempotency) is
source-agnostic and untouched:
1. `hls_dir.exists()` guard — skipped when `source=stac`.
2. Date enumeration — disk glob `*.B04.tif` vs. `hls_stac.search_hls_items` +
   `hrefs_by_date` → `{date: {sensor, hrefs}}`.
3. Per-date band open — disk `band_path`+`open_rasterio(local)` vs.
   `open_rasterio(hrefs[band])` (signed COG; drop-in — same `.rio.crs/.x/.y/.values`).
Also: 05b switches its band-map import to `nilevit.hls_bands` (removes the duplicate;
drift-guard test already protects it). New flags: `--source {disk,stac}` (default
disk), `--cloud-max` (STAC only). Validation: tile one real date via STAC on
Ammar's machine, confirm the Zarr matches the disk-built proof.

## Deferrals / deviations (safe)

- Nov/Dec 2022 backfill (4 winter-2023 label dates) — optional, low priority.
- B2: CHIRPS v2-monthly, ERA5-Land for heat/chirts channels (z-score cancels bias).
- M4-D1 sidecar Zarr stores; merge-vs-sidecar decided at full assembly (M5 shards).
- One 2023 land reference raster is valid for all years (sea/land static).

## Kill-criteria (PRD §13.1)

M3 passed → no subsidence pivot. No baseline ≥ 0.55 (M7) → dataset+baselines paper.
Cross-attn < late-fusion + 3 pp → FiLM fallback. M9 OOD fails → in-distribution only.
Lead-time (5–14 d before indices cross) is the scientific hook (R7).

## Immediate next actions

1. Run cheap-source pulls (today): `pull_year` 2017–2022 + 2024.
2. Build 05b `--source stac` seam (need full 05b), validate one date vs. proof.
3. Stream-tile 81 tiles × 8 yr; package; assemble; **judge §4.4 gate**.
4. Optional: Nov/Dec 2022 backfill. Then M5.
