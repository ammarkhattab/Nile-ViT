# Nile-ViT — Project Status & Gap Audit

Snapshot of every milestone (PRD §13), what is **done / in-progress / blocked /
delayed / not-started**, and the explicit dependency that gates each open item.
The PRD rule holds throughout: each milestone has one **binary** acceptance and
there is **no concurrent forward motion** past a failed/unmet gate.

## Milestone status

| # | Milestone | Acceptance (PRD §13) | Status | Blocking dependency |
|---|---|---|---|---|
| M0 | Repo bootstrapped | `uv sync && pytest` on fresh box | ✅ done | — |
| M1 | Prithvi smoke test | TerraTorch burn-scars demo, F1 within 5 pp | ✅ done | — |
| M2 | ROI data pulled | 4 sources present for 2023 in `data/raw/` | ✅ done | — |
| M3 | 2023 labels | class-3 prevalence ∈ [0.5%, 8%] + Aug sanity | ✅ done (`96e2dd5`) | — |
| — | Script 07 splits | leak-check pytest passes (cell-map + temporal) | ✅ done (`1ef8b75`) | — |
| **M4** | **Full dataset built** | 8 yr × full ROI tiled, labels, splits, **leak-check passes** | 🟡 **partial** | HLS streaming/cache + 8-yr pulls (network) |
| M5 | HF dataset published | `load_dataset(...)` works | ⛔ not started | M4 complete |
| M6 | Baselines table | 5 baselines trained + W&B + README | ⛔ not started | M4/M5; TerraTorch harness |
| M7 | Nile-ViT v0 | macro-F1 ≥ 0.55 (any fusion) | ⛔ not started | M6; fusion module (§5.1/§5.2) |
| M8 | Ablations | 4 fusion × 2 backbones × 3 seeds + CIs | ⛔ not started | M7 |
| M9 | OOD + temporal | R5 + R6 thresholds OR documented | ⛔ not started | M7/M8; OOD HLS tiles |
| M10 | Lead-time | R7 threshold OR pivot narrative | ⛔ not started | M9 |
| M11 | Demo live | Gradio Space, (date,lat,lon) → map ≤ 15 s | ⛔ not started | M7+ |
| M12 | Paper draft v1 | 4-page CCAI LaTeX, figures, review | ⛔ not started | M9/M10 |
| M13 | Submission ready | camera-ready + public code/data/space | ⛔ not started | M12 |

## M4 — exactly what is done vs. outstanding

**Done and validated on 2023 (T36RUU proof), all committed:**
- §4.3 `TileSample` schema + region map (`nilevit/schemas.py`).
- Per-tile **label** packaging, design A (`nilevit/tiles.py`,
  `scripts/labels/package_tile_labels.py`); out-of-ROI drop; class-weight recompute.
- Per-tile **meteo** packaging (`nilevit/meteo.py`,
  `scripts/data/package_tile_meteo.py`); ERA5 daily aggregation grounded in the raw
  files; train-only z-score stats.
- **Dataset assembly + acceptance** (`nilevit/dataset.py`,
  `scripts/data/assemble_dataset.py`); excludes buffer/none as non-members; §4.5
  spatial leak-check; exit-code-gated.
- **ROI tile coverage** (`nilevit/roi_tiles.py`, `scripts/data/make_roi_tiles.py`
  → `configs/roi_tiles.json`): 160 ROI + 238 OOD tiles enumerated from the MGRS
  grid (see found issue F1).
- 55 offline tests green; lint/format clean.

**Outstanding (all network-bound, cannot be sandbox-validated):**
1. **Cheap-source pulls 2017–2022 + 2024** — `scripts/pull_year.py` × 7 (drives
   02/03/04; ERA5/CHIRPS/MODIS, full ROI, idempotent skip-existing). *Status: not
   run — runnable today, no code needed.* CDS queue waits (risk I6).
2. **HLS streaming/cache 2017–2024** — over `configs/roi_tiles.json` × the ROI.
   *Status: needs a resumable multi-tile driver.* `01_download_hls.py` exists but
   (a) had a Delta-only hardcoded tile list — now superseded by `roi_tiles.json` —
   and (b) lacks skip-existing / backoff / resume (risk I3). Per §10.3, HLS is
   fetched **once** and cached (~80 GB) or streamed via Planetary-Computer STAC;
   *"do not re-download raw HLS more than once."* The driver is the next build, but
   needs the bulk-vs-STAC-stream decision + 05b's date-selection confirmed first.
3. **Full-scale packaging re-run** — `package_tile_labels` + `package_tile_meteo`
   over every tile-year; recompute class weights + meteo norm. *Code ready, awaits
   data.*
4. **Assemble + judge the §4.4 gate on the full dataset** — `assemble_dataset.py`
   then verify compound prevalence ∈ [0.5%, 8%]. *Tool ready and exit-code-gated;
   gate unjudgeable until E-Med shelf/coast tiles are populated (the 2023 Delta
   proof has 0% compound — correct geography). The spatial leak-check is trivially
   clean now (no test tiles) and becomes load-bearing once test cells are tiled.*

## Found issues (this session)

- **F1 — ROI tile coverage was Delta-only (FIXED).** `KNOWN_TILES` in
  `01_download_hls.py` listed **5 hand-typed Nile-Delta tiles** out of the **160**
  that intersect the §4.1 ROI, and it **omitted the entire Eastern-Mediterranean**
  shelf/coast — exactly where the §4.4 compound signal concentrates. It also
  *included* `T35RPN`, which sits below the ROI's 30°N edge. Had this driven the
  full HLS pull, M4 would have tiled only the compound-poor Delta and **failed the
  §4.4 gate for a fixable reason.** Now superseded by `configs/roi_tiles.json` (160
  ROI + 238 OOD), enumerated from the MGRS grid and unit-tested. **Action:**
  `01_download_hls.py` should read `roi_tiles.json` instead of `KNOWN_TILES` (small
  follow-up wiring change). **New dependency:** `uv add mgrs`.

## Known deferrals / tracked deviations (safe, reversible)

- **D-M3-5: Nov/Dec 2022 backfill** — 4 winter-2023 label dates skipped (SPEI-3
  window). Optional; pull 2 small months. *Delayed, low priority.*
- **B2 deviations:** CHIRPS **v2**-monthly (not v3) for the SPEI precip baseline;
  **ERA5-Land** (not CHIRTS) for heat-z and the `chirts_*` meteo channels. In
  `docs/B2_CLIMATOLOGY.md`; z-score cancels the bias.
- **M4-D1: sidecar Zarr stores.** Non-destructive; loader resolves
  `sample_id`→variable; `image_path` derived at assembly. **Open for the full run:**
  merge vs. keep-sidecars (affects M5 HF shard layout).
- **M4-D6: buffer/none excluded from the index** as §4.5 spacing.
- **Ocean tiles:** `roi_tiles.json` is a geometric superset; ocean-dominated tiles
  yield no valid land samples and are pruned by the M3 land mask. A coastline-based
  pre-filter could trim the HLS download budget — *deferred, optimisation only.*

## Not-yet-designed (future-milestone scope, deliberately not built early)

- **HLS resumable driver** — next M4 build; pending bulk-vs-STAC-stream decision +
  05b date-selection review (so it doesn't contradict §10.3 streaming intent).
- **Dataset loader / DataModule** — M6/M7; couples to TerraTorch
  `SemanticSegmentationTask` + custom fusion. Not built (no concurrent forward
  motion past M4).
- **Fusion module** (`nilevit/models/fusion.py`, §5.1/§5.2) — M7.
- **Baseline models** — M6. **HF publish** (`08_publish_hf_dataset.py`) — M5.

## Kill-criteria reminders (PRD §13.1)

- M3 passed → no pivot to subsidence.
- No baseline ≥ 0.55 macro-F1 (M7) → "dataset + baselines paper" (NeurIPS D&B).
- Cross-attention < late-fusion + 3 pp compound-F1 → fall back to FiLM, reframe.
- M9 OOD fails → in-distribution only, document honestly.
- Lead-time (events 5–14 days before indices cross) is the scientific hook (R7).

## Immediate next actions (in order)

1. **Run cheap-source pulls** (runnable today): `pull_year` for 2017–2022 + 2024.
2. **Wire `01_download_hls.py` to `roi_tiles.json`** + add the resumable driver
   (next build; confirm bulk-vs-stream first).
3. As tiles land, run packaging + `assemble_dataset.py`; **judge the §4.4 gate**.
4. (Optional warm-up) Nov/Dec 2022 backfill.
5. Only after M4 passes: M5 (HF publish), then M6 (baselines).
