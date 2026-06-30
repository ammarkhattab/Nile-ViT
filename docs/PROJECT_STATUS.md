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
| **M4** | **Full dataset built** | 8 yr × full ROI tiled, labels, splits, **leak-check passes** | 🟡 **partial** | HLS streaming + 8-yr pulls (network) |
| M5 | HF dataset published | `load_dataset(...)` works | ⛔ not started | M4 complete |
| M6 | Baselines table | 5 baselines trained + W&B + README | ⛔ not started | M4/M5; TerraTorch harness |
| M7 | Nile-ViT v0 | macro-F1 ≥ 0.55 (any fusion) | ⛔ not started | M6; fusion module (§5.1/§5.2) |
| M8 | Ablations | 4 fusion × 2 backbones × 3 seeds + CIs | ⛔ not started | M7 |
| M9 | OOD + temporal | R5 + R6 thresholds OR documented | ⛔ not started | M7/M8 |
| M10 | Lead-time | R7 threshold OR pivot narrative | ⛔ not started | M9 |
| M11 | Demo live | Gradio Space, (date,lat,lon) → map ≤ 15 s | ⛔ not started | M7+ |
| M12 | Paper draft v1 | 4-page CCAI LaTeX, figures, review | ⛔ not started | M9/M10 |
| M13 | Submission ready | camera-ready + public code/data/space | ⛔ not started | M12 |

## M4 — exactly what is done vs. outstanding

**Done and validated on 2023 (T36RUU proof), all committed:**
- §4.3 `TileSample` schema + region map (`nilevit/schemas.py`).
- Per-tile **label** packaging, design A (`nilevit/tiles.py`,
  `scripts/labels/package_tile_labels.py`) — resamples M3 ROI labels onto real tile
  grids; out-of-ROI drop; class-weight recompute.
- Per-tile **meteo** packaging (`nilevit/meteo.py`,
  `scripts/data/package_tile_meteo.py`) — 90×7 series, ERA5 daily aggregation
  grounded in the raw files, train-only z-score stats.
- **Dataset assembly + acceptance** (`nilevit/dataset.py`,
  `scripts/data/assemble_dataset.py`) — consolidates sidecars, **excludes
  buffer/none tiles as non-members** (§4.5 spacing, keyed off `get_args(Split)`),
  validates every member as a TileSample, runs the §4.5 spatial leak-check (the M4
  binary acceptance), and exit-code-gates the result.
- 51 offline tests green; lint/format clean.
- Artifacts on disk (2023 proof): `tiles_T36RUU_2023_labels.zarr`,
  `tiles_T36RUU_2023_meteo.zarr`, `tiles_T36RUU_2023_labeled.parquet`,
  `data/processed/dataset_v1.parquet` (504 members, 888 buffer excluded),
  `configs/class_weights_v1.json`, `configs/meteo_norm_v1.json`.

**Outstanding (all network-bound, cannot be sandbox-validated):**
1. **Cheap-source pulls 2017–2022 + 2024** — `pull_year` × 7 (ERA5/CHIRPS/MODIS).
   *Status: not run.* CDS queue waits + month-by-month requests (handoff §4).
2. **HLS streaming 2017–2024** — `01_download_hls.py` (deferred since M1) → `05a`
   harmonize → `05b` tile, for **all ROI MGRS tiles**. *Status: 01 exists but unrun
   at scale; only the T36RUU/2023 August proof tile exists.* The single biggest
   blocker and the project's main I/O risk (`/vsicurl` hardening already paid for).
3. **Full-scale packaging re-run** — `package_tile_labels` + `package_tile_meteo`
   over every tile-year; recompute `class_weights_v1.json` + `meteo_norm_v1.json`.
   *Status: code ready, awaits data.*
4. **Assemble + judge the §4.4 gate on the full dataset** — `assemble_dataset.py`
   then verify compound prevalence ∈ [0.5%, 8%]. *Status: tool ready and
   exit-code-gated; gate unjudgeable until compound-bearing regions (E-Med
   shelf/coast) are tiled — the 2023 delta proof has 0% compound (correct geography).
   The spatial leak-check is trivially clean now (no test tiles) and becomes
   load-bearing once test cells are populated.*

## Known deferrals / tracked deviations (safe, reversible)

- **D-M3-5: Nov/Dec 2022 backfill** — 4 winter-2023 label dates skipped (SPEI-3
  window). Optional; pull 2 small months to fill. *Delayed, low priority.*
- **B2 deviations:** CHIRPS **v2**-monthly (not v3) for the SPEI precip baseline;
  **ERA5-Land** (not CHIRTS) for heat-z and the `chirts_*` meteo channels. Both in
  `docs/B2_CLIMATOLOGY.md`; z-score cancels the bias.
- **M4-D1: sidecar Zarr stores** (`_labels.zarr`, `_meteo.zarr`) rather than one
  store with explicit path columns. Non-destructive; loader resolves
  `sample_id`→variable; `image_path` derived at assembly. **Decision still open for
  the full run:** merge into one canonical store vs. keep sidecars — settle at
  full-dataset assembly (affects M5 HF WebDataset shard layout).
- **M4-D6: buffer/none excluded from the index.** Buffer = §4.5 leakage spacing,
  not a `TileSample`; `assemble_dataset` drops them and reports the counts.

## Not-yet-designed (future-milestone scope, deliberately not built early)

- **Dataset loader / DataModule** — the §4.3 record is packageable and
  assembly-validated, but the PyTorch/TerraTorch loader that yields
  `(x_img[B,3,6,224,224], x_met[B,90,7], mask)` is **M6/M7** and couples to the
  TerraTorch `SemanticSegmentationTask` + custom fusion. Not built, to avoid
  front-running M4's gate (PRD: no concurrent forward motion).
- **Fusion module** (`nilevit/models/fusion.py`, §5.1/§5.2 cross-attention + meteo
  encoder + LoRA) — M7.
- **Baseline models** (ResNet-50, ViT-S/16, LSTM-only, Prithvi linear-probe,
  late-fusion MLP, RF) — M6.
- **HF publish** (`08_publish_hf_dataset.py`) — M5; WebDataset shard mirror.

## Kill-criteria reminders (PRD §13.1)

- M3 passed, so no pivot to subsidence.
- If **no baseline ≥ 0.55 macro-F1** (M7) → label definition suspect; reduce scope
  to "dataset + baselines paper" (NeurIPS D&B).
- If **cross-attention doesn't beat late-fusion by ≥ 3 pp compound-F1** → fall back
  to FiLM and reframe the dataset as the contribution.
- If **M9 OOD fails** → keep the paper in-distribution only, document honestly.
- The **lead-time result** (compound events 5–14 days before indices cross) is the
  scientific hook (R7).

## Immediate next actions (in order)

1. **Run the network-bound M4 inputs** (user-driven, long-running): `pull_year` for
   2017–2022 + 2024, then HLS streaming → 05a/05b for the ROI MGRS tiles. Code is
   ready; this is queue/credential/bandwidth work, not new construction.
2. As tiles land, run packaging + `assemble_dataset.py`; **judge the §4.4 compound
   gate on the full dataset** (M4 acceptance).
3. (Optional warm-up) Nov/Dec 2022 backfill for the 4 winter-2023 label dates.
4. Only after M4 passes: M5 (HF publish), then M6 (baselines).
