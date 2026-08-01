# Nile-ViT — working context for Claude Code

Multimodal ViT detecting compound drought-heat events over the Nile Delta + Eastern
Mediterranean. Frozen Prithvi-EO-2.0-300M + LoRA, cross-attention fusion with a meteo
Transformer, UPerNet head, 4-class compound-extreme segmentation. Target: NeurIPS
Datasets & Benchmarks. Windows/PowerShell, Python 3.11 via `uv`, repo `F:\Nile-ViT`,
GitHub `ammarkhattab/Nile-ViT`.

## Read these first (authoritative state)
- `docs/PROJECT_STATUS.md`  — milestone gap-audit (M0–M13), what's done/blocked.
- `docs/M4_PACKAGING.md`    — dataset packaging decisions D1–D6.
- `docs/SAMPLING_MODEL.md`  — D7: how ~86k samples are selected (per-class keep rates).
- `docs/HLS_STREAMING.md`   — HLS via Planetary-Computer STAC streaming (F2).
- PRD: `Nile-ViT_PRD_and_Implementation_Plan.md` (§4.3 schema, §4.4 gate, §4.5 splits,
  §6.1 embedding budget, §10.3 storage).

## Hard rules
- Milestones have BINARY acceptance gates; NO forward motion past an unmet gate.
- Lint/format: `ruff` at **line-length 100** with select `E,F,I,B,SIM,UP,C4,RET,PERF,N,RUF`.
- Scripts: typer CLIs, `# ruff: noqa: B008` header, lazy heavy imports, idempotent/resumable.
- Validate on real data before calling anything done. Run `uv run pytest` (84 tests currently).
- HLS is STREAMED from PC STAC, never bulk-downloaded (§10.3). Raw scenes not persisted.

## Where M4 stands (all code built + committed; 84 tests green)
Pipeline complete and validated end-to-end:
enumerate ROI tiles (`nilevit/roi_tiles.py`, 160 ROI / 81 land) → land-prune via M3
label mask → STAC stream (`nilevit/hls_stac.py`) → one-scene-per-composite-window +
per-class D7 selection (`nilevit/select.py`) → tile (`scripts/data/05b_tile.py`
`--source stac --labels-dir`) → resumable orchestration (`scripts/data/tile_dataset.py`).

Selection defaults (per-class keep): none 0.015 / drought 0.04 / heat 0.4 / compound 1.0
→ ~86k samples / 66 GB embeddings, compound ~28% (see SAMPLING_MODEL.md).

## What's DONE (execution)
- 2022 cheap sources fully pulled (ERA5 12/12 + CHIRPS/MODIS 36/36).
- 2023 ROI tiling: 72/81 land tiles tiled via `tile_dataset --years 2023`.
  (9 "failures" = T37R** edge/desert tiles with no HLS coverage — benign; they just
  produce no samples. Consider marking these "skipped" not "failed" in tile_dataset.)

## IMMEDIATE NEXT STEP (the current blocker)
The 72 tiled 2023 tiles are TILED but not PACKAGED, so `assemble_dataset.py` still
reports only the old T36RUU proof (504 members from 1 sidecar). To get the first real
§4.4 compound-prevalence gate read:
1. Package all 72 tiles: run `scripts/labels/package_tile_labels.py` +
   `scripts/data/package_tile_meteo.py` over every `data/interim/tiles_*_2023.parquet`.
   These are per-tile — a `package_dataset` DRIVER (analogue of `tile_dataset.py`)
   should be built to batch them resumably. (Was about to build this; needs the two
   package scripts' CLI signatures.)
2. `uv run python scripts/data/assemble_dataset.py` → judge §4.4 prevalence ∈ [0.5%, 8%]
   on the full 2023 ROI (E-Med shelf tiles carry the compound signal; the Delta proof
   showed 0% which is correct-but-unrepresentative).

## Then, to close M4 (pure execution)
- Pull cheap sources 2017–2021, 2024 (`scripts/pull_year.py`; ERA5 CDS queue is slow).
- Build labels per year (M3 label pipeline).
- `tile_dataset --years 2017-2024` (resumable via manifest + 05b known_ids).
- Package → assemble → §4.4 gate → M4 closes → M5 (HF publish).

## Workflow note
Previously edits were staged to a sandbox and hand-moved; in Claude Code you edit
`F:\Nile-ViT` directly, run `uv run pytest` / scripts yourself, and read output directly.
Commit style: `git add <files>; git commit -m "M4: ..."` after tests pass and pre-commit
(ruff/ruff-format) is clean. Pre-commit's ruff-format may reformat on first pass → then
`git add -A` and re-commit.
