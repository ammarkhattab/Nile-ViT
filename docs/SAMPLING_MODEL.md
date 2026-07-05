# Dataset sampling model (M4 decision D7)

How the ~86k-sample dataset is selected from the tile x date x patch space, why,
and how it is reproduced. This closes a gap the PRD left open: it fixes the dataset
*size* (§6.1 embedding budget) and how samples are *split* (§4.5) and *labelled*
(§4.4), but never states the *selection rule*. D7 is that rule, decided from the
data and recorded here for the paper.

## The problem the numbers exposed

Exhaustive tiling of 81 land tiles x 8 warm seasons is infeasible and wrong:
- One peak date alone (2023-08-13), 81 tiles, yielded ~7,100 kept patches — already
  above the stale "~6,000" target (which was one cloud-filtered ROI snapshot under
  the old 5-tile Delta scope, never meant to multiply across tiles/dates/years).
- Full 2023 (19 composite dates) keeping **all** event patches → **92,444/yr** →
  ~740k over 8 yr → ~570 GB embeddings, ~7x the §10.3 80 GB cap.
- The driver of the blow-up is **drought (class 1): 66,359/yr** — drought is a
  broad-area phenomenon over the Nile/Sahel margin, so most land patches are
  drought-affected on most dates. **Compound (class 3), the actual §4.4/R7 target,
  is rare: 3,045/yr.** So "keep all events" over-keeps the wrong class; background
  was never the binding constraint.

## The rule (D7)

**Temporal — one central date per (tile, MODIS-composite window).** Each M3
`label_*.tif` is one 16-day composite; a sample's central date is realised (in 05b)
as the least-cloudy HLS acquisition in that window. Without this, the HLS ~2–3 day
revisit multiplies each window ~6x. ~19 composite dates/yr over the warm season.

**Spatial / class — per-class keep probability.** A patch's class is
compound>heat>drought>none over its label window. Keep it with probability
`keep_prob[class]`:

| class | keep_prob | rationale |
|-------|-----------|-----------|
| 3 compound | **1.00** | rare target (§4.4, R7 lead-time) — keep every one |
| 2 heat | 0.40 | moderately common single-hazard negative |
| 1 drought | 0.04 | broad-area, dominates raw counts — subsample hard |
| 0 none | 0.015 | background |

The decision is a deterministic hash of `(sample_id, seed=20260519)` — reproducible
and order-independent, so the keep-list regenerates identically from public labels.

## Resulting dataset (2023 labels, extrapolated x8 yr)

- **kept/yr 10,770** — by class none 2,343 / drought 2,594 / heat 2,788 / compound
  3,045 (near-balanced four-way).
- **~86,160 samples** over 8 years → **66.3 GB** embeddings (§6.1: 770 KB/sample) —
  under the §10.3 80 GB cap.
- **Compound = 28.3% of samples** — the rare class is richly represented for
  training, while pixel-level prevalence in the *labels* is unchanged (§4.4 is judged
  on label prevalence over valid land, not on the enriched sample mix).

The four `--keep-*` rates are the tuning knobs; the `select_samples` driver reports
size + class mix + embedding GB from labels alone, before any HLS streaming.

## Why this is PRD-faithful and paper-defensible

- **Budget:** the only selection that fits §6.1 / §10.3 at 81-tile scope.
- **Science:** keeping all compound preserves the rare target for the §4.4 gate and
  the R7 lead-time analysis; balanced negatives (drought-only, heat-only, none) are
  the hard contrasts the model must learn to call compound.
- **Reproducibility:** keep-list = deterministic function of public M3 labels +
  seed + these rates; publishes alongside the STAC manifest. Anyone regenerates the
  exact sample set.
- **Honesty:** class enrichment is standard for rare-event segmentation; pixel-level
  imbalance is separately handled by the M4 class weights (`class_weights_v1.json`).

## Reproduce

```
uv run python scripts/data/select_samples.py \
    --labels-dir data/interim/labels_<year> \
    --keep-none 0.015 --keep-drought 0.04 --keep-heat 0.4 --keep-compound 1.0
```
→ `configs/keeplist_v1.json` (tile → composite-date → [[row, col, class]]) +
`configs/keeplist_v1_report.json` (size / class mix / embedding GB). 05b consumes
the keep-list so only selected patches are tiled.

## Status / caveats

- Previewed on **2023** labels (a strong Mediterranean-MHW year) and extrapolated
  x8; milder years likely yield fewer compound patches, so the real dataset may be
  slightly smaller and compound-lighter. Well within budget either way.
- The committed keep-list is provisional (2023 only); it is rebuilt once every
  year's labels exist (needs the per-year cheap-source pulls + label builds).
