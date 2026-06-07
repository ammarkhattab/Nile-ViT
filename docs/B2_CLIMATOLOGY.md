# B2 — Climatology References (decisions & PRD deltas)

This document records the design decisions for the **climatology baselines** that
feed the Script 06 label generator (PRD §4.4), and tracks every deviation from
the PRD so the change-log trail is explicit. Nothing here is a silent change.

The three label indices each need a long-term reference:

| Index | Baseline needed | Source | Script | Status |
|---|---|---|---|---|
| VHI (VCI+TCI) | per-pixel NDVI & LST min/max, 2001–2020 | MODIS MOD13Q1 / MOD11A2 (MPC) | `06a` | ✅ done |
| SPEI-3 (precip half) | monthly precip, 1991–2020 | CHIRPS v2 monthly | `06b` | ✅ done |
| SPEI-3 (PET half) | monthly PET, 1991–2020 | ERA5-Land (Hargreaves) | `06c` | building |
| Heat-z | per-day-of-year Tmax μ/σ, 1991–2020 | ERA5-Land Tmax | `06c` | building |
| SPEI-3 (combine) | per-pixel/per-month water-balance μ/σ | from `06b`+`06c` | `06d` | pending |

All baselines follow a **reduce-don't-hoard** pattern: stream the source
year-by-year, accumulate running statistics, discard the raw data, and write only
compact per-pixel reference grids to `data/climatology/`. Script 06 reprojects
these onto the HLS 30 m grid on demand (the same machinery as `05a`).

---

## Decision 1 — VHI baseline (PRD-faithful)

Per-pixel min/max of MODIS NDVI (MOD13Q1) and LST (MOD11A2) over **2001–2020**,
exactly as PRD §4.4 specifies ("both vs 2001–2020 pixel-wise min/max"). Streamed
from Microsoft Planetary Computer as COGs, resumable per year. No deviation.

## Decision 2 — SPEI method: Gaussian-standardized water balance

PRD §4.4 calls for a "standardized" precipitation–evapotranspiration index but
does not mandate the log-logistic fit of classical SPEI. We standardize the
3-month water balance (P − PET) as a **per-pixel, per-calendar-month Gaussian
z-score** against the 1991–2020 baseline, and label the output
**"SPEI-like (Gaussian)"** in the data/model cards. This is compliant with the
PRD wording and far more robust to fit instabilities on short/arid series.

## Decision 3 — SPEI precip source: CHIRPS **v2** monthly

**Deviation from PRD §B** (which lists "CHIRPS v3"). The 1991–2020 precip
climatology is built from the **CHIRPS v2 consolidated monthly** NetCDF, because:
- it is a single verified file with a stable endpoint (no v3-monthly directory
  ambiguity), and an established long-term climatology reference;
- it sits on the **same 0.05° grid as v3**, so it co-registers spatially with the
  v3 daily data already on disk;
- SPEI standardization is robust to the small v2/v3 differences.

The model's **daily `chirps_p` meteo input remains CHIRPS v3** (PRD §4.3) — that
is a separate consumer from this label baseline, so there is no version conflict
in the model inputs. `06b` accepts a `--url` override to swap in v3 monthly later.

## Decision 4 — Temperature source: ERA5-Land for **both** heat-z and PET

**Deviation from PRD §4.4** (heat-z line specifies `CHIRTS_Tmax`). We use
**ERA5-Land** Tmax/Tmin for the heat-z baseline as well as for PET. Rationale:

- **CHIRTS-ERA5 (1980→present) cannot be obtained over just the ROI.** The CHC
  server is global static files only (a 30-year *daily* Tmax+Tmin pull is
  ~200 GB), and neither Google Earth Engine nor the IRI Data Library host
  CHIRTS-**ERA5** — both carry only **CHIRTS v1, which ends 2016**, so it covers
  neither the 2017–2020 baseline tail nor the 2023 target. A 200 GB download also
  violates the PRD §10.3 storage budget (~80 GB) and the project's stated
  consumer-hardware / limited-bandwidth envelope.
- **The heat label is a per-pixel z-score, so ERA5's cool bias largely cancels.**
  `Tmax_z = (Tmax(d) − μ_climo) / σ_climo` with all terms from one product: a
  roughly constant cool offset subtracts out of the numerator. ERA5's known cool
  bias suppresses *absolute* hot-day counts, not *relative* anomalies. (Minor
  residual: ERA5 may compress the hot tail, shrinking σ and slightly inflating z
  — identical for baseline and target, so internally consistent.)
- **It keeps everything inside a PRD-named dataset.** ERA5-Land is already the
  PRD's PET source (§4.4: "ERA5-Land PET (Hargreaves on T2m)"), and B1's 2023
  data is ERA5-Land — so baseline and target are the same product on the same grid.

CHIRTS-ERA5 is left **pluggable** in `06c` for a future refinement if a
subsetting endpoint appears.

### 4a — Access method: hourly `reanalysis-era5-land`, daily stats computed locally

We pull **hourly** ERA5-Land 2m-temperature over the ROI via CDS
(`reanalysis-era5-land`, the PRD §4.2 dataset) and compute daily Tmax/Tmin as the
max/min of each day's 24 hourly values. We deliberately **avoid**
`derived-era5-land-daily-statistics`: it has a documented bug where
`daily_maximum`/`daily_minimum` return hourly arrays rather than the reduced
statistic. Computing the daily reduction ourselves is reliable and uses the exact
PRD dataset. Cost: hourly volume + CDS queue (PRD risk I6) — mitigated by ROI
subsetting, per-year chunking, and resumable skip-existing caching.

## Decision 5 — PET via Hargreaves (no extra downloads)

Monthly PET from Hargreaves: `PET = 0.0023 · Ra · (Tmean + 17.8) · √(Tmax − Tmin)`,
with extraterrestrial radiation `Ra` computed analytically from latitude and
day-of-year (no data download). Daily PET is summed to monthly totals, giving a
1991–2020 monthly PET series that combines with `06b`'s monthly precip in `06d`.

---

## Outputs inventory (`data/climatology/`)

| File | Shape | From |
|---|---|---|
| `ndvi_{min,max}_<tile>.tif`, `lst_{min,max}_<tile>.tif` | per-tile 2-D | `06a` |
| `precip_monthly_1991_2020.nc` | (time=360, lat, lon) | `06b` |
| `tmax_doy_{mean,std}.nc` | (doy=366, lat, lon) | `06c` |
| `pet_monthly_1991_2020.nc` | (time=360, lat, lon) | `06c` |
| `spei_wb_{mean,std}.nc` | (month=12, lat, lon) | `06d` |

## Change-log

| Date | Change |
|---|---|
| 2026-06 | Decisions 1–5 recorded; Decision 3 (v2 precip) and Decision 4 (ERA5-Land for heat-z) flagged as the two PRD deviations, both with rationale above. |
