# ruff: noqa: B008
"""PRD §4.4 acceptance gate: compound-class prevalence on the NATURAL labels.

§4.4: "Class-3 prevalence must be in [0.5%, 8%] of pixels on the full dataset.
If <0.5%: thresholds too strict. If >8%: too lax - re-tune."

The gate validates the LABEL DEFINITION (SPEI3 < -1.0, Tmax_z > 2.0, VHI < 35), so
it is measured over **valid land on the unselected label rasters** - the population
M3 scores in each `labels_<year>/prevalence.csv` (sea / no-data = 255 excluded).

It is NOT the compound fraction of the packaged dataset: D7 selection deliberately
keeps every compound patch and subsamples the abundant classes, so the packaged
samples run ~28% compound by design. That is a dataset statistic, not this gate.
Conflating them would read ~29% and wrongly demand re-tuning correct thresholds.

Usage:
    uv run python scripts/data/check_gate.py
    uv run python scripts/data/check_gate.py --json-out configs/gate_v1.json
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import typer

with contextlib.suppress(ImportError):
    import nilevit  # noqa: F401

app = typer.Typer(add_completion=False, help=__doc__)

GATE_LOW = 0.005
GATE_HIGH = 0.08
CLASS_NAMES = {0: "none", 1: "drought", 2: "heat", 3: "compound"}


@app.command()
def main(
    labels_root: Path = typer.Option(
        Path("data/interim"), help="Dir containing labels_<year>/prevalence.csv."
    ),
    json_out: Path | None = typer.Option(None, help="Optional path to write the report."),
) -> None:
    """Pool per-year label prevalence and judge the §4.4 compound gate."""
    import pandas as pd

    files = sorted(labels_root.glob("labels_*/prevalence.csv"))
    if not files:
        typer.echo(f"ERROR: no labels_*/prevalence.csv under {labels_root}")
        raise typer.Exit(code=1)

    per_year: dict[str, float] = {}
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frames.append(frame)
        year = path.parent.name.replace("labels_", "")
        total_y = frame["pixels"].sum()
        comp_y = frame.loc[frame["class"] == 3, "pixels"].sum()
        per_year[year] = float(comp_y / total_y) if total_y else 0.0

    pooled = pd.concat(frames).groupby("class")["pixels"].sum()
    total = int(pooled.sum())
    fractions = {int(k): float(v / total) for k, v in pooled.items()}
    compound = fractions.get(3, 0.0)
    passed = GATE_LOW <= compound <= GATE_HIGH

    typer.echo(f"§4.4 gate over {len(files)} year(s): {', '.join(sorted(per_year))}")
    for cls in sorted(fractions):
        typer.echo(
            f"  {cls} {CLASS_NAMES.get(cls, '?'):<9} "
            f"{fractions[cls]:7.3%}  ({int(pooled[cls]):,} px)"
        )
    typer.echo(f"  per-year compound: { ({y: f'{v:.3%}' for y, v in sorted(per_year.items())}) }")
    verdict = "PASS" if passed else "FAIL"
    typer.echo(
        f"\ncompound prevalence {compound:.3%} in [{GATE_LOW:.1%}, {GATE_HIGH:.0%}] -> {verdict}"
    )
    if not passed:
        hint = "thresholds too strict" if compound < GATE_LOW else "thresholds too lax"
        typer.echo(f"  {hint} - re-tune the §4.4 thresholds.")

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(
                {
                    "years": sorted(per_year),
                    "pooled_pixels": {str(k): int(v) for k, v in pooled.items()},
                    "fractions": {str(k): v for k, v in fractions.items()},
                    "per_year_compound": per_year,
                    "compound_prevalence": compound,
                    "gate": [GATE_LOW, GATE_HIGH],
                    "passed": passed,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        typer.echo(f"wrote {json_out}")

    raise typer.Exit(code=0 if passed else 1)


if __name__ == "__main__":
    app()
