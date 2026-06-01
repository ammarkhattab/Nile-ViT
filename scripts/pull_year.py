"""B1 driver - pull a full year of the cheap sources across the ROI.

Orchestrates the per-month downloads needed for the label path (Script 06 / M3):
CHIRPS precip, ERA5-Land meteo, and MODIS NDVI + LST. These cover the whole ROI
and total only a few hundred MB per year - unlike HLS, which the PRD streams /
caches selectively at training time (M4), not in bulk here.

It calls the existing download CLIs as subprocesses (same venv), one per
(month, source). Each underlying script is idempotent - it skips files already
on disk - so this driver is fully resumable: re-run it and it picks up where it
left off. A failure in one (month, source) is logged and the run continues; the
final summary shows a month x source status matrix.

HLS is intentionally NOT included (see PRD 10.3 + the B1 plan).

Usage
-----
Preview the plan (no downloads):
  uv run python scripts/pull_year.py --year 2023 --dry-run

Pull all of 2023 (cheap sources):
  uv run python scripts/pull_year.py --year 2023

Resume a sub-range / subset of sources:
  uv run python scripts/pull_year.py --year 2023 --start-month 7 --end-month 9
  uv run python scripts/pull_year.py --year 2023 --sources chirps,era5
"""

# typer's standard pattern uses function calls in argument defaults.
# ruff: noqa: B008

from __future__ import annotations

import contextlib
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import typer
from rich.console import Console
from rich.table import Table

with contextlib.suppress(ImportError):
    import nilevit  # noqa: F401

app = typer.Typer(
    add_completion=False,
    help="Pull a full year of CHIRPS + ERA5 + MODIS across the ROI (B1 driver).",
)
console = Console()


class Step(NamedTuple):
    """One download invocation: a label, the data script, and fixed extra args."""

    key: str  # short selector, e.g. "era5", "modis_ndvi"
    label: str  # human label for the summary
    script: str  # filename inside scripts/data/
    extra: tuple[str, ...]  # args appended after `--month YYYY-MM`


# The cheap sources, in pull order. MODIS runs once per product.
SOURCE_STEPS: tuple[Step, ...] = (
    Step("era5", "ERA5-Land", "02_download_era5.py", ()),
    Step("chirps", "CHIRPS v3", "03_download_chirps.py", ()),
    Step("modis_ndvi", "MODIS NDVI", "04_download_modis.py", ("--product", "MOD13Q1")),
    Step("modis_lst", "MODIS LST", "04_download_modis.py", ("--product", "MOD11A2")),
)


def iter_months(year: int, start: int = 1, end: int = 12) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings for a year and month range."""
    if not (1 <= start <= 12 and 1 <= end <= 12):
        raise ValueError("months must be 1..12")
    if start > end:
        raise ValueError("start month must be <= end month")
    return [f"{year:04d}-{m:02d}" for m in range(start, end + 1)]


def select_steps(steps: tuple[Step, ...], sources: str | None) -> list[Step]:
    """Filter steps by a comma-separated --sources string. None -> all."""
    if not sources:
        return list(steps)
    wanted = {s.strip().lower() for s in sources.split(",") if s.strip()}
    return [st for st in steps if st.key in wanted]


def build_command(python_exe: str, data_dir: Path, step: Step, ym: str) -> list[str]:
    """Construct the subprocess argv for one (step, month)."""
    return [python_exe, str(data_dir / step.script), "--month", ym, *step.extra]


@app.command()
def main(
    year: int = typer.Option(2023, "--year", "-y", help="Year to pull."),
    start_month: int = typer.Option(1, "--start-month", help="First month (1-12)."),
    end_month: int = typer.Option(12, "--end-month", help="Last month (1-12)."),
    sources: str | None = typer.Option(
        None,
        "--sources",
        help="Comma-separated subset: era5,chirps,modis_ndvi,modis_lst (default all).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the planned commands; run nothing."
    ),
    stop_on_fail: bool = typer.Option(
        False, "--stop-on-fail", help="Abort the whole run on the first failure."
    ),
) -> None:
    """Pull CHIRPS + ERA5 + MODIS for a year, month by month, resumably."""
    try:
        months = iter_months(year, start_month, end_month)
    except ValueError as exc:
        console.print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=2) from exc

    steps = select_steps(SOURCE_STEPS, sources)
    if not steps:
        console.print(f"[red]ERROR[/red] no sources matched {sources!r}.")
        raise typer.Exit(code=2)

    data_dir = Path(__file__).resolve().parent / "data"
    missing = [st.script for st in steps if not (data_dir / st.script).exists()]
    if missing:
        console.print(f"[red]ERROR[/red] missing data scripts: {missing} in {data_dir}")
        raise typer.Exit(code=1)

    plan = Table(title="B1 full-year pull plan", show_header=False, title_style="bold cyan")
    plan.add_column(style="dim")
    plan.add_column(style="bold")
    plan.add_row("Year", str(year))
    plan.add_row("Months", f"{months[0]} .. {months[-1]} ({len(months)})")
    plan.add_row("Sources", ", ".join(st.label for st in steps))
    plan.add_row("Total steps", str(len(months) * len(steps)))
    plan.add_row("Mode", "dry-run" if dry_run else "run")
    plan.add_row("On failure", "stop" if stop_on_fail else "continue")
    console.print(plan)
    console.print()

    if dry_run:
        console.print("[bold]Planned commands:[/bold]")
        for ym in months:
            for st in steps:
                cmd = build_command(sys.executable, data_dir, st, ym)
                console.print(f"  [{ym}] {st.label}: {' '.join(cmd[1:])}")
        console.print("\n[yellow]Dry run - nothing executed.[/yellow]")
        raise typer.Exit(code=0)

    # status[(ym, step.key)] = "ok" | "fail"
    status: dict[tuple[str, str], str] = {}
    n_ok, n_fail = 0, 0
    for ym in months:
        for st in steps:
            console.rule(f"[bold]{ym} - {st.label}")
            cmd = build_command(sys.executable, data_dir, st, ym)
            result = subprocess.run(cmd, check=False)
            ok = result.returncode == 0
            status[ym, st.key] = "ok" if ok else "fail"
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                console.print(f"[red]FAILED[/red] {ym} {st.label} (exit {result.returncode})")
                if stop_on_fail:
                    console.print("[red]--stop-on-fail set; aborting.[/red]")
                    _print_summary(months, steps, status)
                    raise typer.Exit(code=1)

    _print_summary(months, steps, status)
    console.print(f"\n[green]Completed: {n_ok} ok, {n_fail} failed.[/green]")
    if n_fail:
        console.print(
            "[yellow]Some steps failed. Re-run the same command to retry - "
            "completed months are skipped automatically.[/yellow]"
        )
        raise typer.Exit(code=1)


def _print_summary(
    months: list[str], steps: list[Step], status: dict[tuple[str, str], str]
) -> None:
    """Render a month x source status matrix."""
    table = Table(title="Pull summary", title_style="bold cyan")
    table.add_column("Month", style="bold")
    for st in steps:
        table.add_column(st.label, justify="center")
    for ym in months:
        cells = []
        for st in steps:
            s = status.get((ym, st.key))
            if s == "ok":
                cells.append("[green]ok[/green]")
            elif s == "fail":
                cells.append("[red]fail[/red]")
            else:
                cells.append("[dim]-[/dim]")
        table.add_row(ym, *cells)
    console.print()
    console.print(table)


if __name__ == "__main__":
    app()
