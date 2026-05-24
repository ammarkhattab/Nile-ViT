"""Nile-ViT command-line interface.

Usage:
    nilevit version
    nilevit info
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.table import Table

from nilevit import __version__

app = typer.Typer(
    name="nilevit",
    help="Nile-ViT: multimodal compound climate-extreme detection.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Print the installed Nile-ViT version."""
    console.print(f"nilevit {__version__}")


@app.command()
def info() -> None:
    """Print environment and dependency info."""
    table = Table(title="Nile-ViT environment", show_lines=False)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("nilevit", __version__)
    table.add_row("python", sys.version.split()[0])
    table.add_row("platform", sys.platform)

    # Optional imports — report only if present.
    for pkg in ("torch", "terratorch", "peft", "transformers", "rioxarray"):
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            table.add_row(pkg, ver)
        except ImportError:
            table.add_row(pkg, "[red]not installed[/red]")

    console.print(table)


if __name__ == "__main__":
    app()
