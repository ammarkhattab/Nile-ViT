"""Smoke tests — verify the package is importable and CLI works.

These tests deliberately avoid importing torch / terratorch / geospatial
deps so they run in seconds and form the CI fast-path.
"""

from __future__ import annotations

import subprocess
import sys


def test_import_nilevit() -> None:
    """The top-level package imports and has a version."""
    import nilevit

    assert nilevit.__version__
    assert isinstance(nilevit.__version__, str)
    # SemVer-ish: at least one dot
    assert "." in nilevit.__version__


def test_subpackages_importable() -> None:
    """All declared subpackages import without error."""
    import nilevit.data
    import nilevit.eval
    import nilevit.labels
    import nilevit.models
    import nilevit.train  # noqa: F401


def test_cli_version() -> None:
    """`nilevit version` returns 0 and prints the version string."""
    result = subprocess.run(
        [sys.executable, "-m", "nilevit.cli", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "nilevit" in result.stdout
