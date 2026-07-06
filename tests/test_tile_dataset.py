"""Offline tests for the tile_dataset stream-tiling driver."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "data" / "tile_dataset.py"


def _load():
    spec = importlib.util.spec_from_file_location("tile_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_years() -> None:
    m = _load()
    assert m.parse_years("2023") == [2023]
    assert m.parse_years("2017-2020") == [2017, 2018, 2019, 2020]
    with pytest.raises(ValueError, match="start must be"):
        m.parse_years("2024-2020")


def test_build_command_has_selection_flags() -> None:
    m = _load()
    cmd = m.build_command(
        "py",
        Path("05b.py"),
        "T36RUU",
        2023,
        Path("labels"),
        Path("out"),
        40.0,
        {0: 0.015, 1: 0.04, 2: 0.4, 3: 1.0},
    )
    assert "--source" in cmd and "stac" in cmd
    assert "--all-dates" in cmd
    assert cmd[cmd.index("--keep-compound") + 1] == "1.0"
    assert cmd[cmd.index("--tile") + 1] == "T36RUU"


def test_dry_run_skips_years_without_labels(tmp_path) -> None:
    m = _load()
    roi = tmp_path / "roi.json"
    roi.write_text(json.dumps({"land_roi_tiles": ["T_A", "T_B"]}))
    (tmp_path / "labels_2023").mkdir()  # only 2023 has labels

    result = CliRunner().invoke(
        m.app,
        [
            "--years",
            "2022-2023",
            "--roi-tiles",
            str(roi),
            "--labels-root",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "man.json"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    # 2022 has no labels -> skipped; 2023 x 2 tiles = 2 work items.
    assert "No labels" in result.output
    assert "To tile" in result.output


def test_manifest_skips_completed(tmp_path) -> None:
    m = _load()
    man = tmp_path / "man.json"
    m._save_manifest(man, {("T_A", 2023)})
    loaded = m._load_manifest(man)
    assert ("T_A", 2023) in loaded
