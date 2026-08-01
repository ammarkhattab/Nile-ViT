"""Offline tests for check_gate.py (§4.4) and package_dataset.py (batch + aggregate)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from typer.testing import CliRunner

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "data"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prevalence(path: Path, counts: dict[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(counts.values())
    lines = ["class,pixels,fraction"]
    lines += [f"{c},{n},{n / total}" for c, n in sorted(counts.items())]
    path.write_text("\n".join(lines) + "\n")


def test_check_gate_passes_on_real_2023_shape(tmp_path) -> None:
    m = _load("check_gate")
    # The real 2023 numbers: compound 1550 / 207689 = 0.746%.
    _prevalence(
        tmp_path / "labels_2023" / "prevalence.csv", {0: 136269, 1: 63254, 2: 6616, 3: 1550}
    )
    result = CliRunner().invoke(m.app, ["--labels-root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output
    assert "0.746%" in result.output


def test_check_gate_fails_when_too_lax(tmp_path) -> None:
    m = _load("check_gate")
    _prevalence(tmp_path / "labels_2023" / "prevalence.csv", {0: 50, 1: 50, 2: 50, 3: 850})
    result = CliRunner().invoke(m.app, ["--labels-root", str(tmp_path)])
    assert result.exit_code == 1
    assert "FAIL" in result.output
    assert "too lax" in result.output


def test_check_gate_pools_multiple_years(tmp_path) -> None:
    m = _load("check_gate")
    _prevalence(tmp_path / "labels_2022" / "prevalence.csv", {0: 900, 1: 90, 2: 5, 3: 5})
    _prevalence(tmp_path / "labels_2023" / "prevalence.csv", {0: 900, 1: 90, 2: 5, 3: 5})
    out = tmp_path / "gate.json"
    result = CliRunner().invoke(m.app, ["--labels-root", str(tmp_path), "--json-out", str(out)])
    assert result.exit_code == 0, result.output
    doc = json.loads(out.read_text())
    assert doc["years"] == ["2022", "2023"]
    assert doc["passed"] is True
    assert abs(doc["compound_prevalence"] - 0.005) < 1e-9


def test_tiled_inputs_skips_labeled_sidecars(tmp_path) -> None:
    m = _load("package_dataset")
    for name in (
        "tiles_T36SXA_2023.parquet",
        "tiles_T36SYB_2023.parquet",
        "tiles_T36SXA_2023_labeled.parquet",  # must be ignored
        "tiles_T36SXA_2022.parquet",  # wrong year
    ):
        (tmp_path / name).write_bytes(b"")
    found = m.tiled_inputs(tmp_path, 2023, None)
    assert [t for t, _ in found] == ["T36SXA", "T36SYB"]
    subset = m.tiled_inputs(tmp_path, 2023, {"T36SYB"})
    assert [t for t, _ in subset] == ["T36SYB"]


def test_aggregate_class_weights_pools_histograms(tmp_path) -> None:
    m = _load("package_dataset")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "weights_T1_2023.json").write_text(
        json.dumps({"counts": {"0": 100, "1": 10, "2": 5, "3": 5}})
    )
    (scratch / "weights_T2_2023.json").write_text(
        json.dumps({"counts": {"0": 100, "1": 10, "2": 5, "3": 5}})
    )
    out = tmp_path / "weights.json"
    doc = m.aggregate_class_weights(scratch, 2023, out)
    assert doc["n_tiles"] == 2
    assert doc["counts"] == {0: 200, 1: 20, 2: 10, 3: 10}
    assert len(doc["class_weights_median_freq"]) == 4
    # The enrichment stat must be labelled as NOT the gate.
    assert "not the §4.4 gate" in doc["note"].lower()
    assert json.loads(out.read_text())["n_tiles"] == 2


def test_labels_and_meteo_commands_use_scratch_paths() -> None:
    m = _load("package_dataset")
    lab = m.labels_cmd(
        "py", Path("s"), "T36SXA", 2023, Path("i"), Path("lab"), Path("sp.json"), Path("scr")
    )
    assert "--weights-out" in lab
    assert "scr" in lab[lab.index("--weights-out") + 1]
    met = m.meteo_cmd("py", Path("s"), "T36SXA", 2023, Path("i"), Path("e"), Path("c"), Path("scr"))
    assert "--norm-out" in met
    assert "_labeled.parquet" in met[met.index("--labeled-parquet") + 1]
