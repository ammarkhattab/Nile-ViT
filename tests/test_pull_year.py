"""Unit tests for `scripts/pull_year.py` (B1 driver).

Offline-only: month-range generation, source selection, and command
construction. The subprocess execution is not exercised in CI.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "pull_year.py"


@pytest.fixture(scope="module")
def mod():
    if not SCRIPT_PATH.exists():
        pytest.skip(f"script not found at {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("pull_year", SCRIPT_PATH)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["pull_year"] = m
    spec.loader.exec_module(m)
    return m


# ---- iter_months -------------------------------------------------------------
class TestIterMonths:
    def test_full_year(self, mod):
        months = mod.iter_months(2023)
        assert len(months) == 12
        assert months[0] == "2023-01"
        assert months[-1] == "2023-12"

    def test_sub_range(self, mod):
        assert mod.iter_months(2023, 7, 9) == ["2023-07", "2023-08", "2023-09"]

    def test_single_month(self, mod):
        assert mod.iter_months(2023, 8, 8) == ["2023-08"]

    def test_zero_padding(self, mod):
        assert mod.iter_months(2023, 1, 1) == ["2023-01"]

    @pytest.mark.parametrize("bad", [(0, 12), (1, 13), (9, 3)])
    def test_invalid_raises(self, mod, bad):
        with pytest.raises(ValueError):
            mod.iter_months(2023, *bad)


# ---- SOURCE_STEPS ------------------------------------------------------------
class TestSourceSteps:
    def test_four_steps(self, mod):
        assert len(mod.SOURCE_STEPS) == 4

    def test_keys(self, mod):
        keys = {s.key for s in mod.SOURCE_STEPS}
        assert keys == {"era5", "chirps", "modis_ndvi", "modis_lst"}

    def test_no_hls(self, mod):
        # HLS is intentionally excluded from B1.
        for s in mod.SOURCE_STEPS:
            assert "hls" not in s.script.lower()

    def test_modis_products(self, mod):
        by_key = {s.key: s for s in mod.SOURCE_STEPS}
        assert by_key["modis_ndvi"].extra == ("--product", "MOD13Q1")
        assert by_key["modis_lst"].extra == ("--product", "MOD11A2")

    def test_scripts_are_numbered_data_scripts(self, mod):
        for s in mod.SOURCE_STEPS:
            assert s.script.endswith(".py")
            assert s.script[0].isdigit()


# ---- select_steps ------------------------------------------------------------
class TestSelectSteps:
    def test_none_returns_all(self, mod):
        assert len(mod.select_steps(mod.SOURCE_STEPS, None)) == 4

    def test_empty_returns_all(self, mod):
        assert len(mod.select_steps(mod.SOURCE_STEPS, "")) == 4

    def test_subset(self, mod):
        sel = mod.select_steps(mod.SOURCE_STEPS, "era5,chirps")
        assert {s.key for s in sel} == {"era5", "chirps"}

    def test_whitespace_and_case(self, mod):
        sel = mod.select_steps(mod.SOURCE_STEPS, " ERA5 , Chirps ")
        assert {s.key for s in sel} == {"era5", "chirps"}

    def test_unknown_filtered_out(self, mod):
        sel = mod.select_steps(mod.SOURCE_STEPS, "era5,bogus")
        assert {s.key for s in sel} == {"era5"}


# ---- build_command -----------------------------------------------------------
class TestBuildCommand:
    def test_era5(self, mod):
        step = next(s for s in mod.SOURCE_STEPS if s.key == "era5")
        cmd = mod.build_command("python", Path("/repo/scripts/data"), step, "2023-08")
        assert cmd == [
            "python",
            str(Path("/repo/scripts/data") / "02_download_era5.py"),
            "--month",
            "2023-08",
        ]

    def test_modis_ndvi_has_product(self, mod):
        step = next(s for s in mod.SOURCE_STEPS if s.key == "modis_ndvi")
        cmd = mod.build_command("py", Path("/d"), step, "2023-08")
        assert "--product" in cmd
        assert "MOD13Q1" in cmd
        # Fixed args follow `--month YYYY-MM`.
        assert cmd[-4:] == ["--month", "2023-08", "--product", "MOD13Q1"]

    def test_month_present(self, mod):
        step = mod.SOURCE_STEPS[0]
        cmd = mod.build_command("py", Path("/d"), step, "2023-03")
        assert "--month" in cmd
        assert "2023-03" in cmd
