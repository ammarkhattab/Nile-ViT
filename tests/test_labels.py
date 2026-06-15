"""Tests for nilevit.labels - the PRD 4.4 label logic.

Includes the PRD 4.4 acceptance test: the label function on a hand-built
synthetic set of pixels produces all four classes, with the documented
precedence (compound > heat > drought > none).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "nilevit" / "labels.py"


@pytest.fixture(scope="module")
def lab():
    spec = importlib.util.spec_from_file_location("nilevit_labels", MODULE_PATH)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["nilevit_labels"] = m
    spec.loader.exec_module(m)
    return m


# ---- indicator formulas ----
class TestIndicators:
    def test_vci_endpoints(self, lab):
        # NDVI at min -> 0, at max -> 100, midpoint -> 50.
        assert float(lab.vci(0.2, 0.2, 0.8)) == pytest.approx(0.0)
        assert float(lab.vci(0.8, 0.2, 0.8)) == pytest.approx(100.0)
        assert float(lab.vci(0.5, 0.2, 0.8)) == pytest.approx(50.0)

    def test_tci_is_inverted(self, lab):
        # Hottest LST -> TCI 0; coldest -> 100.
        assert float(lab.tci(320.0, 280.0, 320.0)) == pytest.approx(0.0)
        assert float(lab.tci(280.0, 280.0, 320.0)) == pytest.approx(100.0)

    def test_vhi_average(self, lab):
        assert float(lab.vhi(30.0, 50.0)) == pytest.approx(40.0)

    def test_flat_range_is_nan(self, lab):
        # min == max -> undefined index -> NaN, not inf.
        assert np.isnan(float(lab.vci(0.5, 0.5, 0.5)))

    def test_zscore_and_zero_sigma(self, lab):
        assert float(lab.zscore(10.0, 4.0, 3.0)) == pytest.approx(2.0)
        assert np.isnan(float(lab.zscore(10.0, 4.0, 0.0)))


# ---- PRD 4.4 acceptance ----
class TestClassifyAcceptance:
    def test_all_four_classes_present(self, lab):
        # one pixel engineered for each class
        spei = np.array([0.0, -1.5, 0.0, -1.5])
        tmaxz = np.array([0.0, 0.0, 2.5, 2.5])
        vhi = np.array([50.0, 50.0, 50.0, 20.0])
        out = lab.classify(spei, tmaxz, vhi)
        assert out.tolist() == [lab.NONE, lab.DROUGHT, lab.HEAT, lab.COMPOUND]
        assert set(np.unique(out)) == {0, 1, 2, 3}
        assert out.dtype == np.uint8

    def test_heat_outranks_drought_only(self, lab):
        # drought AND heat but no veg stress -> heat-only (2), not compound/drought
        out = lab.classify(np.array([-2.0]), np.array([3.0]), np.array([80.0]))
        assert int(out[0]) == lab.HEAT

    def test_compound_needs_all_three(self, lab):
        # all three satisfied -> compound
        out = lab.classify(np.array([-1.2]), np.array([2.1]), np.array([10.0]))
        assert int(out[0]) == lab.COMPOUND

    def test_threshold_boundaries_are_strict(self, lab):
        # exactly at thresholds should NOT fire (strict < / >)
        out = lab.classify(
            np.array([lab.SPEI3_THRESHOLD]),
            np.array([lab.TMAX_Z_THRESHOLD]),
            np.array([lab.VHI_THRESHOLD]),
        )
        assert int(out[0]) == lab.NONE

    def test_nan_indicator_degrades_safely(self, lab):
        # NaN SPEI but strong heat -> heat-only (cannot be compound w/o drought)
        out = lab.classify(np.array([np.nan]), np.array([3.0]), np.array([10.0]))
        assert int(out[0]) == lab.HEAT
        # all-NaN -> none
        out2 = lab.classify(np.array([np.nan]), np.array([np.nan]), np.array([np.nan]))
        assert int(out2[0]) == lab.NONE


# ---- end-to-end raster convenience ----
def test_label_from_rasters_compound(lab):
    # NDVI low (stress), LST hot (stress) -> low VHI; hot Tmax; dry WB -> compound
    out = lab.label_from_rasters(
        ndvi=0.25,
        ndvi_min=0.2,
        ndvi_max=0.9,
        lst=318.0,
        lst_min=280.0,
        lst_max=320.0,
        tmax=44.0,
        tmax_mu=38.0,
        tmax_sigma=2.0,
        wb3=-200.0,
        wb_mu=-40.0,
        wb_sigma=30.0,
    )
    assert int(out) == lab.COMPOUND


def test_label_from_rasters_none_for_healthy(lab):
    out = lab.label_from_rasters(
        ndvi=0.8,
        ndvi_min=0.2,
        ndvi_max=0.9,
        lst=295.0,
        lst_min=280.0,
        lst_max=320.0,
        tmax=37.0,
        tmax_mu=38.0,
        tmax_sigma=2.0,
        wb3=-30.0,
        wb_mu=-40.0,
        wb_sigma=30.0,
    )
    assert int(out) == lab.NONE
