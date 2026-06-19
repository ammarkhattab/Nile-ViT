"""Offline tests for scripts/labels/build_labels.py pure helpers.

Covers the temporal cadence, SPEI-window math, Hargreaves PET (must match 06c),
and the geographic regrid path (rename->lat/lon->bilinear). The sinusoidal-MODIS
reproject and full I/O orchestration are validated on the real run.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "labels" / "build_labels.py"


@pytest.fixture(scope="module")
def bl():
    spec = importlib.util.spec_from_file_location("build_labels", MODULE_PATH)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["build_labels"] = m
    spec.loader.exec_module(m)
    return m


class TestCadence:
    def test_23_dates_every_16_days(self, bl):
        d = bl.modis_16day_dates(2023)
        assert len(d) == 23
        assert d[0] == date(2023, 1, 1)
        assert d[1] == date(2023, 1, 17)
        assert d[-1] == date(2023, 12, 19)  # DOY 353
        gaps = {(b - a).days for a, b in itertools.pairwise(d)}
        assert gaps == {16}

    def test_doy(self, bl):
        assert bl.doy_of(date(2023, 1, 1)) == 1
        assert bl.doy_of(date(2023, 8, 13)) == 225
        assert bl.doy_of(date(2023, 12, 31)) == 365


class TestSpeiWindow:
    def test_mid_year(self, bl):
        assert bl.wb3_window_months(date(2023, 8, 15)) == [
            (2023, 6),
            (2023, 7),
            (2023, 8),
        ]

    def test_year_boundary(self, bl):
        assert bl.wb3_window_months(date(2023, 1, 10)) == [
            (2022, 11),
            (2022, 12),
            (2023, 1),
        ]
        assert bl.wb3_window_months(date(2023, 2, 1)) == [
            (2022, 12),
            (2023, 1),
            (2023, 2),
        ]


class TestHargreaves:
    def test_ra_summer_peak(self, bl):
        # Northern-hemisphere mid-lat Ra peaks near summer solstice (~41 MJ/m2/day).
        ra_summer = float(bl.hargreaves_ra(30.0, 172))
        ra_winter = float(bl.hargreaves_ra(30.0, 355))
        assert 38.0 < ra_summer < 44.0
        assert ra_winter < ra_summer

    def test_pet_zero_when_no_range(self, bl):
        ra = bl.hargreaves_ra(30.0, 172)
        # tmax == tmin -> temp range 0 -> PET 0 (sqrt(0))
        assert float(bl.hargreaves_pet(30.0, 30.0, ra)) == pytest.approx(0.0)

    def test_pet_positive_and_scales(self, bl):
        ra = bl.hargreaves_ra(30.0, 172)
        hot = float(bl.hargreaves_pet(40.0, 25.0, ra))
        mild = float(bl.hargreaves_pet(25.0, 20.0, ra))
        assert hot > mild > 0.0


class TestRegrid:
    def test_to_latlon_renames_and_sorts(self, bl):
        import xarray as xr

        da = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            coords={"latitude": [30.0, 20.0], "longitude": [10.0, 0.0]},
            dims=["latitude", "longitude"],
        )
        out = bl._to_latlon(da)
        assert out.dims == ("lat", "lon")
        assert list(out["lat"].values) == [20.0, 30.0]  # ascending
        assert list(out["lon"].values) == [0.0, 10.0]

    def test_interp_to_bilinear_midpoint(self, bl):
        import xarray as xr

        # 0..1 ramp in lon; midpoint should interpolate to 0.5
        da = xr.DataArray(
            np.array([[0.0, 1.0], [0.0, 1.0]]),
            coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
            dims=["lat", "lon"],
        )
        out = bl.interp_to(da, np.array([0.5]), np.array([0.5]))
        assert out.shape == (1, 1)
        assert float(out.values[0, 0]) == pytest.approx(0.5)


def test_default_data_root_is_repo_data(bl):
    # parents[2] of scripts/labels/build_labels.py is the repo root
    root = bl.default_data_root()
    assert root.name == "data"
