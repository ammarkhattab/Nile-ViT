"""Offline tests for Script 06b (CHIRPS monthly precip climatology).

No network: exercises the URL/source registry, the surrogate sanitizer, and the
ROI + year-range subsetting on a small synthetic monthly dataset.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "06b_climatology_precip.py"


def load_module():
    spec = importlib.util.spec_from_file_location("climatology_precip", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


M = load_module()


# ---- source registry / URL resolution ----


def test_sources_registry_has_v2_consolidated():
    assert "v2-consolidated" in M.SOURCES
    url = M.SOURCES["v2-consolidated"]
    assert url.startswith("https://data.chc.ucsb.edu/")
    assert url.endswith("chirps-v2.0.monthly.nc")


def test_resolve_url_named_source():
    assert M.resolve_url("v2-consolidated", None) == M.SOURCES["v2-consolidated"]


def test_resolve_url_override_wins():
    custom = "https://example.org/chirps-v3.0.monthly.nc"
    assert M.resolve_url("v2-consolidated", custom) == custom


def test_resolve_url_unknown_source_raises():
    with pytest.raises(ValueError, match="unknown source"):
        M.resolve_url("nope", None)


def test_roi_bbox_constant():
    assert M.ROI_BBOX == (22.0, 30.0, 36.0, 37.0)


# ---- attribute sanitizer ----


def test_clean_value_scrubs_lone_surrogate():
    bad = "ok\udc80end"  # lone surrogate that breaks utf-8 encoding
    cleaned = M._clean_value(bad)
    cleaned.encode("utf-8")  # must not raise


def test_clean_value_passthrough_for_numbers():
    assert M._clean_value(3.14) == 3.14
    assert M._clean_value(7) == 7


def test_sanitize_attrs_in_place():
    xr = pytest.importorskip("xarray")
    import numpy as np

    ds = xr.Dataset(
        {"precip": ("x", np.arange(3.0))},
        coords={"x": [0, 1, 2]},
        attrs={"title": "bad\udc80title"},
    )
    ds["precip"].attrs["units"] = "mm\udcffmonth"
    M._sanitize_attrs(ds)
    ds.attrs["title"].encode("utf-8")
    ds["precip"].attrs["units"].encode("utf-8")


# ---- subsetting on synthetic monthly data ----


def _synthetic_monthly(lat_descending: bool):
    xr = pytest.importorskip("xarray")
    import numpy as np
    import pandas as pd

    lats = np.arange(25.0, 40.0 + 0.01, 0.5)  # covers ROI S=30..N=37
    if lat_descending:
        lats = lats[::-1]
    lons = np.arange(18.0, 40.0 + 0.01, 0.5)  # covers ROI W=22..E=36
    time = pd.date_range("1990-01-01", "2021-12-01", freq="MS")
    data = np.random.default_rng(0).random((time.size, lats.size, lons.size))
    return xr.Dataset(
        {"precip": (("time", "latitude", "longitude"), data.astype("float32"))},
        coords={"time": time, "latitude": lats, "longitude": lons},
    )


@pytest.mark.parametrize("lat_descending", [False, True])
def test_subset_and_slice_extent_and_years(tmp_path, lat_descending):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")

    src = tmp_path / "global_monthly.nc"
    _synthetic_monthly(lat_descending).to_netcdf(src, engine="h5netcdf")

    dst = tmp_path / "precip_monthly_1991_2020.nc"
    sub = M.subset_and_slice(src, dst, M.ROI_BBOX, 1991, 2020)

    # 30 baseline years x 12 months
    assert sub.sizes["time"] == 30 * 12
    # ROI latitude bounds 30..37, longitude 22..36
    lat = sub["latitude"].values
    lon = sub["longitude"].values
    assert lat.min() >= 30.0 - 1e-6 and lat.max() <= 37.0 + 1e-6
    assert lon.min() >= 22.0 - 1e-6 and lon.max() <= 36.0 + 1e-6
    # file written and re-openable
    assert dst.exists()
    with xr.open_dataset(dst) as reopened:
        assert "precip" in reopened.data_vars


def test_subset_excludes_out_of_range_years(tmp_path):
    pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")

    src = tmp_path / "global_monthly.nc"
    _synthetic_monthly(False).to_netcdf(src, engine="h5netcdf")

    dst = tmp_path / "precip_monthly_2000_2005.nc"
    sub = M.subset_and_slice(src, dst, M.ROI_BBOX, 2000, 2005)
    assert sub.sizes["time"] == 6 * 12
    years = sub["time"].dt.year.values
    assert years.min() == 2000 and years.max() == 2005


# ---- misc helpers ----


def test_default_data_root_is_path():
    assert isinstance(M.default_data_root(), Path)


def test_is_colab_false_here():
    assert M.is_colab() is False
