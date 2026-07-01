"""Offline tests for nilevit/hls_stac.py pure logic (no network).

Fake items mirror the real PC structure confirmed by the probe:
  id  = 'HLS.S30.T36RUU.2023243T082611.v2.0'
  assets keyed by raw band codes (B02.. / B8A / Fmask), each with an .href.
"""

from __future__ import annotations

from datetime import date, datetime

from nilevit.hls_stac import (
    date_from_item,
    filter_items_for_tile,
    hrefs_by_date,
    item_band_hrefs,
    manifest_rows_for_items,
    sensor_from_id,
    tile_from_id,
)

# The 05b band map, inlined so the test doesn't depend on the script import path.
S30_MAP = {
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "nir_narrow": "B8A",
    "swir1": "B11",
    "swir2": "B12",
}
L30_MAP = {
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "nir_narrow": "B05",
    "swir1": "B06",
    "swir2": "B07",
}
BAND_MAP = {"S30": S30_MAP, "L30": L30_MAP}


class _Asset:
    def __init__(self, href: str) -> None:
        self.href = href


class _Item:
    def __init__(self, item_id: str, dt: datetime, cloud: float, codes: list[str]):
        self.id = item_id
        self.datetime = dt
        self.properties = {"datetime": dt.isoformat(), "eo:cloud_cover": cloud}
        self.assets = {c: _Asset(f"https://blob/{item_id}/{c}.tif") for c in codes}


S30_CODES = ["B02", "B03", "B04", "B8A", "B11", "B12", "Fmask", "SAA", "thumbnail"]
L30_CODES = ["B02", "B03", "B04", "B05", "B06", "B07", "Fmask", "SZA"]


def _s30(tile="T36RUU", day=1):
    return _Item(
        f"HLS.S30.{tile}.2023{200 + day:03d}T082611.v2.0",
        datetime(2023, 8, day, 8, 42, 15),
        0.0,
        S30_CODES,
    )


def test_sensor_and_tile_parsing() -> None:
    assert sensor_from_id("HLS.S30.T36RUU.2023243T082611.v2.0") == "S30"
    assert sensor_from_id("HLS.L30.T36RVU.2023243T082351.v2.0") == "L30"
    assert tile_from_id("HLS.S30.T36RUU.2023243T082611.v2.0") == "T36RUU"
    assert sensor_from_id("garbage") is None
    assert tile_from_id("garbage") is None


def test_date_from_item() -> None:
    assert date_from_item(_s30(day=14)) == date(2023, 8, 14)


def test_item_band_hrefs_s30_and_l30() -> None:
    s = item_band_hrefs(_s30(), BAND_MAP)
    assert s is not None
    assert s["nir_narrow"].endswith("/B8A.tif")  # S30 nir = B8A
    assert s["swir2"].endswith("/B12.tif")
    assert "Fmask" in s and len(s) == 7

    litem = _Item(
        "HLS.L30.T36RUU.2023243T082351.v2.0",
        datetime(2023, 8, 31, 8, 23, 51),
        0.0,
        L30_CODES,
    )
    lmap = item_band_hrefs(litem, BAND_MAP)
    assert lmap is not None
    assert lmap["nir_narrow"].endswith("/B05.tif")  # L30 nir = B05
    assert lmap["swir1"].endswith("/B06.tif")


def test_item_band_hrefs_missing_band_returns_none() -> None:
    incomplete = _Item(
        "HLS.S30.T36RUU.2023243T082611.v2.0",
        datetime(2023, 8, 1),
        0.0,
        ["B02", "B03", "Fmask"],  # missing red/nir/swir
    )
    assert item_band_hrefs(incomplete, BAND_MAP) is None


def test_filter_items_for_tile() -> None:
    items = [_s30("T36RUU", 1), _s30("T36RTT", 2), _s30("T36RUU", 3)]
    kept = filter_items_for_tile(items, "T36RUU")
    assert len(kept) == 2
    assert all(tile_from_id(i.id) == "T36RUU" for i in kept)


def test_canonical_band_map_matches_expected() -> None:
    # Guards nilevit/hls_bands.py against silent drift from the confirmed spec.
    from nilevit.hls_bands import HLS_BAND_MAP

    assert HLS_BAND_MAP["S30"] == S30_MAP
    assert HLS_BAND_MAP["L30"] == L30_MAP


def test_hrefs_by_date_one_scene_per_date() -> None:
    # Two S30 scenes on distinct dates + a duplicate date -> first wins.
    dup = _s30(day=1)
    dup.id = "HLS.L30.T36RUU.2023213T090000.v2.0"  # same date, different sensor/id
    dup.assets = {c: _Asset(f"https://blob/dup/{c}.tif") for c in L30_CODES}
    grouped = hrefs_by_date([_s30(day=1), dup, _s30(day=6)])
    assert set(grouped) == {"2023-08-01", "2023-08-06"}
    assert grouped["2023-08-01"]["sensor"] == "S30"  # first item kept
    assert set(grouped["2023-08-01"]["hrefs"]) == {
        "blue",
        "green",
        "red",
        "nir_narrow",
        "swir1",
        "swir2",
        "Fmask",
    }


def test_manifest_rows() -> None:
    rows = list(manifest_rows_for_items([_s30(day=1), _s30(day=6)]))
    assert len(rows) == 2
    r = rows[0]
    assert r["tile"] == "T36RUU"
    assert r["sensor"] == "S30"
    assert r["date"] == "2023-08-01"
    assert set(r["hrefs"]) == {
        "blue",
        "green",
        "red",
        "nir_narrow",
        "swir1",
        "swir2",
        "Fmask",
    }
