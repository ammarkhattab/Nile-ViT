"""Offline tests for the TileSample schema and region mapping (PRD section 4.3)."""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from nilevit.schemas import (
    CLASS_NAMES,
    LABEL_NODATA,
    NUM_CLASSES,
    TileSample,
    region_for_point,
)


def _sample(**overrides) -> dict:
    base = {
        "sample_id": "T36RUU_2023-08-15_R000",
        "mgrs_tile": "36RUU",
        "center_lon": 31.0,
        "center_lat": 31.0,
        "date": dt.date(2023, 8, 15),
        "region": "delta",
        "image_path": "tiles/T36RUU/2023-08-15/image",
        "meteo_path": "tiles/T36RUU/2023-08-15/meteo",
        "label_path": "tiles/T36RUU/2023-08-15/label",
        "cloud_pct": 0.1,
        "valid_pct": 0.95,
        "split": "train",
    }
    base.update(overrides)
    return base


def test_label_constants() -> None:
    assert NUM_CLASSES == 4
    assert LABEL_NODATA == 255
    assert CLASS_NAMES == {0: "none", 1: "drought", 2: "heat", 3: "compound"}


def test_valid_sample_roundtrips() -> None:
    sample = TileSample(**_sample())
    assert sample.region == "delta"
    assert sample.date == dt.date(2023, 8, 15)


def test_cloud_and_valid_pct_are_bounded() -> None:
    with pytest.raises(ValidationError):
        TileSample(**_sample(cloud_pct=1.5))
    with pytest.raises(ValidationError):
        TileSample(**_sample(valid_pct=-0.1))


def test_split_and_region_are_constrained() -> None:
    with pytest.raises(ValidationError):
        TileSample(**_sample(split="buffer"))
    with pytest.raises(ValidationError):
        TileSample(**_sample(region="sahara"))


def test_region_for_point_delta_and_coast_precedence() -> None:
    # Delta core sits inside the coast bbox; delta must win.
    assert region_for_point(31.0, 31.2) == "delta"
    # A coast point outside the delta core.
    assert region_for_point(26.0, 31.5) == "n_coast"
    # ROI but neither sub-region -> em_shelf.
    assert region_for_point(24.0, 33.0) == "em_shelf"


def test_region_for_point_ood() -> None:
    assert region_for_point(-5.0, 33.0) == "ood_nw_africa"


def test_region_for_point_outside_raises() -> None:
    with pytest.raises(ValueError, match="outside"):
        region_for_point(60.0, 10.0)
