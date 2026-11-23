"""Catalog-open hardening: bounded timeouts + retry (PRD risk I3).

Covers the failure modes a 2018 CMR campaign hit that the existing search-level
retries did not: a connect that hangs forever because requests applies no default
timeout, and a connect timeout raised from ``open_catalog`` - outside the guarded
block - which aborted the tile with exit 1 instead of skipping it.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from nilevit import hls_stac


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse backoff so retry tests stay fast."""
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)


class _FakeClient:
    """Stand-in for pystac_client.Client, recording how it was opened."""

    def __init__(self, url: str, **kwargs: Any) -> None:
        self.url = url
        self.kwargs = kwargs


def _patch_client_open(monkeypatch: pytest.MonkeyPatch, side_effect: Any) -> list[dict[str, Any]]:
    """Replace ``Client.open`` with ``side_effect``; return the recorded calls.

    The replacement names ``timeout`` explicitly, because ``_open_timeout_kwargs``
    inspects the signature it is about to call - a ``**kwargs``-only stand-in would
    look like a pystac_client too old to accept one.
    """
    import pystac_client

    calls: list[dict[str, Any]] = []

    def _open(url: str, *, timeout: float | None = None, **kwargs: Any) -> Any:
        calls.append({"url": url, "timeout": timeout, **kwargs})
        return side_effect(url, timeout=timeout, **kwargs)

    monkeypatch.setattr(pystac_client.Client, "open", staticmethod(_open))
    return calls


# ---- timeouts ----------------------------------------------------------------


def test_open_catalog_passes_bounded_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A timeout must reach requests; its absence is what allowed a ~26h hang."""
    calls = _patch_client_open(monkeypatch, _FakeClient)

    hls_stac.open_catalog("cmr")

    assert len(calls) == 1
    assert calls[0]["timeout"] == hls_stac.STAC_TIMEOUT
    assert 0 < hls_stac.STAC_TIMEOUT <= 120


def test_open_catalog_honours_explicit_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_client_open(monkeypatch, _FakeClient)

    hls_stac.open_catalog("cmr", timeout=5.0)

    assert calls[0]["timeout"] == 5.0


def test_open_timeout_kwargs_omits_unsupported_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a pystac_client without ``timeout``, pass nothing rather than TypeError."""
    import pystac_client

    monkeypatch.setattr(pystac_client.Client, "open", staticmethod(lambda url: _FakeClient(url)))

    assert hls_stac._open_timeout_kwargs(30.0) == {}


def test_open_catalog_uses_configured_url(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_client_open(monkeypatch, _FakeClient)

    hls_stac.open_catalog("cmr")

    assert calls[0]["url"] == hls_stac.CMR_STAC_URL


# ---- retry -------------------------------------------------------------------


def test_open_catalog_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient connect failure must not cost the tile."""
    attempts = {"n": 0}

    def _flaky(url: str, **kwargs: Any) -> Any:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("connect timed out")
        return _FakeClient(url, **kwargs)

    _patch_client_open(monkeypatch, _flaky)

    catalog = hls_stac.open_catalog("cmr", attempts=4, retry_delay=0.0)

    assert attempts["n"] == 3
    assert isinstance(catalog, _FakeClient)


def test_open_catalog_raises_runtimeerror_when_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exhaustion must surface as RuntimeError: that is what 05b catches to skip."""

    def _always_fail(url: str, **kwargs: Any) -> Any:
        raise ConnectionError("connect timed out")

    calls = _patch_client_open(monkeypatch, _always_fail)

    with pytest.raises(RuntimeError) as excinfo:
        hls_stac.open_catalog("cmr", attempts=3, retry_delay=0.0)

    assert len(calls) == 3
    assert "cmr" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ConnectionError)


def test_open_catalog_backs_off_between_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backoff must be exponential, so a throttling endpoint gets room to recover."""
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)

    def _always_fail(url: str, **kwargs: Any) -> Any:
        raise ConnectionError("connect timed out")

    _patch_client_open(monkeypatch, _always_fail)

    with pytest.raises(RuntimeError):
        hls_stac.open_catalog("cmr", attempts=4, retry_delay=2.0)

    assert slept == [2.0, 4.0, 8.0]


# ---- shared retry helper -----------------------------------------------------


def test_retry_returns_falsy_result_without_retrying() -> None:
    """An empty search result is a valid answer, not a failure to retry."""
    calls = {"n": 0}

    def _empty() -> list[Any]:
        calls["n"] += 1
        return []

    assert hls_stac._retry("empty", _empty, attempts=3, delay=0.0) == []
    assert calls["n"] == 1


def test_search_with_retry_still_retries_then_raises() -> None:
    """The pre-existing search retry must survive the refactor onto _retry."""

    class _FailingCatalog:
        def __init__(self) -> None:
            self.n = 0

        def search(self, **_kwargs: Any) -> Any:
            self.n += 1
            raise ValueError("APIError: <html>")

    catalog = _FailingCatalog()

    with pytest.raises(RuntimeError, match="STAC search"):
        hls_stac._search_with_retry(catalog, {}, attempts=3, delay=0.0)

    assert catalog.n == 3


def test_search_with_retry_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FlakyCatalog:
        def __init__(self) -> None:
            self.n = 0

        def search(self, **_kwargs: Any) -> Any:
            self.n += 1
            if self.n < 2:
                raise ValueError("APIError: <html>")
            return _Result()

    class _Result:
        def items(self) -> list[str]:
            return ["item"]

    catalog = _FlakyCatalog()

    assert hls_stac._search_with_retry(catalog, {}, attempts=3, delay=0.0) == ["item"]


# ---- GDAL environment --------------------------------------------------------


def test_configure_earthdata_gdal_sets_read_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Band reads need bounds too: a stalled socket raises nothing on its own."""
    for key in (
        "GDAL_HTTP_CONNECTTIMEOUT",
        "GDAL_HTTP_TIMEOUT",
        "GDAL_HTTP_LOW_SPEED_LIMIT",
        "GDAL_HTTP_LOW_SPEED_TIME",
    ):
        monkeypatch.delenv(key, raising=False)

    hls_stac.configure_earthdata_gdal()

    import os

    assert int(os.environ["GDAL_HTTP_CONNECTTIMEOUT"]) > 0
    assert int(os.environ["GDAL_HTTP_TIMEOUT"]) > 0
    assert int(os.environ["GDAL_HTTP_LOW_SPEED_LIMIT"]) > 0
    assert int(os.environ["GDAL_HTTP_LOW_SPEED_TIME"]) > 0


def test_configure_earthdata_gdal_does_not_override_operator_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An env var set by hand for one run must win (setdefault, not set)."""
    monkeypatch.setenv("GDAL_HTTP_TIMEOUT", "600")

    hls_stac.configure_earthdata_gdal()

    import os

    assert os.environ["GDAL_HTTP_TIMEOUT"] == "600"
