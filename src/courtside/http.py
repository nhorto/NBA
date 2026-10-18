"""Single HTTP client: curl_cffi chrome impersonation, rate limiting, retries."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlparse

from curl_cffi import requests as cffi_requests

MIN_INTERVAL: dict[str, float] = {"stats.nba.com": 0.7}
DEFAULT_MIN_INTERVAL = 0.3

_sleep = time.sleep  # monkeypatch in tests
_monotonic = time.monotonic
_session: Any = None
_last_request: dict[str, float] = {}


class FetchError(Exception):
    def __init__(self, message: str, *, status: int | None = None, url: str | None = None):
        super().__init__(message)
        self.status = status
        self.url = url


class BlockedError(FetchError):
    """HTML / non-JSON / Akamai page or 403."""


def _get_session() -> Any:
    global _session
    if _session is None:
        _session = cffi_requests.Session(impersonate="chrome")
    return _session


def _build_headers(url: str, headers: dict[str, str] | None) -> dict[str, str]:
    host = urlparse(url).hostname or ""
    out: dict[str, str] = {}
    if host.endswith("nba.com"):
        out["Referer"] = "https://www.nba.com/"
        out["Origin"] = "https://www.nba.com"
    if headers:
        out.update(headers)
    return out


def _throttle(url: str) -> None:
    host = urlparse(url).hostname or ""
    interval = MIN_INTERVAL.get(host, DEFAULT_MIN_INTERVAL)
    last = _last_request.get(host)
    if last is not None:
        wait = interval - (_monotonic() - last)
        if wait > 0:
            _sleep(wait)
    _last_request[host] = _monotonic()


def _looks_like_html(resp: Any) -> bool:
    ctype = (resp.headers.get("content-type") or "").lower()
    if "html" in ctype:
        return True
    return resp.content[:64].lstrip().startswith(b"<")


def _request(url, params, headers, timeout, retries, backoff, *, parse_json: bool):
    hdrs = _build_headers(url, headers)
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            _sleep(backoff ** attempt)
        _throttle(url)
        try:
            resp = _get_session().get(url, params=params, headers=hdrs, timeout=timeout)
            status = resp.status_code
            if status == 404:
                raise FetchError(f"404 Not Found: {url}", status=404, url=url)
            if status == 403:
                raise BlockedError(f"403 Forbidden: {url}", status=403, url=url)
            if status == 429 or status >= 500:
                raise FetchError(f"HTTP {status}: {url}", status=status, url=url)
            if status >= 400:
                raise FetchError(f"HTTP {status}: {url}", status=status, url=url)
            if _looks_like_html(resp):
                raise BlockedError(f"HTML/blocked response: {url}", status=status, url=url)
            if not parse_json:
                return resp.content
            try:
                return json.loads(resp.content)
            except ValueError as e:
                raise BlockedError(f"Non-JSON response: {url}: {e}", status=status, url=url) from e
        except FetchError as e:
            if e.status == 404 or (e.status is not None and 400 <= e.status < 500
                                   and e.status not in (403, 429)):
                raise
            last_exc = e
        except Exception as e:  # timeouts / connection errors from curl_cffi
            last_exc = FetchError(f"{type(e).__name__}: {e}", url=url)
            last_exc.__cause__ = e
    assert last_exc is not None
    raise last_exc


def fetch_json(url: str, *, params: dict | None = None, headers: dict | None = None,
               timeout: float = 25.0, retries: int = 4, backoff: float = 1.5) -> dict | list:
    return _request(url, params, headers, timeout, retries, backoff, parse_json=True)


def fetch_bytes(url: str, *, params: dict | None = None, headers: dict | None = None,
                timeout: float = 25.0, retries: int = 4, backoff: float = 1.5) -> bytes:
    return _request(url, params, headers, timeout, retries, backoff, parse_json=False)
