import pytest

from courtside import http


class Resp:
    def __init__(self, status=200, body=b'{"ok": 1}', ctype="application/json"):
        self.status_code = status
        self.content = body
        self.headers = {"content-type": ctype}


class FakeSession:
    def __init__(self, items):
        self.items = list(items)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers))
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def sleeps(monkeypatch):
    rec = []
    monkeypatch.setattr(http, "_sleep", rec.append)
    monkeypatch.setattr(http, "_last_request", {})
    return rec


def use(monkeypatch, *items):
    s = FakeSession(items)
    monkeypatch.setattr(http, "_get_session", lambda: s)
    return s


def test_success_and_nba_headers(monkeypatch, sleeps):
    s = use(monkeypatch, Resp())
    assert http.fetch_json("https://cdn.nba.com/x", params={"a": 1}) == {"ok": 1}
    _, params, h = s.calls[0]
    assert params == {"a": 1}
    assert h["Referer"] == "https://www.nba.com/" and h["Origin"] == "https://www.nba.com"


def test_non_nba_host_no_referer_and_override(monkeypatch, sleeps):
    s = use(monkeypatch, Resp(), Resp())
    http.fetch_json("https://site.api.espn.com/x")
    http.fetch_json("https://stats.nba.com/x", headers={"Referer": "https://example.com"})
    assert "Referer" not in s.calls[0][2]
    assert s.calls[1][2]["Referer"] == "https://example.com"
    assert s.calls[1][2]["Origin"] == "https://www.nba.com"


def test_retry_5xx_429_timeout_with_backoff(monkeypatch, sleeps):
    s = use(monkeypatch, Resp(500, b"x"), Resp(429, b"x"), TimeoutError("t"), Resp())
    assert http.fetch_json("https://a.com/x", backoff=2.0) == {"ok": 1}
    assert len(s.calls) == 4
    assert [x for x in sleeps if x >= 1.0] == [2.0, 4.0, 8.0]


def test_retries_exhausted_raises_last(monkeypatch, sleeps):
    s = use(monkeypatch, *[Resp(503, b"x")] * 3)
    with pytest.raises(http.FetchError) as ei:
        http.fetch_json("https://a.com/x", retries=2)
    assert ei.value.status == 503 and len(s.calls) == 3


def test_html_detected_by_content_type_and_body(monkeypatch, sleeps):
    use(monkeypatch, Resp(200, b"{}", "text/html"), Resp(200, b"  <html>", "application/json"),
        Resp(200, b"not json", "application/json"))
    with pytest.raises(http.BlockedError):
        http.fetch_json("https://a.com/x", retries=2)


def test_blocked_then_recovers(monkeypatch, sleeps):
    s = use(monkeypatch, Resp(403, b"denied"), Resp(200, b"<html>", "text/html"), Resp())
    assert http.fetch_json("https://a.com/x") == {"ok": 1}
    assert len(s.calls) == 3


def test_404_no_retry(monkeypatch, sleeps):
    s = use(monkeypatch, Resp(404, b"nope"))
    with pytest.raises(http.FetchError) as ei:
        http.fetch_json("https://a.com/x")
    assert ei.value.status == 404 and not isinstance(ei.value, http.BlockedError)
    assert len(s.calls) == 1


def test_per_host_rate_limit(monkeypatch, sleeps):
    t = [100.0]
    monkeypatch.setattr(http, "_monotonic", lambda: t[0])
    use(monkeypatch, Resp(), Resp(), Resp())
    http.fetch_json("https://stats.nba.com/a")
    http.fetch_json("https://stats.nba.com/b")
    assert sleeps == [pytest.approx(0.7)]
    sleeps.clear()
    http.fetch_json("https://other.com/c")  # different host, no wait
    assert sleeps == []


def test_fetch_bytes(monkeypatch, sleeps):
    use(monkeypatch, Resp(200, b"%PDF-1.4 data", "application/pdf"))
    assert http.fetch_bytes("https://ak-static.cms.nba.com/x.pdf") == b"%PDF-1.4 data"
