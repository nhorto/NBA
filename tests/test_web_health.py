"""Model Health page: /health, /api/health and the critical-alert banner.

The MLOps tables are filled with real-shaped rows (as `courtside mlops monitor` writes them) in a
temporary database; nothing touches the network or the repo's var/ directory.
"""

from __future__ import annotations

import json
import re
import shutil
import warnings
from datetime import date, timedelta
from pathlib import Path

import pytest

warnings.filterwarnings("ignore", message="Using `httpx`")
from fastapi.testclient import TestClient  # noqa: E402

from courtside import db  # noqa: E402
from courtside.web import health  # noqa: E402
from courtside.web.app import create_app  # noqa: E402

SAMPLE = Path(__file__).parent / "fixtures" / "brief" / "sample_brief.json"
DAY = "2026-02-20"
START = date(2026, 1, 1)
N_DAYS = 60


def no_none(html: str) -> None:
    text = re.sub(r"<script.*?</script>", "", html, flags=re.S)
    assert not re.search(r"\bNone\b", text), re.search(r".{40}\bNone\b.{40}", text).group(0)
    assert not re.search(r"\bnan\b", text, re.I), re.search(r".{40}\bnan\b.{40}", text, re.I).group(0)
    assert "undefined" not in text


def _bins(shift: float = 0.0) -> list[dict]:
    out = []
    for i in range(10):
        p = 0.05 + 0.1 * i
        out.append({"lo": round(i / 10, 2), "hi": round((i + 1) / 10, 2), "n": 0 if i == 0 else 20 + i,
                    "pred": None if i == 0 else round(p, 4),
                    "actual": None if i == 0 else round(min(1, max(0, p - shift * (p - 0.5))), 4)})
    return out


def _snap(as_of: str, model: str, window: str, n: int, ll: float, *, mkt: bool = True) -> tuple:
    d = {"calibration": _bins(0.2 if model == "gbm_opinion" else 0.0), "ll_se": 0.02,
         "spiegelhalter_z": 2.8 if model == "gbm_opinion" else 0.4,
         "home_bias_se": 0.03, "margin_bias_se": 0.9, "n_margin": n,
         "total_bias_se": 1.1, "n_total": n, "first_day": "2026-01-01", "last_day": as_of,
         "cal_intercept": -0.02, "cal_slope": 0.8 if model == "gbm_opinion" else 0.97,
         "cal_slope_se": 0.09}
    diff = None if model == "market" or not mkt else ll - 0.60
    return (as_of, model, "v1" if model != "market" else None, window, n, ll, ll * 0.36, 0.64,
            0.03, 0.58, 0.54, 3.1, 11.0, -2.2, 14.0, n if mkt else 0, 0.6, 0.215, diff,
            0.004 if diff is not None else None, json.dumps(d))


def fill(conn, *, critical: bool = False, warn: bool = True) -> None:
    rows = []
    for i in range(N_DAYS):
        as_of = (START + timedelta(days=i)).isoformat()
        n = 10 + i * 2
        for model, ll in (("gbm_market", 0.605), ("gbm_opinion", 0.62), ("market", 0.60)):
            for window in ("7d", "30d", "last100", "last300", "season"):
                rows.append(_snap(as_of, model, window, min(n, 100) if window == "last100" else n,
                                  ll + 0.01 * (i % 5) / 5))
    conn.executemany("INSERT INTO metric_snapshots VALUES (" + ",".join("?" * 21) + ")", rows)
    last = (START + timedelta(days=N_DAYS - 1)).isoformat()
    conn.executemany("INSERT INTO drift_snapshots VALUES (?,?,?,?,?)", [
        (last, "pred_total", 0.43, "2025-26 before window", 183),
        (last, "elo_diff", 0.08, "2025-26 before window", 183),
        (last, "rest_diff", 0.55, "2025-26 before window", 183)])
    conn.executemany("INSERT INTO data_checks VALUES (?,?,?,?,?,?)", [
        ("2026-03-02T10:00:00Z", "postgame.finals_archived", "fail", 0.0, 1.0, "9 of 9 games missing"),
        ("2026-03-01T10:00:00Z", "postgame.finals_archived", "ok", 1.0, 1.0, "all archived"),
        ("2026-03-02T10:00:00Z", "pregame.odds_coverage", "ok", 1.0, 0.9, "odds for 9 of 9"),
        ("2026-03-02T10:00:00Z", "stale.pregame", "ok", 3.5, 16.0, "ran 3.5 h ago"),
        ("2026-03-02T10:00:00Z", "postgame.facts_validation", "skip", None, None, "nothing to validate")])
    conn.execute("INSERT INTO model_versions (version_id, model_name, spec_hash, code_sha, status, "
                 "created_at, promoted_at) VALUES ('gbm_market@v1-abc12345','gbm_market','h',"
                 "'2aa9a898aa0e-dirty-0531ff20','champion','2026-01-01T00:00:00Z','2026-01-02T00:00:00Z')")
    conn.executemany("INSERT INTO registry_events (ts, version_id, action, reason) VALUES (?,?,?,?)", [
        ("2026-01-01T00:00:00Z", "gbm_market@v1-abc12345", "register", "bootstrap"),
        ("2026-01-02T00:00:00Z", "gbm_market@v1-abc12345", "promote", "initial champion"),
        ("2026-01-09T00:00:00Z", "gbm_market@v2-zzz", "reject", "failed the offline gate")])
    alerts = []
    if warn:
        alerts.append(("drift:pred_total", "warn", "drift", "Live input pred_total has shifted: PSI 0.43"))
    if critical:
        alerts.append(("calib:gbm_opinion", "critical", "calibration",
                       "gbm_opinion is badly overconfident: slope 0.6"))
    for key, sev, kind, msg in alerts:
        conn.execute("INSERT INTO alerts (key, created_at, updated_at, severity, kind, message) "
                     "VALUES (?,?,?,?,?,?)", (key, "2026-03-01T10:00:00Z", "2026-03-02T10:00:00Z",
                                              sev, kind, msg))
    conn.execute("INSERT INTO alerts (key, created_at, updated_at, severity, kind, message, status,"
                 " resolved_at) VALUES ('bias:old','2026-02-01T00:00:00Z','2026-02-05T00:00:00Z',"
                 "'warn','bias','Over-predicting totals','resolved','2026-02-05T00:00:00Z')")
    conn.commit()


def make_client(data_dir: Path, *, populate: bool = True, **kw) -> TestClient:
    (data_dir / "briefs").mkdir(parents=True, exist_ok=True)
    shutil.copy(SAMPLE, data_dir / "briefs" / f"{DAY}.json")
    conn = db.connect()
    if populate:
        fill(conn, **kw)
    conn.close()
    return TestClient(create_app(data_dir))


# ------------------------------------------------------------------ rendering

def test_health_renders_with_real_shaped_data(data_dir):
    c = make_client(data_dir)
    r = c.get("/health")
    assert r.status_code == 200
    html = r.text
    no_none(html)
    for s in ("Scoreboard", "Performance over time", "Calibration", "Over- and under-prediction",
              "Feature drift", "Alerts", "Data checks", "Model history"):
        assert s in html
    assert "As of" in html and "v1-abc12345" in html and "2aa9a89" in html
    assert 'class="chart hm-chart' in html                    # time-series SVG
    assert 'class="chart calib-chart' in html                 # reliability diagram
    assert "worse than the market" in html or "better than the market" in html \
        or "Level with the market" in html
    assert "Overconfident" in html or "overconfident" in html  # z=2.8 for gbm_opinion
    assert "Over-predicts" in html or "on target" in html
    assert "0.25" in html and "0.5" in html                   # drift thresholds
    assert "Recently resolved" in html and "Check history" in html
    assert "Rejected" in html and "failed the offline gate" in html


def test_health_state_is_warn_with_only_warnings(data_dir):
    c = make_client(data_dir)
    body = c.get("/api/health").json()
    assert body["status"] == "warn"


@pytest.mark.parametrize("populate", [True, False])
def test_health_renders_with_empty_tables(data_dir, populate):
    c = make_client(data_dir, populate=populate)
    r = c.get("/health")
    assert r.status_code == 200
    no_none(r.text)
    if not populate:
        assert "Monitoring hasn" in r.text
        assert "No scored games yet" in r.text
        assert "No open alerts" in r.text
        assert "No data checks recorded yet" in r.text
        assert "No model changes recorded" in r.text
        assert c.get("/api/health").json()["status"] == "unknown"


def test_health_renders_without_a_database(data_dir):
    (data_dir / "briefs").mkdir(parents=True)
    shutil.copy(SAMPLE, data_dir / "briefs" / f"{DAY}.json")
    c = TestClient(create_app(data_dir))
    r = c.get("/health")
    assert r.status_code == 200
    no_none(r.text)
    assert c.get("/").status_code == 200


def test_early_season_few_games_has_no_broken_charts(data_dir):
    conn = db.connect()
    rows = [_snap("2026-01-05", m, w, 4, 0.7) for m in ("gbm_market", "market")
            for w in ("7d", "30d", "last100", "last300", "season")]
    conn.executemany("INSERT INTO metric_snapshots VALUES (" + ",".join("?" * 21) + ")", rows)
    conn.commit()
    conn.close()
    (data_dir / "briefs").mkdir(parents=True, exist_ok=True)
    c = TestClient(create_app(data_dir))
    html = c.get("/health").text
    no_none(html)
    assert 'class="chart hm-chart' not in html
    assert "Charts start after about" in html
    assert "Too early to compare" in html


def test_missing_optional_details_do_not_break(data_dir):
    c = make_client(data_dir)
    conn = db.connect()
    conn.execute("UPDATE metric_snapshots SET details_json=NULL, accuracy=NULL, brier=NULL")
    conn.commit()
    conn.close()
    r = c.get("/health")
    assert r.status_code == 200
    no_none(r.text)


# ------------------------------------------------------------------ banner

def test_banner_only_for_critical_alerts(data_dir):
    c = make_client(data_dir, warn=True, critical=False)
    assert "hm-banner" not in c.get("/").text
    assert "hm-banner" not in c.get(f"/brief/{DAY}").text


def test_banner_shows_for_critical_alert_and_links_to_health(data_dir):
    c = make_client(data_dir, warn=True, critical=True)
    for url in ("/", f"/brief/{DAY}"):
        html = c.get(url).text
        assert 'class="hm-banner"' in html and 'href="/health"' in html
        assert "1 critical alert." in html
        no_none(html)
    assert c.get("/api/health").json()["status"] == "critical"
    assert "needs attention" in c.get("/health").text.lower()


def test_resolved_critical_alert_does_not_trigger_banner(data_dir):
    c = make_client(data_dir, warn=False, critical=True)
    conn = db.connect()
    conn.execute("UPDATE alerts SET status='resolved', resolved_at='2026-03-03T00:00:00Z'")
    conn.commit()
    conn.close()
    assert "hm-banner" not in c.get("/").text


def test_site_links_to_health(data_dir):
    c = make_client(data_dir, populate=False)
    assert 'href="/health"' in c.get("/").text
    assert 'href="/health"' in c.get("/archive").text


# ------------------------------------------------------------------ API

def test_api_health_shape(data_dir):
    c = make_client(data_dir, critical=True)
    r = c.get("/api/health")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    j = r.json()
    assert {"status", "headline", "as_of", "snapshot_age_days", "champions", "alerts", "models",
            "calibration", "bias", "drift", "checks"} <= set(j)
    assert j["status"] == "critical" and j["as_of"] == "2026-03-01"
    assert j["champions"][0]["version_id"] == "gbm_market@v1-abc12345"
    assert j["champions"][0]["code_sha"] == "2aa9a89"
    assert j["alerts"]["critical"] == 1 and j["alerts"]["open"] == 2
    m = {x["model"]: x for x in j["models"]}
    assert set(m) == {"gbm_market", "gbm_opinion", "market"}
    assert m["gbm_market"]["season"]["n"] > 0 and "ll_diff_se" in m["gbm_market"]["season"]
    assert m["market"]["vs_market"] is None
    assert j["drift"]["max_psi"] == pytest.approx(0.55)
    assert j["checks"]["fail"] == 1 and j["checks"]["ok"] == 2 and j["checks"]["skip"] == 1
    assert any(x["verdict"] for x in j["calibration"])
    json.dumps(j, allow_nan=False)


def test_api_health_empty(data_dir):
    c = make_client(data_dir, populate=False)
    j = c.get("/api/health").json()
    assert j["status"] == "unknown" and j["as_of"] is None
    assert j["models"] == [] and j["drift"] is None and j["champions"] == []


def test_health_is_fast(data_dir):
    import time
    c = make_client(data_dir)
    c.get("/health")
    t0 = time.perf_counter()
    c.get("/health")
    assert (time.perf_counter() - t0) < 0.15


# ------------------------------------------------------------------ plain-English helpers

def test_vs_market_wording():
    w = health.vs_market(0.005, 0.0025, 300)
    assert w["text"] == "0.005 worse than the market (≈2 SE)" and w["tone"] == "behind"
    assert health.vs_market(-0.012, 0.004, 300)["tone"] == "ahead"
    assert "within noise" in health.vs_market(0.001, 0.01, 300)["text"]
    assert health.vs_market(None, None, 300)["tone"] == "none"
    assert "Too early" in health.vs_market(0.01, 0.01, 12)["text"]


def test_calibration_verdicts():
    assert health.calib_verdict(0.4, 300)["label"] == "Well calibrated"
    assert health.calib_verdict(2.0, 300)["label"] == "Slightly overconfident"
    assert health.calib_verdict(-2.7, 300)["label"] == "Underconfident"
    assert health.calib_verdict(4.0, 300)["label"] == "Badly overconfident"
    assert health.calib_verdict(4.0, 40)["label"] == "Too early to judge"
    assert health.calib_verdict(None, 300)["tone"] == "none"


def test_bias_phrases():
    assert health._bias_phrase(0.3, 0.4, "totals", "Game total")[1] == "ok"
    assert health._bias_phrase(1.5, -1.0, "totals", "Game total")[0].startswith("Slight under")
    text, tone = health._bias_phrase(2.5, 2.7, "totals", "Game total")
    assert text == "Over-predicts totals by 2.7 pts" and tone == "warn"
