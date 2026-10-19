"""The checked-in sample Briefs are the contract the web app is built against.

* `sample_brief.json`      schema 1.0, results-only `elo_sched` (injuries listed, not priced): the
  caveat path. Kept as is: 1.1 only adds keys, so a 1.0 document stays a valid input.
* `sample_brief_gbm.json`  the current schema, built from the production models with the
  2026-02-19/20 official injury reports priced in (day 2026-02-20, real data).
"""

import json
from pathlib import Path

import pytest

from courtside.brief import assemble, show

FIX = Path(__file__).parent / "fixtures" / "brief"
SAMPLE = FIX / "sample_brief.json"
SAMPLE_GBM = FIX / "sample_brief_gbm.json"
SECTIONS = {"meta", "headline", "last_night", "team_pulse", "player_pulse", "injury_watch",
            "tonight", "scorecard"}


@pytest.mark.parametrize("path,version,model", [(SAMPLE, "1.0", "elo_sched"),
                                                (SAMPLE_GBM, assemble.SCHEMA_VERSION,
                                                 "gbm_market")])
def test_sample_brief_matches_the_schema_version_and_sections(path, version, model):
    b = json.loads(path.read_text())
    assert set(b) == SECTIONS
    assert b["meta"]["schema_version"] == version and b["meta"]["model"] == model
    assert path.stat().st_size < 300_000
    assert b["last_night"]["games"] and b["tonight"]["games"]
    assert b["scorecard"]["n"] > 100 and b["scorecard"]["vs_market"]["n"] > 100
    assert b["injury_watch"]["diff_available"] and b["injury_watch"]["out_tonight"]
    for g in b["last_night"]["games"]:
        assert g["facts"] and g["text"] == {"headline": None, "story": None}
        assert g["prediction"] and g["result"] and g["correct"] in (True, False)
    for g in b["tonight"]["games"]:
        assert g["prediction"] and g["market"] and g["edge"] and g["factors"]
        assert {"home", "away"} == set(g["injuries"])
    assert "COURTSIDE MORNING BRIEF" in show.format_brief(b)


def test_gbm_sample_has_the_1_1_fields():
    b = json.loads(SAMPLE_GBM.read_text())
    m = b["meta"]
    assert m["model_uses_injuries"] is True and m["opinion_model"] == "gbm_opinion"
    assert m["model_uses_market"] is True and m["availability_basis"]["injury_report"] > 0
    assert b["scorecard"]["opinion"]["model"] == "gbm_opinion"
    for g in b["tonight"]["games"]:
        assert g["opinion"]["model"] == "gbm_opinion" and g["edge"]["model"] == "gbm_opinion"
        assert g["availability"]["home"]["basis"] in ("injury_report", "regulars_played",
                                                      "full_roster")
        mf = [f for f in g["factors"] if f["kind"] == "model_factor"]
        assert mf and len(mf) <= 4 and g["factors"][:len(mf)] == mf
        assert all({"factor", "impact_pts", "detail", "favors"} <= set(f) for f in mf)
    outs = b["injury_watch"]["out_tonight"]
    assert any(e["model_effect_margin"] is not None for e in outs)
