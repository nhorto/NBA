"""The checked-in sample Brief is the contract the web app is built against."""

import json
from pathlib import Path

from courtside.brief import assemble, show

SAMPLE = Path(__file__).parent / "fixtures" / "brief" / "sample_brief.json"


def test_sample_brief_matches_the_schema_version_and_sections():
    b = json.loads(SAMPLE.read_text())
    assert set(b) == {"meta", "headline", "last_night", "team_pulse", "player_pulse",
                      "injury_watch", "tonight", "scorecard"}
    assert b["meta"]["schema_version"] == assemble.SCHEMA_VERSION
    assert SAMPLE.stat().st_size < 300_000
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
