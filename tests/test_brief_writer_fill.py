"""fill() against the real sample Brief (docs/BRIEF.md) and null/missing-field fuzzing."""
from __future__ import annotations

import copy
import glob
import json
import re
from pathlib import Path

import pytest

from courtside.brief import llm, templates, writer

HERE = Path(__file__).parent
SAMPLE = HERE / "fixtures" / "brief" / "sample_brief.json"
GAMES = sorted(glob.glob(str(HERE / "fixtures" / "brief_writer" / "game_facts_*.json")))

needs_sample = pytest.mark.skipif(not SAMPLE.exists(), reason="sample brief fixture not present")


def sample():
    return json.loads(SAMPLE.read_text())


def echo_backend(brief):
    """A fake LLM that answers every prompt with the template prose for the items it is shown."""
    facts = {c["facts"]["game_id"]: c["facts"] for c in brief["last_night"]["games"] if c["facts"]}

    def complete(prompt, *, system=None, timeout=0):
        if "Write the recap" in prompt:
            ids = re.findall(r'"game_id":"(\d+)"', prompt)
            return json.dumps({"games": {i: templates.game(facts[i]) for i in ids}})
        if "Write the Pulse section" in prompt:
            tsec = prompt.split("TEAM ENTRIES (key = tricode)\n")[1].split("\n\n")[0]
            psec = prompt.split("PLAYER ENTRIES (key = player key)\n")[1].split("\n\n")[0]
            t = json.loads(tsec) if tsec != "(none)" else []
            p = json.loads(psec) if psec != "(none)" else []
            return json.dumps({"headline": templates.pulse_headline(t, p),
                               "teams": {e["team"]: templates.team_blurb(e) for e in t},
                               "players": {e["key"]: templates.player_blurb(e) for e in p}})
        if "preview line" in prompt:
            n = json.loads(prompt.split("FACTS (JSON, tonight's slate)\n")[1].split("\n\nOUTPUT FORMAT")[0])
            return json.dumps({"games": {x["game_id"]: templates.tonight(writer._tonight_template_input(x))
                                         for x in n}})
        if "headline for this morning" in prompt:
            s = json.loads(prompt.split("SUMMARY (JSON)\n")[1].split("\n\nOUTPUT FORMAT")[0])
            return json.dumps({"headline": templates.brief_headline(s)})
        raise AssertionError("unexpected prompt")
    return complete


def owned_slots(b):
    """Every slot fill() is responsible for."""
    for c in b["last_night"]["games"]:
        if c["facts"]:
            yield ("game", c["game_id"], "headline"), c["text"]["headline"]
            yield ("game", c["game_id"], "story"), c["text"]["story"]
    for lst in ("rising", "sliding"):
        for e in b["team_pulse"][lst]:
            yield ("team", e["team"], lst), e["text"]
    for lst in ("hot", "cold"):
        for e in b["player_pulse"][lst]:
            yield ("player", e["player"], lst), e["text"]
    for g in b["tonight"]["games"]:
        yield ("tonight", g["game_id"], "preview"), g["text"]["preview"]
    yield ("brief", "headline", ""), b["headline"]["text"]


@needs_sample
def test_fill_sample_brief_fills_every_owned_slot_and_stays_valid_json(monkeypatch):
    monkeypatch.delenv("COURTSIDE_NO_WRITER", raising=False)
    b = sample()
    before = json.dumps(b, sort_keys=True)
    inner, seen = echo_backend(b), []

    def complete(prompt, **kw):
        seen.append(prompt)
        return inner(prompt, **kw)
    out = writer.fill(b, complete=complete)
    assert len(seen) == 4                                         # stories, pulse, tonight, headline
    assert '"model_prices_injuries":false' in seen[2]             # the results-only model caveat
    assert json.dumps(b, sort_keys=True) == before
    json.dumps(out, allow_nan=False)
    missing = [k for k, v in owned_slots(out) if not (isinstance(v, str) and v.strip())]
    assert missing == []
    counts = out["meta"]["writer"]["counts"]
    assert counts.get("llm_ok", 0) >= 30 and set(counts) <= {"llm_ok", "llm_retry_ok", "template"}
    # nothing outside the text slots changed
    for sec in ("scorecard", "injury_watch"):
        assert out[sec] == json.loads(before)[sec]
    assert out["last_night"]["summary"] == json.loads(before)["last_night"]["summary"]


@needs_sample
def test_fill_respects_no_writer_env_unless_a_backend_is_injected(monkeypatch):
    monkeypatch.setenv("COURTSIDE_NO_WRITER", "1")
    b = sample()
    assert writer.fill(b) == b                                   # real CLIs off: untouched
    out = writer.fill(b, complete=echo_backend(b))               # explicit fake still works
    assert out["headline"]["text"]


@needs_sample
def test_llm_complete_refuses_real_clis_under_no_writer(monkeypatch):
    monkeypatch.setenv("COURTSIDE_NO_WRITER", "1")
    with pytest.raises(llm.AllBackendsFailed) as e:
        llm.complete("anything")
    assert "COURTSIDE_NO_WRITER" in json.dumps(e.value.errors)


@needs_sample
def test_fill_sample_with_dead_llm_still_fills_everything_from_templates(monkeypatch):
    monkeypatch.delenv("COURTSIDE_NO_WRITER", raising=False)
    b = sample()
    dead = llm.AllBackendsFailed({"claude": "Not logged in"})

    def complete(prompt, **kw):
        raise dead
    out = writer.fill(b, complete=complete)
    assert [k for k, v in owned_slots(out) if not (isinstance(v, str) and v.strip())] == []
    assert set(out["meta"]["writer"]["counts"]) == {"template"}


@needs_sample
def test_sample_cards_with_null_largest_lead_entry_do_not_crash():
    b = sample()
    cards = [c for c in b["last_night"]["games"]
             if c["facts"] and any(v is None for v in (c["facts"].get("largest_lead") or {}).values())]
    assert cards, "fixture should include a team that never led"
    for c in cards:
        compact = writer.compact_game(c["facts"])
        json.dumps(compact)
        assert templates.game(c["facts"])["story"]


# ---- null / missing fuzz over every fixture --------------------------------------------------

def _paths(d, depth=2, prefix=()):
    for k, v in d.items():
        yield prefix + (k,)
        if depth > 1 and isinstance(v, dict):
            yield from _paths(v, depth - 1, prefix + (k,))


def _mutate(f, path, mode):
    g = copy.deepcopy(f)
    cur = g
    for k in path[:-1]:
        cur = cur[k]
    if mode == "null":
        cur[path[-1]] = None
    else:
        del cur[path[-1]]
    return g


@pytest.mark.parametrize("gamefile", GAMES)
def test_every_null_or_missing_field_is_survivable(gamefile):
    f = json.loads(Path(gamefile).read_text())
    n = 0
    for path in _paths(f):
        if path in (("game_id",),) or path[0] == "game_id":
            continue
        for mode in ("null", "delete"):
            g = _mutate(f, path, mode)
            json.dumps(writer.compact_game(g))
            t = writer._safe_game_template(g)
            assert t["headline"] and t["story"]
            # and the whole pipeline, with the model down and with it answering nonsense
            out = writer.write_game_stories([g], complete=lambda p, **kw: "{}")
            assert out[g["game_id"]]["source"] == "template"
            n += 1
    assert n > 40


def test_facts_list_with_junk_entries_is_tolerated():
    f = json.loads(Path(GAMES[0]).read_text())
    out = writer.write_game_stories([None, {}, {"game_id": None}, f], complete=lambda p, **kw: "{}")
    assert list(out) == [f["game_id"]]
