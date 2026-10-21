"""proj/why_v0.py: exact post-game decomposition, notable picks, template text, the LLM guard."""
from __future__ import annotations

import json
import random

import pytest

from courtside import db
from courtside.brief import factcheck
from courtside.proj import why_v0 as W
from courtside.proj.dfs.rules import dk_points

ST = W.STATS


def line(pts=0, fg3m=0, reb=0, ast=0, stl=0, blk=0, tov=0, mn=30.0, **extra):
    return {"pts": pts, "fg3m": fg3m, "reb": reb, "ast": ast, "stl": stl, "blk": blk,
            "tov": tov, "min": mn, **extra}


def exp_counts(pts=15.0, fg3m=1.5, reb=6.0, ast=4.0, stl=1.0, blk=0.5, tov=2.0):
    return {"pts": pts, "fg3m": fg3m, "reb": reb, "ast": ast, "stl": stl, "blk": blk, "tov": tov}


def exp_dk_of(c, bonus):
    return sum(W.WEIGHTS[s] * c[s] for s in ST) + bonus


def total(d):
    return sum(d.terms.values())


# ============================================================================ exact sums

CASES = {
    "plain": (30, 34.0, line(pts=18, fg3m=2, reb=7, ast=5, stl=1, blk=1, tov=2, mn=33)),
    "double_double_gained": (29, 33.0, line(pts=14, fg3m=1, reb=15, ast=3, stl=0, blk=0, tov=3, mn=36)),
    "triple_double_gained": (34, 45.0, line(pts=22, fg3m=3, reb=11, ast=12, stl=2, blk=1, tov=4, mn=39)),
    "double_double_lost": (36, 48.0, line(pts=9, fg3m=0, reb=9, ast=4, stl=0, blk=0, tov=1, mn=36)),
    "td_expected_dd_actual": (36, 52.0, line(pts=20, fg3m=2, reb=11, ast=10, stl=0, blk=0, tov=3, mn=34)),
    "dnp_after_projection": (28, 31.0, line(mn=0.0)),
    "overtime": (30, 33.0, line(pts=38, fg3m=5, reb=12, ast=7, stl=2, blk=1, tov=5, mn=52)),
    "garbage_minutes": (30, 33.0, line(pts=4, fg3m=0, reb=2, ast=1, stl=0, blk=0, tov=0, mn=11)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_exact_sum_named_cases(name):
    m_e, dk_e, row = CASES[name]
    d = W.decompose(m_e, dk_e, exp_counts(), row["min"], row)
    assert abs(total(d) - (d.actual_dk - d.expected_dk)) < 1e-6
    assert abs(d.residual) < 1e-9
    assert d.actual_dk == pytest.approx(dk_points(row))
    assert sum(d.act_comps.values()) == pytest.approx(d.actual_dk)
    assert sum(d.exp_comps.values()) == pytest.approx(d.expected_dk)


def test_dnp_is_a_pure_minutes_term():
    m_e, dk_e, row = CASES["dnp_after_projection"]
    d = W.decompose(m_e, dk_e, exp_counts(), 0.0, row)
    assert d.actual_dk == 0.0
    assert d.terms["minutes"] == pytest.approx(-dk_e)
    assert all(abs(v) < 1e-12 for k, v in d.terms.items() if k != "minutes")


def test_double_double_bonus_lands_in_the_bonus_term():
    _, dk_e, row = CASES["double_double_gained"]
    d = W.decompose(29, dk_e, exp_counts(), row["min"], row)
    assert d.act_comps["bonus"] == pytest.approx(1.5)
    tri = W.decompose(34, exp_dk_of(exp_counts(), 0.4), exp_counts(), 39,
                      CASES["triple_double_gained"][2])
    assert tri.act_comps["bonus"] == pytest.approx(4.5)     # +1.5 DD and +3.0 TD
    assert tri.terms["bonus"] > 0
    lost = W.decompose(36, exp_dk_of(exp_counts(), 1.2), exp_counts(), 36,
                       CASES["double_double_lost"][2])
    assert lost.act_comps["bonus"] == 0.0 and lost.terms["bonus"] < 0


def test_minutes_term_is_extra_minutes_at_the_expected_rate():
    row = line(pts=15, fg3m=1.5, reb=6, ast=4, stl=1, blk=0.5, tov=2, mn=36)   # same rates, +6 min
    c = exp_counts()
    dk_e = exp_dk_of(c, 0.5)
    d = W.decompose(30, dk_e, c, 36, row)
    # per-minute rates are identical at 36 minutes only if the line scales with minutes
    scaled = {s: c[s] * 36 / 30 for s in ST}
    d2 = W.decompose(30, dk_e, c, 36, {**scaled, "min": 36})
    assert d2.terms["minutes"] == pytest.approx(6 * dk_e / 30)
    for s in ST:
        assert d2.terms[s] == pytest.approx(0, abs=1e-9)
    assert abs(total(d) - d.delta) < 1e-9


def test_exact_sum_random_cases():
    rng = random.Random(7)
    for _ in range(500):
        m_e = rng.choice([0.0, 0.3, rng.uniform(1, 44)])
        c = {s: rng.uniform(0, 12) * (m_e / 30) for s in ST}
        bonus_e = rng.uniform(-1, 4) if m_e else 0.0
        dk_e = exp_dk_of(c, bonus_e) if m_e else 0.0
        mn = rng.choice([0.0, rng.uniform(1, 25), rng.uniform(25, 58)])
        row = line(pts=rng.randint(0, 55), fg3m=rng.randint(0, 9), reb=rng.randint(0, 22),
                   ast=rng.randint(0, 18), stl=rng.randint(0, 5), blk=rng.randint(0, 6),
                   tov=rng.randint(0, 8), mn=mn) if mn else line(mn=0.0)
        d = W.decompose(m_e, dk_e, c, mn, row)
        assert abs(total(d) + d.residual - d.delta) < 1e-9
        assert abs(total(d) - d.delta) < 1e-6, (m_e, mn)
        assert abs(d.residual) < 1e-6


def test_degenerate_no_expected_minutes_stays_exact():
    row = line(pts=10, reb=4, ast=2, mn=14)
    d = W.decompose(0.0, 0.0, exp_counts(0, 0, 0, 0, 0, 0, 0), 14, row)
    assert abs(total(d) - d.delta) < 1e-9
    assert d.terms["minutes"] == pytest.approx(d.actual_dk)


# ============================================================================ DB end to end

def _fresh(tmp_path):
    return db.connect(tmp_path / "why.db")


GID = "0022600001"
DAY = "2026-11-02"


def seed_game(conn, gid=GID, day=DAY, home="DET", away="BOS", hs=130, as_=104, n_ot=0,
              tip=f"{DAY}T00:00:00Z"):
    conn.execute("INSERT INTO games (game_id, season, nba_day, tipoff_utc, home, away, status) "
                 "VALUES (?, '2026-27', ?, ?, ?, ?, 'final')", (gid, day, tip, home, away))
    for team, opp, pts, hm in ((home, away, hs, 1), (away, home, as_, 0)):
        conn.execute("INSERT INTO team_games (game_id, team, season, season_type, game_date, "
                     "opp, is_home, min, pts) VALUES (?, ?, '2026-27', 'Regular Season', ?, ?, ?, "
                     "?, ?)", (gid, team, day, opp, hm, 240 + 25 * n_ot, pts))


def seed_player(conn, gid, pid, name, team, row, *, exp_min, exp_dk, stats=None, created="2026-11-01T22:00:00Z",
                run_id="r1", p_play=1.0, inputs=None):
    if row is not None:
        conn.execute(
            "INSERT INTO player_games (game_id, player_id, player_name, team, season, game_date, min, "
            "fg3m, reb, ast, stl, blk, tov, pts) VALUES (?, ?, ?, ?, '2026-27', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (gid, pid, name, team, DAY, row["min"], row["fg3m"], row["reb"], row["ast"],
             row["stl"], row["blk"], row["tov"], row["pts"]))
    cur = conn.execute(
        "INSERT INTO player_projections (game_id, nba_day, player_id, team, run_id, created_at, "
        "model_version, p_play, minutes_json, inputs_json) VALUES (?, ?, ?, ?, ?, ?, 'v0', ?, ?, ?)",
        (gid, DAY, pid, team, run_id, created, p_play, json.dumps({"mean_all": exp_min}),
         json.dumps(inputs or {})))
    pj = cur.lastrowid
    base = {"min": exp_min, "dk": exp_dk, **(stats or {})}
    for s, v in base.items():
        conn.execute("INSERT INTO player_projection_stats (projection_id, stat, cond, mean) "
                     "VALUES (?, ?, 'all', ?)", (pj, s, v))
    return pj


RATES = {s: r for s, r in zip(ST, (0.5, 0.06, 0.2, 0.12, 0.03, 0.02, 0.07))}


def seeded(tmp_path):
    conn = _fresh(tmp_path)
    seed_game(conn, n_ot=1)
    # Duren: monster double-double, many extra minutes
    seed_player(conn, GID, 1, "Jalen Duren", "DET",
                line(pts=22, fg3m=0, reb=21, ast=3, stl=1, blk=2, tov=2, mn=38),
                exp_min=29, exp_dk=34.0, stats={"pts": 13.0, "reb": 8.5, "ast": 2.0, "fg3m": 0.1})
    # Brown: quiet night
    seed_player(conn, GID, 2, "Jaylen Brown", "BOS",
                line(pts=8, fg3m=0, reb=3, ast=2, stl=0, blk=0, tov=4, mn=27),
                exp_min=34, exp_dk=42.0, stats={"pts": 24.0, "reb": 5.5, "ast": 4.0, "fg3m": 2.2})
    # small gap -> not notable
    seed_player(conn, GID, 3, "Role Player", "BOS",
                line(pts=9, fg3m=1, reb=4, ast=2, stl=1, blk=0, tov=1, mn=22),
                exp_min=22, exp_dk=21.0, stats={"pts": 9.0, "reb": 4.0, "ast": 2.0, "fg3m": 1.0})
    # projected rotation player who did not play (row present with 0 min)
    seed_player(conn, GID, 4, "Scratched Guy", "DET", line(mn=0.0, fg3m=0),
                exp_min=26, exp_dk=28.0, stats={"pts": 12.0, "reb": 5.0, "ast": 3.0, "fg3m": 1.0})
    # a later, post-tip run that must be ignored
    seed_player(conn, GID, 1, "Jalen Duren", "DET", None, exp_min=40, exp_dk=60.0, run_id="r2",
                created="2026-11-02T01:00:00Z", stats={"pts": 30.0, "reb": 12.0, "ast": 5.0, "fg3m": 0.1})
    conn.commit()
    rates = {p: dict(RATES) for p in (1, 2, 3, 4)}
    return conn, rates


def test_build_writes_notable_rows_with_exact_terms(tmp_path):
    conn, rates = seeded(tmp_path)
    s = W.build(conn, DAY, llm=False, rates=rates)
    assert s["games_with_run"] == 1 and s["players"] == 4
    assert s["max_abs_residual"] < 1e-9
    rows = {r[0]: r for r in conn.execute(
        "SELECT player_id, expected, actual, terms_json, text FROM postgame_why_v0")}
    assert set(rows) == {1, 2, 4}                    # Role Player is not notable
    duren = rows[1]
    assert duren[1] == pytest.approx(34.0)           # the pre-tip run, not the post-tip one
    blob = json.loads(duren[3])
    assert blob["run_id"] == "r1" and blob["pre_tip"] is True
    assert abs(sum(blob["terms"].values()) - (duren[2] - duren[1])) < 1e-6
    assert abs(blob["residual"]) < 1e-9
    assert blob["facts"]["overtime"] == 1 and blob["facts"]["blowout_margin"] == 26
    assert duren[4].startswith("Jalen Duren: ")
    assert "double-double bonus" in duren[4] or "rebounds" in duren[4]
    assert "from minutes (38 vs 29 expected)" in duren[4]
    assert rows[4][4].startswith("Scratched Guy: 0.0 DK vs 28.0 expected (-28.0): did not play")


def test_build_is_idempotent_and_top_stories(tmp_path):
    conn, rates = seeded(tmp_path)
    W.build(conn, DAY, llm=False, rates=rates)
    W.build(conn, DAY, llm=False, rates=rates)
    assert conn.execute("SELECT COUNT(*) FROM postgame_why_v0").fetchone()[0] == 3
    top = W.top_stories(conn, DAY, n=2)
    assert len(top) == 2                              # max 2 per game
    assert abs(top[0]["diff"]) >= abs(top[1]["diff"])
    assert top[0]["player"] and top[0]["text"] and top[0]["terms"]["minutes"] is not None
    assert W.top_stories(conn, "2026-11-03") == []


def test_historical_runs_after_tip_need_the_flag(tmp_path):
    conn = _fresh(tmp_path)
    seed_game(conn)
    seed_player(conn, GID, 1, "Jalen Duren", "DET",
                line(pts=22, reb=21, ast=3, stl=1, blk=2, tov=2, mn=38), exp_min=29, exp_dk=34.0,
                created="2026-11-05T12:00:00Z", stats={"pts": 13.0, "reb": 8.5, "ast": 2.0, "fg3m": 0.1})
    conn.commit()
    rates = {1: dict(RATES)}
    assert W.build(conn, DAY, llm=False, rates=rates)["written"] == 0
    s = W.build(conn, DAY, llm=False, rates=rates, allow_post_tip=True)
    assert s["written"] == 1
    blob = json.loads(conn.execute("SELECT terms_json FROM postgame_why_v0").fetchone()[0])
    assert blob["pre_tip"] is False


def test_cap_and_floor(tmp_path):
    conn = _fresh(tmp_path)
    seed_game(conn)
    for i in range(1, 41):
        row = line(pts=10 + i, reb=5, ast=3, stl=1, blk=0, tov=1, mn=30)
        seed_player(conn, GID, 100 + i, f"Player {i}", "DET" if i % 2 else "BOS", row,
                    exp_min=30, exp_dk=25.0, stats={"pts": 10.0, "reb": 5.0, "ast": 3.0, "fg3m": 0.0})
    conn.commit()
    rates = {100 + i: dict(RATES) for i in range(1, 41)}
    s = W.build(conn, DAY, llm=False, rates=rates, cap=30)
    assert s["notable"] == 30
    deltas = [abs(r["delta"]) for r in s["rows"]]
    assert deltas == sorted(deltas, reverse=True) and min(deltas) >= W.NOTABLE_DK


# ============================================================================ text style

def synth_rec(seed: int) -> dict:
    rng = random.Random(seed)
    m_e = rng.uniform(14, 38)
    c = {s: rng.uniform(0.3, 1.4) * RATES[s] * m_e for s in ST}
    dk_e = exp_dk_of(c, rng.uniform(0, 2))
    mn = rng.choice([rng.uniform(11, 20), rng.uniform(20, 44), rng.uniform(44, 54)])
    k = mn / 30
    row = line(pts=int(rng.uniform(2, 38) * k), fg3m=rng.randint(0, 6), reb=int(rng.uniform(1, 18) * k),
               ast=int(rng.uniform(0, 14) * k), stl=rng.randint(0, 4), blk=rng.randint(0, 4),
               tov=rng.randint(0, 7), mn=round(mn))
    p = {"player_id": seed, "team": "DET", "p_play": 0.95, "inputs": {}, "minutes": {},
         "stats": {("min", "all"): m_e, ("dk", "all"): dk_e, **{(s, "all"): c[s] for s in W.PROJ_STATS}}}
    ctx = {"margin": rng.choice([3, 11, 24, 31]), "n_ot": rng.choice([0, 0, 0, 1, 2]),
           "opp": {"DET": "BOS", "BOS": "DET"}}
    absent = rng.choice([[], [], ["Cade Cunningham"], ["Cade Cunningham", "Tobias Harris"]])
    rec = W.build_record(p, "0022600001", dict(row, team="DET", player_name=f"Test Player{seed}"),
                         dict(RATES), ctx, name=f"Alvin Tester{chr(65 + seed % 26)}",
                         run={"run_id": "r1", "pre_tip": True}, absent=absent,
                         started=rng.random() < 0.5)
    rec["name"] = rng.choice(["Jalen Duren", "Jaylen Brown", "Nikola Jokic", "Ausar Thompson"])
    return rec


def test_template_text_style_and_numbers():
    for seed in range(1, 151):
        rec = synth_rec(seed)
        text = W.template_text(rec)
        assert text.startswith(rec["name"] + ": ")
        assert factcheck.BETTING_RE.search(text) is None
        assert not factcheck._CLICHE_RE.search(text)
        assert "!" not in text and "**" not in text
        # every number is a displayed fact, and no negative term is described as helping
        res = factcheck.check(text, W.llm_facts(rec), kind="blurb")
        assert [v for v in res.violations if v.kind in ("number", "pair", "betting")] == [], text
        assert W.check_signs(text, W.signed_figures(rec), W.rounded_terms(rec)) == [], text
        assert factcheck.count_sentences(text) <= 3


def test_template_example_shape():
    row = line(pts=22, reb=21, ast=3, stl=1, blk=2, tov=2, mn=38)
    c = exp_counts(pts=13, fg3m=0.1, reb=8.5, ast=2, stl=0.8, blk=0.5, tov=1.5)
    p = {"player_id": 1, "team": "DET", "p_play": 1.0, "inputs": {}, "minutes": {},
         "stats": {("min", "all"): 29.0, ("dk", "all"): 34.0, **{(s, "all"): c[s] for s in W.PROJ_STATS}}}
    rec = W.build_record(p, "g", dict(row, team="DET"), dict(RATES), {"margin": 6, "n_ot": 0, "opp": {}},
                         name="Jalen Duren", run={}, absent=[], started=True)
    text = W.template_text(rec)
    assert text.startswith("Jalen Duren: ") and " DK vs 34.0 expected (+" in text
    assert "from minutes (38 vs 29 expected)" in text


# ============================================================================ the LLM guard

def faithful(facts: dict) -> str:
    """What a good writer says: biggest contributions, signs respected."""
    dk = facts["dk"]
    verb = "beat" if dk["diff"] > 0 else "missed"
    parts = []
    for c in facts["contributions"][:2]:
        v = c["dk_points"]
        parts.append(f"{c['source']} {'added' if v > 0 else 'cost him'} {abs(v):.1f}")
    return (f"{facts['player']} {verb} our number with {dk['actual']} DK points against "
            f"{dk['expected']} expected: " + " and ".join(parts) + ".")


def make_stub(recs, mode):
    """mode(i, facts) -> text; the stub answers under each record's key."""
    keys = [f"{r['game_id']}:{r['player_id']}" for r in recs]
    facts = {k: W.llm_facts(r) for k, r in zip(keys, recs)}
    calls = []

    def complete(prompt, system=None, timeout=0):
        calls.append(prompt)
        ids = [k for k in keys if f'"id":"{k}"' in prompt]
        return json.dumps({"stories": {k: mode(keys.index(k), facts[k], len(calls)) for k in ids}})
    return complete, calls, keys


def records(n=50):
    out = []
    for i in range(1, n + 1):
        r = synth_rec(i)
        r["player_id"] = i
        out.append(r)
    return out


def flip_sign(f):
    c = f["contributions"][0]
    v = c["dk_points"]
    verb = "cost him" if v > 0 else "added"      # the wrong way round
    return f"{f['player']} had {f['dk']['actual']} DK points because {c['source']} {verb} {abs(v):.1f}."


def _negative_as_help(f):
    neg = [c for c in f["contributions"] if c["dk_points"] < 0]
    c = (neg or f["contributions"])[0]
    verb = "added" if c["dk_points"] < 0 else "cost him"
    return f"{f['player']} had {f['dk']['actual']} DK points; {c['source']} {verb} {abs(c['dk_points']):.1f}."


BAD = {
    "unsupported_number": lambda f: faithful(f).replace(str(f["dk"]["actual"]), "99.9"),
    "negative_as_help": lambda f: _negative_as_help(f),
    "wrong_signed_figure": lambda f: (
        f"{f['player']} had {f['dk']['actual']} DK points: "
        f"{'-' if f['contributions'][0]['dk_points'] > 0 else '+'}"
        f"{abs(f['contributions'][0]['dk_points']):.1f} from {f['contributions'][0]['source']}."),
    "flipped_verb": flip_sign,
    "betting": lambda f: faithful(f) + " A bet to fade.",
    "invented_name": lambda f: faithful(f) + " Giannis Antetokounmpo watched.",
    "no_json_key": None,
}


def test_guard_accepts_faithful_text_50_samples():
    recs = records(50)
    complete, calls, keys = make_stub(recs, lambda i, f, n: faithful(f))
    report = {}
    got = W.rewrite(recs, complete=complete, report=report)
    assert len(calls) == 1
    bad = []
    for k, r in zip(keys, recs):
        t = got.get(k)
        assert t is not None, (k, report["problems"].get(k))
        facts = W.llm_facts(r)
        bad += W.validate_story(t, r, facts)
        assert W.check_signs(t, W.signed_figures(r), W.rounded_terms(r)) == []
        assert factcheck.check_numbers(t, factcheck.FactIndex.build(facts)) == []
    assert bad == []


@pytest.mark.parametrize("kind", [k for k, v in BAD.items() if v])
def test_guard_rejects_bad_text_and_falls_back(kind):
    recs = records(12)
    complete, calls, keys = make_stub(recs, lambda i, f, n: BAD[kind](f))
    report = {}
    got = W.rewrite(recs, complete=complete, report=report)
    assert len(calls) == 2                      # first draft + one retry
    assert got == {}, (kind, got)
    assert sorted(report["fell_back"]) == sorted(keys)


def test_mixed_batch_keeps_good_and_drops_bad():
    recs = records(50)
    # every third story is deliberately bad, cycling through the failure kinds
    kinds = [k for k, v in BAD.items() if v]

    def mode(i, f, n):
        return BAD[kinds[(i // 3) % len(kinds)]](f) if i % 3 == 0 else faithful(f)
    complete, calls, keys = make_stub(recs, mode)
    got = W.rewrite(recs, complete=complete)
    for i, (k, r) in enumerate(zip(keys, recs)):
        if i % 3 == 0:
            assert k not in got
        else:
            assert k in got
    # nothing that survived has a sign or number problem
    for k, r in zip(keys, recs):
        if k in got:
            assert W.validate_story(got[k], r) == []


def test_retry_can_repair():
    recs = records(6)
    complete, calls, keys = make_stub(
        recs, lambda i, f, n: BAD["flipped_verb"](f) if n == 1 else faithful(f))
    got = W.rewrite(recs, complete=complete)
    assert len(calls) == 2 and len(got) == 6


def test_llm_down_keeps_templates(tmp_path):
    from courtside.brief import llm as L

    def down(prompt, system=None, timeout=0):
        raise L.AllBackendsFailed("no backend")
    conn, rates = seeded(tmp_path)
    s = W.build(conn, DAY, llm=True, complete=down, rates=rates)
    assert s["llm"]["kept"] == 0 and s["written"] == 3
    assert all(r["source"] == "template" for r in s["rows"])


def test_build_with_stub_llm_marks_source(tmp_path):
    conn, rates = seeded(tmp_path)

    def complete(prompt, system=None, timeout=0):
        # answer only for entries whose figures the faithful writer can quote
        import re
        ids = re.findall(r'"id":"([^"]+)"', prompt)
        data = json.loads(re.search(r'\[\{"id".*\}\]', prompt, re.S).group(0))
        return json.dumps({"stories": {d["id"]: faithful(d) for d in data}})
    s = W.build(conn, DAY, llm=True, complete=complete, rates=rates)
    srcs = {r["name"]: r["source"] for r in s["rows"]}
    assert srcs["Jalen Duren"] == "llm" and srcs["Jaylen Brown"] == "llm"
    assert srcs["Scratched Guy"] == "template"      # did not play: never sent to the LLM
    blob = json.loads(conn.execute("SELECT terms_json FROM postgame_why_v0 WHERE player_id=1")
                      .fetchone()[0])
    assert blob["text_source"] == "llm"


def test_check_signs_unit():
    signed = [6.1, -3.2, 2.9]          # minutes +6.1, assists -3.2, total +2.9
    ok = "He gained 6.1 from minutes, while assists cost him 3.2."
    assert W.check_signs(ok, signed) == []
    assert W.check_signs("Assists added 3.2 to the total.", signed)
    assert W.check_signs("Minutes cost him 6.1.", signed)
    assert W.check_signs("That was +3.2 from assists.", signed)
    assert W.check_signs("That was -6.1 from minutes.", signed)
    assert W.check_signs("-3.2 from assists, +6.1 from minutes", signed) == []
    assert W.check_signs("He underperformed our number.", signed)
    assert W.check_signs("He outperformed our number.", [-4.0, -4.0]) != []
    # a figure attached to a named source is checked against that source alone
    terms = {"minutes": -10.0, "reb": 7.2}
    sg = [-10.0, 7.2, 10.0]
    assert W.check_signs("That was +10.0 from minutes.", sg, terms)
    assert W.check_signs("Minutes cost him 10.0 and rebounds added 7.2.", sg, terms) == []
    assert W.check_signs("Rebounds cost him 7.2.", sg, terms)
    assert W.check_signs("He finished with 35.5 DK points, +10.0 against our number.", sg, terms) == []


def test_absent_regulars_ignores_players_traded_away_in_the_summer():
    """Opener 2026-10-20: last season's BOS regular Jaylen Brown (now PHI) was reported as
    'did not play' for BOS. The current roster snapshot and tonight's box rule him out."""
    import sqlite3
    from courtside.proj import why_v0 as W
    c = sqlite3.connect(":memory:")
    c.executescript("""
        CREATE TABLE team_games (game_id TEXT, team TEXT, game_date TEXT);
        CREATE TABLE player_games (game_id TEXT, team TEXT, player_id INTEGER, player_name TEXT,
                                   min REAL);
        CREATE TABLE team_rosters (season TEXT, team_id INTEGER, team TEXT, player_id INTEGER,
                                   player TEXT);
    """)
    for i in range(5):
        c.execute("INSERT INTO team_games VALUES (?, 'BOS', ?)", (f"g{i}", f"2026-04-0{i + 1}"))
        for pid, name in ((1, "Jaylen Brown"), (2, "Jayson Tatum"), (3, "Derrick White")):
            c.execute("INSERT INTO player_games VALUES (?, 'BOS', ?, ?, 34)", (f"g{i}", pid, name))
    box = {2: {"team": "BOS", "min": 33}}            # Tatum played; White did not
    # no roster snapshot yet: the heuristic keeps everyone who did not play
    assert set(W.absent_regulars(c, "x", "BOS", "2026-10-20", box)) == {"Jaylen Brown",
                                                                         "Derrick White"}
    # Brown played for PHI tonight -> not a BOS absence
    box_phi = {**box, 1: {"team": "PHI", "min": 30}}
    assert W.absent_regulars(c, "x", "BOS", "2026-10-20", box_phi) == ["Derrick White"]
    # roster snapshot without Brown -> not a BOS absence even if he sat for PHI
    for pid in (2, 3):
        c.execute("INSERT INTO team_rosters VALUES ('2026-27', 1, 'BOS', ?, '')", (pid,))
    assert W.absent_regulars(c, "x", "BOS", "2026-10-20", box) == ["Derrick White"]
