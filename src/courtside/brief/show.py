"""Plain-text rendering of a Brief (`courtside brief show`). Reads the JSON only."""

from __future__ import annotations


def _pct(x, nd=0):
    return "-" if x is None else f"{100 * x:.{nd}f}%"


def _num(x, nd=1, sign=False):
    if x is None:
        return "-"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _teams(g):
    return g["away"]["team"], g["home"]["team"]


def format_brief(b: dict, width: int = 100) -> str:
    L: list[str] = []
    m = b["meta"]
    L.append(f"COURTSIDE MORNING BRIEF  {m['day']}   (schema {m['schema_version']}, "
             f"built {m['built_at']}, model {m['model']})")
    if b["headline"].get("text"):
        L.append(b["headline"]["text"])
    for n in m["notes"]:
        L.append(f"  note: {n}")

    ln = b["last_night"]
    s = ln["summary"]
    L += ["", f"LAST NIGHT  {ln['day']}  ({s['games']} games; our picks {s['picks_correct']}/"
          f"{s['picks_scored']}, market favourites {s['market_favorites_correct']}/"
          f"{s['market_favorites_scored']})"]
    for g in ln["games"]:
        a, h = _teams(g)
        res = g["result"]
        score = f"{g['away']['score']}-{g['home']['score']}" if res else g["status"]
        line = f"  {a} @ {h}  {score}"
        if res and res["overtimes"]:
            line += f" ({res['overtimes']}OT)"
        p = g["prediction"]
        if p:
            mark = {True: "RIGHT", False: "WRONG", None: ""}[g["correct"]]
            line += (f"   pick {p['pick']} {_pct(p['pick_prob'])} {p['confidence']} {mark}")
        mk = g["market"]
        if mk and mk.get("home_spread") is not None:
            line += f"   line {h} {_num(mk['home_spread'], 1, True)}"
        L.append(line)
        f = g.get("facts")
        if f:
            tp = f.get("turning_point") or {}
            uh = (f.get("unsung_hero") or [])
            uh = uh[0] if isinstance(uh, list) and uh else (uh if isinstance(uh, dict) else None)
            bits = []
            if tp.get("run"):
                fr = tp.get("from") or {}
                bits.append(f"turning point: {tp.get('team')} {tp['run']} run "
                            f"({tp.get('reason', '').replace('_', ' ')}, "
                            f"from {fr.get('label', '')} {fr.get('clock', '')})")
            if uh and uh.get("name"):
                bits.append(f"unsung: {uh['name']}")
            if bits:
                L.append("      " + "; ".join(bits)[: width - 6])
        t = g["text"]
        if t.get("headline"):
            L.append(f"      \"{t['headline']}\"")

    tp = b["team_pulse"]
    L += ["", "TEAM PULSE"]
    if tp.get("available"):
        for key, lab in (("rising", "rising"), ("sliding", "sliding")):
            for t in tp[key]:
                L.append(f"  {lab:8} {t['team']} {t['record']} (L10 {t['last10']}, "
                         f"{t['streak']})  z={_num(t['trend']['z'], 2)} "
                         f"delta={_num(t['trend']['delta'], 1, True)}")
        L.append("  power: " + ", ".join(f"{p['rank']}.{p['team']}" for p in tp["power_top10"]))
        for n in tp["notes"]:
            L.append(f"  note: {n}")
    else:
        L.append(f"  unavailable ({tp.get('reason')})")

    pp = b["player_pulse"]
    L += ["", "PLAYER PULSE"]
    if pp.get("available"):
        for key in ("hot", "cold"):
            for p in pp[key]:
                L.append(f"  {key:5} {p['player']} ({p['team']})  {p.get('detail', '')}"[:width])
        for r in pp["role_changes"][:6]:
            L.append(f"  role  {r['player']} ({r['team']}): {'; '.join(r['details'])}"[:width])
        imp = pp["leaders_last10"].get("impact", [])[:5]
        if imp:
            L.append("  impact leaders (L10): " + ", ".join(
                f"{r['player']} {_num(r['value'])}" for r in imp))
    else:
        L.append(f"  unavailable ({pp.get('reason')})")

    iw = b["injury_watch"]
    L += ["", f"INJURY WATCH  (latest snapshot {iw['latest_snapshot']}, "
          f"baseline {iw['baseline_snapshot']})"]
    for key, lab in (("new_absences", "NEW"), ("returns", "BACK"), ("status_changes", "CHG")):
        for c in iw["changes"][key][:8]:
            L.append(f"  {lab:4} {c['team']} {c['player']}: {c['from_status'] or 'not listed'} -> "
                     f"{c['to_status']}  (mpg {_num(c['mpg'])}, impact {_num(c['impact'])})")
    if iw["out_tonight"]:
        L.append("  highest-impact players out/questionable tonight:")
        for e in iw["out_tonight"][:8]:
            L.append(f"    {e['team']:3} {e['player']:24} {e['status']:12} "
                     f"mpg {_num(e['mpg'])}  impact {_num(e['impact'])}")
    elif not iw["available"]:
        L.append("  no injury data")

    tn = b["tonight"]
    L += ["", f"TONIGHT  {tn['day']}  ({tn['summary']['games']} games, "
          f"{tn['summary']['upset_watch']} on upset watch)"]
    for g in tn["games"]:
        a, h = _teams(g)
        p, mk, e = g["prediction"], g["market"], g["edge"]
        line = f"  {a} @ {h}"
        if p:
            line += (f"   {h} {_pct(p['home_win_prob'])}  margin {_num(p['home_margin'], 1, True)}"
                     f"  total {_num(p['total'])}  [{p['confidence']}]")
        if mk:
            line += (f"   | mkt {h} {_pct(mk.get('home_prob'))} spread "
                     f"{_num(mk.get('home_spread'), 1, True)} total {_num(mk.get('total'))}")
        if e and e.get("win_prob") is not None:
            line += f"   edge {e['win_prob'] * 100:+.0f}pt"
        if g["upset_watch"]:
            line += f"   UPSET WATCH: {g['upset']['underdog']}"
        L.append(line)
        why = [f"{f['team']}: {f['detail']}" for f in g["factors"]
               if f["kind"] not in ("form", "model_vs_market")]
        if why:
            L.append("      " + "; ".join(why)[: width - 6])

    sc = b["scorecard"]
    L += ["", f"SCORECARD  {sc['season']} through {sc['through']}  (model {sc['model']})"]
    if sc["n"]:
        L.append(f"  record {sc['correct']}-{sc['incorrect']} ({_pct(sc['accuracy'], 1)})  "
                 f"Brier {_num(sc['brier'], 4)}  log-loss {_num(sc['log_loss'], 4)}  "
                 f"ECE {_num(sc['ece'], 3)}")
        v = sc["vs_market"]
        if v:
            L.append(f"  vs market on {v['n']} games: Brier {_num(v['model']['brier'], 4)} vs "
                     f"{_num(v['market']['brier'], 4)};  acc {_pct(v['model']['accuracy'], 1)} "
                     f"vs {_pct(v['market']['accuracy'], 1)}")
        r = sc["recent"]
        L.append(f"  last night {r['last_night']['correct']}/{r['last_night']['n']}, "
                 f"last 7 days {r['last_7_days']['correct']}/{r['last_7_days']['n']}")
    else:
        L.append("  no scored predictions yet")
    return "\n".join(L)
