"""Templated "why this lineup" v0 (docs/PLAYERS.md §2.4): no LLM, numbers only from the data.

`why_lineup(lineup, pool)` returns a JSON-able dict that later runs store in
``dfs_lineups.why_json``; `render_lines(why)` turns it into display strings.

Inputs (plain dicts, so the web page and the runner can both call it):

* lineup: ``{"slots": [{"slot", "player_id", "name", "team", "salary", "mean"}, ...],
  "salary": int, "mean": float}`` (the parsed ``slots_json`` of a ``dfs_lineups`` row).
* pool: the slate's players, ``{"id", "name", "team", "salary", "proj", "p_play", "status",
  "game", "min_mean", "inputs"}``. Only ``id``, ``salary`` and ``proj`` are required.
  ``min_mean`` is the player's projected minutes when he plays, and ``inputs`` is the parsed
  ``player_projections.inputs_json``. The minutes reason reads ``inputs["ewma_min"]`` (his recent
  minutes average) and, if present, ``inputs["absent"]`` (teammates out: names, or dicts with a
  ``name``). Without ``ewma_min`` the minutes reason is omitted; without ``absent`` it is stated
  as a plain "above his recent average" with no cause.

Output::

    {"version": 0,
     "reasons": [{"kind": "minutes"|"price"|"game", "text": str, "facts": {...}}, ...],  # <= 3
     "flags": [{"name", "p_play", "status", "text"}],     # P(play) < 0.9 or listed Q / GTD
     "summary": {"salary_used", "cap", "salary_left", "mean", "games": [{"label", "n"}], "text"}}

Every number printed in a text is in that reason's ``facts`` (rounded as printed), and every fact
comes from the lineup or pool dicts; nothing is estimated here. The voice follows docs/STYLE.md:
information, no advice.
"""

from __future__ import annotations

import json
from statistics import median
from typing import Any, Iterable, Mapping

VERSION = 0
CAP = 50_000
MAX_REASONS = 3
MIN_DELTA_MIN = 4.0          # minutes above his recent average that earn a reason (§2.4)
P_PLAY_FLAG = 0.9
MIN_SALARY_FOR_RATE = 3000   # per-$1k rates are meaningless for the minimum-salary tail
TOP_RANK = 10                # a rank is quoted only when the player is in the slate's top 10


def money(x: float | int) -> str:
    return f"${int(round(x)):,}"


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _num(x: Any) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v or v in (float("inf"), float("-inf")) else v


def per_k(proj: Any, salary: Any) -> float | None:
    p, s = _num(proj), _num(salary)
    if p is None or not s or s <= 0:
        return None
    return p / (s / 1000.0)


def slate_rates(pool: Iterable[Mapping]) -> tuple[float | None, dict[str, int], int]:
    """(median points per $1k, {player id: rank by points per $1k}, n ranked) over the pool's
    players with a positive projection and a salary of at least $3,000."""
    rows = []
    for p in pool:
        v = per_k(p.get("proj"), p.get("salary"))
        if v is not None and v > 0 and (_num(p.get("salary")) or 0) >= MIN_SALARY_FOR_RATE:
            rows.append((v, str(p.get("id"))))
    if not rows:
        return None, {}, 0
    rows.sort(key=lambda t: (-t[0], t[1]))
    return median(v for v, _ in rows), {pid: i + 1 for i, (_, pid) in enumerate(rows)}, len(rows)


def _names(raw: Any) -> list[str]:
    out = []
    for it in raw or []:
        if isinstance(it, str):
            out.append(it)
        elif isinstance(it, Mapping) and it.get("name"):
            out.append(str(it["name"]))
    return out


def _surname(name: str) -> str:
    parts = name.split()
    while len(parts) > 1 and parts[-1].lower().rstrip(".") in {"jr", "sr", "ii", "iii", "iv", "v"}:
        parts = parts[:-1]
    return parts[-1] if parts else name


def _list_and(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def why_lineup(lineup: Mapping, pool: Iterable[Mapping], *, cap: int = CAP,
               max_reasons: int = MAX_REASONS) -> dict:
    pool = list(pool)
    by_id = {str(p.get("id")): p for p in pool}
    med, ranks, n_ranked = slate_rates(pool)
    slots = list(lineup.get("slots") or [])

    members = []
    for s in slots:
        pid = str(s.get("player_id"))
        p = by_id.get(pid, {})
        members.append({"slot": s, "pool": p, "pid": pid,
                        "name": s.get("name") or p.get("name") or pid,
                        "salary": _num(s.get("salary")) if s.get("salary") is not None
                        else _num(p.get("salary")),
                        "proj": _num(s.get("mean")) if s.get("mean") is not None
                        else _num(p.get("proj"))})

    # ---- minutes opened: projected minutes well above his recent average
    minutes = []
    for m in members:
        p = m["pool"]
        inputs = p.get("inputs") or {}
        ewma, mm = _num(inputs.get("ewma_min")), _num(p.get("min_mean"))
        if ewma is None or mm is None or mm - ewma < MIN_DELTA_MIN:
            continue
        absent = _names(inputs.get("absent") or inputs.get("absent_teammates"))
        delta = round(mm - ewma)
        name = _surname(m["name"])
        facts = {"player": m["name"], "minutes": round(mm), "recent_avg": round(ewma),
                 "delta_min": delta}
        if absent:
            facts["absent"] = absent
            text = (f"+{delta} min for {name} against his recent average ({round(ewma)} to "
                    f"{round(mm)} projected), with {_list_and([_surname(a) for a in absent])} out")
        else:
            text = (f"{name} is projected for {round(mm)} min, {delta} above his recent "
                    f"average of {round(ewma)}")
        minutes.append((mm - ewma, {"kind": "minutes", "text": text, "facts": facts}))
    minutes.sort(key=lambda t: -t[0])

    # ---- price: points per $1k against the slate median
    price_items = []
    if med is not None:
        for m in members:
            v = per_k(m["proj"], m["salary"])
            if v is None or (m["salary"] or 0) < MIN_SALARY_FOR_RATE or v <= med:
                continue
            price_items.append((v - med, m, v))
        price_items.sort(key=lambda t: -t[0])
    prices = []
    for _, m, v in price_items[:2]:
        rank = ranks.get(m["pid"])
        f = {"player": m["name"], "salary": int(m["salary"]), "proj": round(m["proj"], 1),
             "per_k": round(v, 1), "slate_median_per_k": round(med, 1)}
        note = f"slate median {med:.1f}"
        if rank is not None and rank <= TOP_RANK:
            note = f"{_ordinal(rank)} of {n_ranked} on the slate; " + note
            f["rank"], f["of"] = rank, n_ranked
        text = (f"{_surname(m['name'])}: {money(m['salary'])} for {m['proj']:.1f} pts, "
                f"{v:.1f} per $1k ({note})")
        prices.append({"kind": "price", "text": text, "facts": f})

    # ---- game exposure
    games: dict[str, int] = {}
    for m in members:
        g = m["pool"].get("game")
        if g:
            games[g] = games.get(g, 0) + 1
    game_list = sorted(({"label": g, "n": n} for g, n in games.items()),
                       key=lambda d: (-d["n"], d["label"]))
    game = None
    if game_list and game_list[0]["n"] >= 3:
        g = game_list[0]
        game = {"kind": "game", "text": f"{g['n']} of 8 players come from {g['label']}",
                "facts": {"game": g["label"], "n": g["n"], "of": len(members)}}
        if len(members) != 8:
            game["text"] = f"{g['n']} of {len(members)} players come from {g['label']}"

    # one of each kind first (minutes, price, game), then a second minutes / price reason
    mins = [r for _, r in minutes]
    reasons = mins[:1] + prices[:1] + ([game] if game else []) + mins[1:2] + prices[1:2]
    reasons = reasons[:max_reasons]

    # ---- flags: P(play) < 0.9, or listed questionable with no P(play) yet
    flags = []
    for m in members:
        p = m["pool"]
        pp, st = _num(p.get("p_play")), p.get("status")
        name = _surname(m["name"])
        if pp is not None and pp < P_PLAY_FLAG:
            pct = round(pp * 100)
            tag = f" (listed {st})" if st else ""
            flags.append({"name": m["name"], "p_play": round(pp, 3), "status": st,
                          "text": f"{name}: {pct}% to play{tag}"})
        elif pp is None and st in ("Q", "GTD", "D", "OUT"):
            flags.append({"name": m["name"], "p_play": None, "status": st,
                          "text": f"{name} is listed {st}; no P(play) for this run"})

    used = _num(lineup.get("salary"))
    if used is None:
        used = sum(m["salary"] or 0 for m in members)
    mean = _num(lineup.get("mean"))
    summary = {"salary_used": int(used), "cap": int(cap), "salary_left": int(cap - used),
               "mean": None if mean is None else round(mean, 1), "games": game_list,
               "text": f"Uses {money(used)} of {money(cap)}; {money(cap - used)} left"}
    return {"version": VERSION, "reasons": reasons, "flags": flags, "summary": summary}


def render_lines(why: Mapping | None) -> dict:
    """{'reasons': [str], 'flags': [str], 'summary': str} for templates (empty when no why)."""
    if not why:
        return {"reasons": [], "flags": [], "summary": ""}
    return {"reasons": [r["text"] for r in why.get("reasons") or []],
            "flags": [f["text"] for f in why.get("flags") or []],
            "summary": (why.get("summary") or {}).get("text", "")}


def dumps(why: Mapping) -> str:
    """The string stored in ``dfs_lineups.why_json``."""
    return json.dumps(why, sort_keys=True)


def loads(s: str | None) -> dict | None:
    if not s:
        return None
    try:
        v = json.loads(s)
    except ValueError:
        return None
    return v if isinstance(v, dict) and "reasons" in v else None
