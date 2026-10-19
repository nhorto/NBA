"""`docs/MODEL_CARD.md`, generated from the registry (`courtside mlops card`)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from . import registry

ROLE = {"gbm_market": "headline win probability, margin and total shown in the Brief",
        "gbm_opinion": "\"our number\": independent of the market, used for the edge vs the line"}

LIMITATIONS = [
    "**Injury data only from 2026.** Official/ESPN injury reports are archived from the "
    "2026-27 season on. The 2012-26 benchmark never saw a real injury report: availability "
    "there is a leak-safe proxy (below), so live accuracy on days with real reports is not yet "
    "measured.",
    "**Availability proxy.** Backtest player strength uses the engine's `avail_tight` mode: the "
    "rotation regulars who actually played (projection >= players.REGULAR_MIN) plus fringe "
    "players on the recent roster. It ignores real absences of players under ~20 minutes and "
    "still knows which regulars played, so true production performance sits between this and "
    "the (leaky) full oracle. Live days use the injury report or the full recent roster.",
    "**No edge against the spread.** Every ATS and over/under record in the benchmark is inside "
    "noise and below the 52.4% break-even at -110. `gbm_market` is about as good as the closing "
    "line (it uses the line); `gbm_opinion` is ~0.01 log-loss worse than the line.",
    "**Early-season priors.** In the first weeks of a season rolling form, pace and rating "
    "windows are padded with a regressed prior-season prior and player ratings carry over the "
    "summer; new rosters, rookies and trades are only learned as games are played.",
    "**Playoffs.** Fewer games, tighter rotations; the market is clearly better there "
    "(see vs-market rows).",
    "**Market input timing.** `gbm_market` is trained on closing lines but priced on the latest "
    "pre-tip line available in the morning, a slightly noisier input.",
]


def _f(x, nd=4, sign=False):
    if x is None:
        return "-"
    try:
        return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _summary(bench: dict | None) -> dict:
    if not bench:
        return {}
    return bench.get("summary", bench)


def _metrics_rows(s: dict) -> list[str]:
    out = ["| scope | n | log-loss | Brier | accuracy | ECE |", "|---|---|---|---|---|---|"]
    for key, label in (("all", "all test games"), ("regular_season", "regular season"),
                       ("playoffs", "playoffs + play-in"), ("last_season", "last season")):
        m = s.get(key)
        if not m or not m.get("n"):
            continue
        lab = f"{label} ({m['season']})" if key == "last_season" else label
        out.append(f"| {lab} | {m['n']:,} | {_f(m.get('log_loss'))} | {_f(m.get('brier'))} | "
                   f"{_f(m.get('accuracy'), 3)} | {_f(m.get('ece'), 3)} |")
    return out


def _market_rows(s: dict) -> list[str]:
    vm = s.get("vs_market") or {}
    if not vm:
        return []
    out = ["| games with a closing moneyline | n | model LL | market LL "
           "| model - market (paired SE) |",
           "|---|---|---|---|---|"]
    for key, label in (("regular_season", "regular season"),
                       ("regular_season_last3", "regular season, last 3 seasons"),
                       ("playoffs", "playoffs")):
        v = vm.get(key)
        if not v:
            continue
        d = v["ll_minus_market"]
        out.append(f"| {label} | {v['n']:,} | {_f(v['log_loss'])} | {_f(v['market_log_loss'])} | "
                   f"{_f(d['diff'], sign=True)} ± {_f(d['se'])} |")
    return out


def _calibration(s: dict) -> list[str]:
    cal = [c for c in s.get("calibration") or [] if c.get("n")]
    if not cal:
        return []
    return ["| P(home) bin | n | mean predicted | observed |", "|---|---|---|---|"] + [
        f"| {c['lo']:.1f}-{c['hi']:.1f} | {c['n']:,} | {_f(c['mean_p'], 3)} | "
        f"{_f(c['frac_pos'], 3)} |" for c in cal]


def _params_rows(p: dict) -> list[str]:
    out = ["| head | leaves | min leaf | lr | L2 | feature frac | bagging | half-life | trees |",
           "|---|---|---|---|---|---|---|---|---|"]
    for t, (cfg, n) in sorted((p.get("fixed") or {}).items()):
        hl = cfg.get("half_life")
        out.append(f"| {t} | {cfg['num_leaves']} | {cfg['min_data_in_leaf']} | "
                   f"{cfg['learning_rate']} | {cfg['lambda_l2']} | {cfg['feature_fraction']} | "
                   f"{cfg['bagging_fraction']} | {'none' if hl is None else f'{hl:g} seasons'} | "
                   f"{n} |")
    return out


def _data_span(conn) -> str:
    try:
        r = conn.execute("SELECT MIN(game_date), MAX(game_date), COUNT(DISTINCT game_id), "
                         "MIN(season), MAX(season) FROM team_games").fetchone()
    except sqlite3.OperationalError:
        return "unknown (no team_games table)"
    if not r or r[0] is None:
        return "no games loaded"
    return (f"{r[3]} → {r[4]}: {r[2]:,} games ({r[0]} to {r[1]}) in `team_games`; the first 3 "
            f"seasons are burn-in for the player engine and the GBM heads")


def _live(conn, names) -> list[str]:
    try:
        rows = conn.execute(
            "SELECT * FROM metric_snapshots WHERE as_of = (SELECT MAX(as_of) FROM "
            "metric_snapshots) AND window IN ('season', 'last300') ORDER BY model, window"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    cols = None
    out = []
    for r in rows:
        d = dict(r) if hasattr(r, "keys") else None
        if d is None:
            cols = cols or [c[1] for c in conn.execute("PRAGMA table_info(metric_snapshots)")]
            d = dict(zip(cols, r))
        if d["model"] not in names or not d.get("n"):
            continue
        out.append(f"| {d['model']} | {d['window']} | {d['n']} | {_f(d.get('log_loss'))} | "
                   f"{_f(d.get('ll_diff_vs_mkt'), sign=True)} ± {_f(d.get('ll_diff_se'))} | "
                   f"{_f(d.get('ece'), 3)} |")
    if not out:
        return []
    return ["| model | window | n | log-loss | vs market (SE) | ECE |",
            "|---|---|---|---|---|---|"] + out


def render(conn) -> str:
    names = registry.production_names()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    champs = {n: registry.get_champion(conn, n) for n in names}
    L = ["# Courtside model card", "",
         f"_Generated {now} by `courtside mlops card` from the model registry "
         "(`model_versions`). Do not edit by hand; see docs/MODEL.md for the methodology and "
         "docs/MLOPS.md for how versions change._", "",
         "## Intended use", "",
         "Pre-game win probabilities, expected margins and totals for NBA games, shown in the "
         "Courtside Morning Brief as **information, not betting advice**. The model has no "
         "demonstrated edge against the betting market (see limitations). It is not meant for "
         "wagering, player evaluation or any decision with money at stake.", "",
         "## Champion versions", "",
         "| model | role in the product | version | champion since | code | spec hash |",
         "|---|---|---|---|---|---|"]
    for n in names:
        c = champs[n]
        if c is None:
            L.append(f"| {n} | {ROLE.get(n, '')} | _unregistered (gbm.FROZEN)_ | - | - | - |")
            continue
        L.append(f"| {n} | {ROLE.get(n, '')} | `{c['version_id']}` | "
                 f"{(c['promoted_at'] or '')[:10]} | `{c['code_sha']}` | "
                 f"`{c['spec_hash'][:12]}` |")
    L += ["", "## Training data", "", f"- {_data_span(conn)}.",
          "- Production refits every model every morning on **all games strictly before the "
          "day** (continuous training; a refit is not a new version). Hyperparameters are "
          "fixed per version; recency weighting comes from the version's half-life.",
          "- Inputs: player-strength engine (box scores), Elo, schedule/travel, rolling team "
          "form and four factors, league environment, home-court drift terms; `gbm_market` "
          "also the closing (backtest) or latest pre-tip (live) spread and total; injury "
          "reports priced into expected lineups when available.", ""]
    for n in names:
        c = champs[n]
        if c is None:
            continue
        p, s = c["params"] or {}, _summary(c["benchmark"])
        L += [f"## {n} (`{c['version_id']}`)", "",
              f"Market input: {'yes' if p.get('market') else 'no'}. Search grid the fixed "
              f"values come from: {len(p.get('search_grid') or [])} configs "
              "(inner time-ordered validation over the last 3 training seasons).", ""]
        L += _params_rows(p) + [""]
        if s:
            src = (c["benchmark"] or {}).get("source") or (c["benchmark"] or {}).get("procedure")
            L += [f"**Benchmark** (walk-forward, test seasons {' → '.join(s.get('seasons', []))}"
                  f"; {src}):", ""]
            L += _metrics_rows(s) + [""]
            mr = _market_rows(s)
            if mr:
                L += ["**Versus the closing line** (same games):", ""] + mr + [""]
            cal = _calibration(s)
            if cal:
                L += ["**Calibration** (all test games, 10 bins):", ""] + cal + [""]
    live = _live(conn, names)
    if live:
        L += ["## Live performance (latest monitoring snapshot)", ""] + live + [""]
    others = [v for v in registry.list_versions(conn) if v["status"] in ("shadow", "candidate")]
    L += ["## Versions in evaluation", ""]
    if others:
        L += ["| version | status | created | notes |", "|---|---|---|---|"]
        L += [f"| `{v['version_id']}` | {v['status']} | {v['created_at'][:10]} | "
              f"{v['notes'] or ''} |" for v in others]
    else:
        L.append("None.")
    ev = registry.events(conn, limit=12)
    if ev:
        L += ["", "## Recent registry events", "", "| time | version | action | reason |",
              "|---|---|---|---|"]
        L += [f"| {e['ts'][:16].replace('T', ' ')} | `{e['version_id']}` | {e['action']} | "
              f"{(e['reason'] or '').replace('|', '/')[:140]} |" for e in ev]
    L += ["", "## Known limitations", ""] + [f"- {x}" for x in LIMITATIONS]
    L += ["", "## Promotion policy", "",
          "A new version must beat the champion on the frozen walk-forward benchmark by more "
          "than 2 paired standard errors of per-game log-loss, not be worse on the most recent "
          "season beyond noise, and not worsen ECE by more than 0.005; then predict live in "
          "shadow (never shown) for at least 100 games with paired log-loss ≤ the champion's "
          "and no open critical alert. Rollback restores the previous champion in one command "
          "(`courtside mlops rollback <model> --reason ...`).", ""]
    return "\n".join(L)


def write(conn, path: Path | None = None) -> Path:
    path = Path(path or config.REPO_ROOT / "docs" / "MODEL_CARD.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(conn))
    return path

