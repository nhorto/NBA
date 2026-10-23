"""Reserve extras (IMPROVEMENTS.md B2b, switch COURTSIDE_RESERVE_EXTRAS, tag `+rx`).

Docs: docs/research/reserve-extras.md.  Code under test: `minutes.RESERVE_EXTRAS`,
`minutes.rx_rows`, the reserve block of `minutes.project_arrays`.

    python research/reserve_extras.py decomp           # minutes outside the last-10 candidates:
                                                       # career debutants vs players with history
    python research/reserve_extras.py fit              # src/courtside/proj/reserve_extras_params.json
    python research/reserve_extras.py harness S PROTO  # recon_reserve.py's harness (rows + teams
                                                       # parquet) + the flagged extras per team-game
    python research/reserve_extras.py dk OFF ON LABEL  # recon_reserve.py's paired DK table + team
                                                       # sums + the flagged extras' plan vs actual

OFF identity: `tests/test_reserve_extras.py` + research/p2_identity.py and the rr `id_driver`
(HEAD tree vs this tree, switch unset) - see the write-up.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import recon_reserve as RR  # noqa: E402

OUT = RR.OUT


def team_games(since: str = "2012-13") -> pd.DataFrame:
    """Regular-season team-games: `res` = box minutes of players outside the team's last-10-game
    candidates (recon_reserve.reserve_rows), `deb` = those of career debutants (no earlier box
    row: nobody can project them), `hist` = the rest (players with box history, whom the live
    path forces in through the DK pool), `tgn` = team games into the season (date order)."""
    g = RR.reserve_rows(RR.ro_conn())
    g = g[(g["season_type"] == "Regular Season") & (g["season"] >= since)]
    tg = g.groupby(["game_id", "team"]).agg(season=("season", "first"),
                                            date=("game_date", "first")).reset_index()
    r = g[g["res"]].assign(dm=lambda x: np.where(x["deb"], x["min"], 0.0))
    a = r.groupby(["game_id", "team"]).agg(res=("min", "sum"), deb=("dm", "sum")).reset_index()
    t = tg.merge(a, on=["game_id", "team"], how="left").fillna({"res": 0.0, "deb": 0.0})
    t["hist"] = t["res"] - t["deb"]
    t = t.sort_values(["date", "game_id"], kind="mergesort")
    t["tgn"] = t.groupby(["team", "season"]).cumcount()
    return t.reset_index(drop=True)


def cmd_decomp() -> None:
    t = team_games("2014-15")
    cols = ["res", "deb", "hist"]
    print("all team-games (min per team-game)")
    print(t.groupby("season")[cols].mean().round(2).to_string())
    print("openers (team game < 5)")
    print(t[t["tgn"] < 5].groupby("season")[cols].mean().round(2).to_string())
    q = t["res"].quantile(0.9)
    print(f"res >= p90 ({q:.1f}): " + "  ".join(f"{c} {t.loc[t['res'] >= q, c].mean():.2f}"
                                               for c in cols))


def cmd_fit() -> None:
    """minutes.RX_PATH: per target season S, the mean career-debut minutes per team-game by
    minutes.DEBUT_BINS of team games into the season, pooled over the POOL seasons < S."""
    from courtside.proj import minutes as M
    t = team_games()
    t["bin"] = np.clip(np.searchsorted(M.DEBUT_BINS, t["tgn"], side="right") - 1, 0,
                       len(M.DEBUT_BINS) - 1)
    out = {}
    for s in RR.TARGETS:
        tr = sorted(x for x in t["season"].unique() if x < s)[-RR.POOL:]
        x = t[t["season"].isin(tr)]
        m = x.groupby("bin")["deb"].agg(["mean", "size"])
        out[s] = {"train_seasons": tr, "bins": [round(float(v), 4) for v in m["mean"]],
                  "n": [int(v) for v in m["size"]]}
        print(s, tr, out[s]["bins"], out[s]["n"])
    M.RX_PATH.write_text(json.dumps({"doc": "career-debut minutes per team-game by "
                                     "minutes.DEBUT_BINS (team games into the season), "
                                     "regular season, pooled over seasons < S "
                                     "(research/reserve_extras.py fit)",
                                     "bins": list(M.DEBUT_BINS), "seasons": out}, indent=1))
    print(f"-> {M.RX_PATH}")


def _patch_flags() -> list:
    """Record every team-game's flagged extras (`minutes.rx_rows` of its candidate panel)."""
    from courtside.proj import copula as C
    from courtside.proj import minutes as M
    rows: list = []
    init = C.ScenarioMinutes.__init__

    def init2(self, panel, *a, **k):
        init(self, panel, *a, **k)
        p = self.panel
        if len(p):
            f = M.rx_rows(p)
            pool = M.rx_pool(p)
            rows.extend({"date": int(p["date"].iloc[0]), "key": str(p["key"].iloc[0]),
                         "team": str(p["team"].iloc[0]), "player_id": int(pid), "pool": pool}
                        for pid in p["player_id"].to_numpy()[f])
            rows.append({"date": int(p["date"].iloc[0]), "key": str(p["key"].iloc[0]),
                         "team": str(p["team"].iloc[0]), "player_id": -1,
                         "pool": pool})                                     # team marker
    C.ScenarioMinutes.__init__ = init2
    return rows


def cmd_harness(season: str, protocol: str) -> None:
    from courtside.proj import pipeline as P
    flags = _patch_flags()
    RR.cmd_harness(season, protocol)
    f = OUT / f"rxf_{protocol}_{season}_{P.MODEL_VERSION}.parquet"
    pd.DataFrame(flags, columns=["date", "key", "team", "player_id", "pool"]).to_parquet(f)
    print(f"[rx] {sum(r['player_id'] >= 0 for r in flags)} flagged extras over "
          f"{sum(r['player_id'] < 0 for r in flags)} team-games, "
          f"{sum(r['player_id'] < 0 and r['pool'] for r in flags)} with the pool -> {f}",
          flush=True)


def _flag_table(rows_path: str) -> pd.DataFrame | None:
    p = Path(rows_path.replace("rows_", "rxf_"))
    return pd.read_parquet(p) if p.exists() else None


def cmd_dk(off: str, on: str, label: str) -> None:
    """recon_reserve's paired table and team sums, then the flagged extras: their planned
    E[min] (exp_min, both arms) vs actual minutes, per team-game, and the effective reserve
    (E[res], 240 + 25 n_OT - the projected players' sim minutes) vs the actual minutes of
    players who were not projected."""
    RR.cmd_dk(off, on, label)
    fl = _flag_table(on)
    if fl is None:
        fl = _flag_table(off)
    a, b = pd.read_parquet(off), pd.read_parquet(on)
    out: dict = {}
    if fl is not None:
        fx = fl[fl["player_id"] >= 0].copy()
        fx["game_id"] = fx["key"].str.split("|").str[0]
        k = ["game_id", "player_id"]
        rows = a.dropna(subset=["player_id"]).astype({"player_id": np.int64})
        rows_b = b.dropna(subset=["player_id"]).astype({"player_id": np.int64})
        m = fx.merge(rows[k + ["exp_min", "minutes"]], on=k, how="left").merge(
            rows_b[k + ["exp_min"]].rename(columns={"exp_min": "exp_on"}), on=k, how="left")
        m["minutes"] = m["minutes"].fillna(0.0)
        tg = m.groupby(["game_id", "team"]).agg(n=("player_id", "size"),
                                                plan_off=("exp_min", "sum"),
                                                plan_on=("exp_on", "sum"),
                                                act=("minutes", "sum")).reset_index()
        n_tg = int((fl["player_id"] < 0).sum())
        out = {"team_games": n_tg, "flagged_rows": int(len(fx)),
               "teams_with_flags": int(len(tg)),
               "per_team_game": {c: float(tg[c].sum() / max(n_tg, 1))
                                 for c in ("n", "plan_off", "plan_on", "act")}}
        print(f"\n## flagged extras {label}: {len(fx)} rows on {len(tg)} of {n_tg} team-games; "
              "per team-game: " + "  ".join(f"{c} {v:.2f}"
                                            for c, v in out["per_team_game"].items()))
    for nm, path in (("off", off), ("on", on)):
        t = RR._load_teams(path)
        if t is not None and len(t):
            out[f"res_e_{nm}"] = float(t["res_e"].mean())
    print("  E[res] " + "  ".join(f"{k} {v:.2f}" for k, v in out.items() if k.startswith("res_e")))
    (OUT / f"rxflags_{label}.json").write_text(json.dumps(out, indent=1, default=float))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("decomp")
    sub.add_parser("fit")
    h = sub.add_parser("harness")
    h.add_argument("season")
    h.add_argument("protocol", choices=("avail", "live", "live10"))
    d = sub.add_parser("dk")
    d.add_argument("off")
    d.add_argument("on")
    d.add_argument("label")
    x = ap.parse_args(argv)
    if x.cmd == "decomp":
        cmd_decomp()
    elif x.cmd == "fit":
        cmd_fit()
    elif x.cmd == "harness":
        cmd_harness(x.season, x.protocol)
    else:
        cmd_dk(x.off, x.on, x.label)


if __name__ == "__main__":
    main()
