"""Minutes-restriction flags from ESPN injury comments (IMPROVEMENTS.md P7, `docs/research/min-restrict.md`).

ESPN's injuries feed (`sources/espn.py`, archived as `raw/espn/injuries/<day>/injuries__<ts>.json.gz`)
carries, per listed player, a RotoWire-style `shortComment` (the news) and `longComment` (the
analysis) plus the news `date`. This module reads them; it fetches nothing.

`parse_comment(short, long, name)` -> `Restriction(flag, cap, phrase)`:

* **Precision first.** A sentence counts only when it names a minutes restriction outright
  ("minutes restriction / limit / cap", "restricted / limited minutes", "on a minutes limit",
  "will be eased in", "keep a close eye on his minutes", "face restrictions", ...).
  Vetoes, per sentence: a negation or a lifted restriction ("no", "not", "without", "lifted",
  "no longer", ...), a past season ("last season", "2025-26"), contract talk ("restricted free
  agent") and a teammate's sentence (another player tagged "Name (injury)", the feed's convention,
  or a sentence that opens on "With <someone else>").
* **Cap.** A number of minutes tied to the restriction wording ("a minutes cap of 16", "capped at
  20 minutes", "a 20-minute limit", "around 15-20 minutes" in a flagged sentence). A range gives
  its midpoint. Kept in [CAP_MIN, CAP_MAX], else None.

`snapshot_rows(payload)` applies it to every player of a snapshot. `flags_at(conn, at)` is the
derived table, built on the fly: the latest ESPN snapshot captured at or before `at` (its
`injury_snapshots.raw_path`, re-rooted under this machine's data dir), parsed and cached in
memory. No migration and no stored column.
"""

from __future__ import annotations

import gzip
import json
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .. import config, teams

CAP_MIN, CAP_MAX = 6.0, 36.0

# --- positive patterns: each names a restriction on the player's minutes ------------------------
_MIN = r"(?:minutes?|playing[- ]time)"
_RES = r"(?:restrictions?|restricted|limits?|limited|caps?|capped|ceiling)"
_POS = [
    rf"\b{_MIN}[- ](?:restriction|limit|cap|ceiling)s?\b",                    # minutes restriction
    rf"\b\d{{1,2}}[- ]minute (?:restriction|limit|cap)\b",                     # 20-minute limit
    rf"\b(?:restricted|limited|reduced|capped|managed) {_MIN}\b",              # limited minutes
    rf"\b(?:restriction|limit|cap)s? on (?:his |the )?{_MIN}\b",               # a limit on his minutes
    rf"\b{_MIN} (?:will|would|could|should|may|might|is|are|being|be|to be|likely to be)"
    rf"(?: be| remain| likely be| probably be)? (?:restricted|limited|capped|monitored|managed)\b",
    rf"\b(?:eased|ease him|ease [a-z]+|easing him|work(?:ed)? him) (?:back )?in(?:to)?\b",  # eased in
    rf"\b(?:keep|keeping) (?:a )?(?:close )?(?:eye|tabs) on (?:his|the) {_MIN}\b",
    rf"\b(?:monitor|monitoring|monitored|manage|managing|watch|watching) (?:his|the) {_MIN}\b",
    r"\bface (?:some |a |any )?(?:minutes |playing[- ]time )?restrictions?\b",
    rf"\b(?:on|under) (?:a |some )?(?:strict |tight |firm |slight )?{_MIN} (?:restriction|limit|cap)s?\b",
    rf"\bpitch count\b",
    # a number of minutes stated as the plan or the ceiling
    r"~\s*\d{1,2}(?:\s*(?:-|–|to)\s*\d{1,2})?\s*(?:minutes?|mins?)\b",
    r"\b(?:will|'ll|expected to|likely to|set to|slated to|projected to|figures to|should)\s+"
    r"(?:only\s+|just\s+)?(?:play|log|see|get)\s+(?:only |just )?(?:about |around |roughly |"
    r"approximately |no more than |up to )\d{1,2}(?:\s*(?:-|–|to)\s*\d{1,2})?\s+minutes\b",
    r"\b(?:capped|limited|restricted|held)\s+(?:at|to|under)\s+(?:about |around |roughly |~\s*)?"
    r"\d{1,2}(?:\s*(?:-|–|to)\s*\d{1,2})?\s*minutes\b",
]
_POS_RE = [re.compile(p, re.I) for p in _POS]

# --- vetoes ------------------------------------------------------------------------------------
_NEG_RE = re.compile(
    r"\b(?:no|not|without|never|lifted|removed|free of|no longer|isn't|wasn't|won't|"
    r"doesn't|didn't|shouldn't|aren't|nor)\b|n't\b", re.I)
_PAST_RE = re.compile(r"\b(?:last (?:season|year|campaign)|20\d\d-\d\d|previous season|"
                      r"a year ago)\b", re.I)
_CONTRACT_RE = re.compile(r"\bfree agen(?:t|cy)\b|\bcontract\b|\bextension\b", re.I)
# "Grayson Allen (knee)": the feed tags other injured players this way
_TAGGED_RE = re.compile(r"\b([A-Z][A-Za-z'\.\-]+(?: (?:Jr\.|Sr\.|II|III|IV))?) \(([a-z][a-z ,/\-]*)\)")
_WITH_RE = re.compile(r"^\s*(?:With|Without|If)\s+([A-Z][A-Za-z'\.\-]+)(?:\s+([A-Z][A-Za-z'\.\-]+))?")

# --- cap -------------------------------------------------------------------------------------
_NUM = r"(\d{1,2}(?:\.\d)?)"
_RANGE = rf"{_NUM}(?:\s*(?:-|–|to)\s*{_NUM})?"
_APPROX = r"(?:about |around |roughly |approximately |near |~\s*|close to |in the )?"
_CAP_RE = [
    re.compile(rf"\b(?:restriction|limit|cap|ceiling)s?\s+(?:of|at|around|near|set at)\s+"
               rf"{_APPROX}{_RANGE}\s*(?:minutes?|mins?)?", re.I),
    re.compile(rf"\b(?:capped|limited|restricted|held|kept)\s+(?:at|to|under)\s+{_APPROX}{_RANGE}"
               rf"\s*(?:minutes?|mins?)\b", re.I),
    re.compile(rf"\b{_NUM}[- ]minute (?:restriction|limit|cap)\b", re.I),
    re.compile(rf"(?:~|about |around |roughly |approximately |near |close to ){_RANGE}\s*"
               rf"(?:minutes?|mins?)\b", re.I),
    re.compile(rf"\b(?:no more than|not more than|at most|a maximum of|max of|up to)\s+{_RANGE}"
               rf"\s*(?:minutes?|mins?)\b", re.I),
]


@dataclass(frozen=True)
class Restriction:
    flag: bool
    cap: float | None = None          # parsed minutes cap (None: flagged without a number)
    phrase: str = ""                  # the sentence that set the flag ('' when not flagged)


NONE = Restriction(False)


def sentences(text: str) -> list[str]:
    """Split a comment into sentences (keeps 'Jr.' / initials / decimals together)."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])[\"”']?\s+(?=[\"“A-Z0-9])", text)
    out: list[str] = []
    for s in parts:
        # re-glue splits after an abbreviation or an initial ("Jr.", "J.", "St.", "No.")
        if out and re.search(r"(?:\b(?:Jr|Sr|St|No|vs|Dr|Mr)\.|\b[A-Z]\.)$", out[-1]):
            out[-1] = f"{out[-1]} {s}"
        else:
            out.append(s)
    return out


def _last_name(name: str | None) -> str:
    toks = [t for t in re.split(r"\s+", (name or "").strip())
            if t and t.rstrip(".").lower() not in ("jr", "sr", "ii", "iii", "iv")]
    return toks[-1].lower() if toks else ""


def _first_name(name: str | None) -> str:
    toks = (name or "").strip().split()
    return toks[0].lower() if toks else ""


def _about_other(s: str, name: str | None) -> bool:
    """Is the sentence about another player (tagged 'Name (injury)' or opening 'With X ...')?"""
    if not name:
        return False
    last, first = _last_name(name), _first_name(name)
    for m in _TAGGED_RE.finditer(s):
        tok = m.group(1).split()[0].lower().rstrip(".")
        if tok not in (last, first) and _last_name(m.group(1)) != last:
            return True
    m = _WITH_RE.match(s)
    if m:
        toks = {t.lower() for t in m.groups() if t}
        pron = {"the", "him", "his", "he", "that", "this", "a", "an"}
        if toks - pron and not toks & {last, first}:
            return True
    return False


def _cap_of(s: str) -> float | None:
    for rx in _CAP_RE:
        m = rx.search(s)
        if not m:
            continue
        nums = [float(g) for g in m.groups() if g is not None]
        if not nums:
            continue
        v = sum(nums[:2]) / len(nums[:2])
        if CAP_MIN <= v <= CAP_MAX:
            return round(v, 1)
    return None


def sentence_flag(s: str, name: str | None = None) -> bool:
    """Does this one sentence put the subject on a minutes restriction?"""
    if not any(rx.search(s) for rx in _POS_RE):
        return False
    if _NEG_RE.search(s) or _PAST_RE.search(s) or _CONTRACT_RE.search(s):
        return False
    return not _about_other(s, name)


def parse_comment(short: str | None, long: str | None = None, name: str | None = None
                  ) -> Restriction:
    """Restriction flag + optional cap from a player's ESPN short and long comments."""
    for text in (short, long):
        for s in sentences(text or ""):
            if sentence_flag(s, name):
                return Restriction(True, _cap_of(s), s)
    return NONE


# ============================================================================== snapshots

def snapshot_rows(payload: dict) -> list[dict]:
    """Every listed player of an ESPN injuries payload: team (NBA tricode, None if unmapped),
    player_name, espn_id, status, news `date`, and the parsed restriction."""
    out: list[dict] = []
    for tb in payload.get("injuries") or []:
        for inj in tb.get("injuries") or []:
            ath = inj.get("athlete") or {}
            name = ath.get("displayName") or " ".join(
                filter(None, [ath.get("firstName"), ath.get("lastName")]))
            abbr = (ath.get("team") or {}).get("abbreviation") or tb.get("abbreviation")
            try:
                tri = teams.espn_to_tricode(abbr) if abbr else None
            except Exception:  # noqa: BLE001 - unknown abbreviations are skipped by callers
                tri = None
            if tri is None and tb.get("displayName"):
                tri = next((t.tricode for t in teams.TEAMS
                            if f"{t.city} {t.name}" == tb["displayName"]), None)
            r = parse_comment(inj.get("shortComment"), inj.get("longComment"), name)
            aid = None
            for link in ath.get("links") or []:
                m = re.search(r"/id/(\d+)", link.get("href") or "")
                if m:
                    aid = m.group(1)
                    break
            out.append({"team": tri, "player_name": name, "espn_id": aid,
                        "status": inj.get("status") or "", "date": inj.get("date") or "",
                        "flag": r.flag, "cap": r.cap, "phrase": r.phrase})
    return out


def local_raw_path(raw_path: str) -> Path:
    """An `injury_snapshots.raw_path` (absolute on the machine that wrote it) under this
    machine's data dir: everything after the last '/raw/'."""
    p = str(raw_path)
    i = p.rfind("/raw/")
    return config.data_dir() / "raw" / p[i + 5:] if i >= 0 else Path(p)


@lru_cache(maxsize=256)
def _rows_of_file(path: str) -> tuple:
    with gzip.open(path, "rt") as fh:
        return tuple(snapshot_rows(json.load(fh)))


def snapshots(conn: sqlite3.Connection, at_iso: str, since_iso: str | None = None
              ) -> list[tuple[str, str]]:
    """[(captured_at, raw_path)] of the ESPN injuries snapshots captured in [since_iso, at_iso]
    (oldest first); only the newest one at or before `at_iso` when `since_iso` is None."""
    if since_iso is None:
        r = conn.execute("SELECT captured_at, raw_path FROM injury_snapshots WHERE source='espn' "
                         "AND captured_at<=? AND raw_path IS NOT NULL ORDER BY captured_at DESC "
                         "LIMIT 1", (at_iso,)).fetchone()
        return [(r[0], r[1])] if r else []
    return [(c, p) for c, p in conn.execute(
        "SELECT captured_at, MIN(raw_path) FROM injury_snapshots WHERE source='espn' AND "
        "captured_at<=? AND captured_at>=? AND raw_path IS NOT NULL GROUP BY captured_at "
        "ORDER BY captured_at", (at_iso, since_iso))]


def flags_at(conn: sqlite3.Connection, at_iso: str, since_iso: str | None = None
             ) -> tuple[str | None, list[dict]]:
    """(newest snapshot's captured_at, flagged rows) as known at `at_iso`.

    With `since_iso`, a flag is sticky over the window: per (team, player) the newest flagged
    comment of any snapshot captured since then (ESPN replaces a comment with the next news,
    which often drops the restriction remark), unless the newest snapshot lists him Out. Rows
    carry `seen_at` (the capture that had the comment). Snapshots whose raw file is missing on
    this machine are skipped; (None, []) when there is none."""
    snaps = [(c, local_raw_path(p)) for c, p in snapshots(conn, at_iso, since_iso)]
    snaps = [(c, p) for c, p in snaps if p.exists()]
    if not snaps:
        return None, []
    latest = _rows_of_file(str(snaps[-1][1]))
    out_now = {(r["team"], r["player_name"]) for r in latest
               if (r.get("status") or "").strip().lower() == "out"}
    flagged: dict[tuple, dict] = {}
    for cap, path in snaps:
        for r in _rows_of_file(str(path)):
            if r["flag"]:
                flagged[(r["team"], r["player_name"])] = {**r, "seen_at": cap}
    return snaps[-1][0], [r for k, r in flagged.items() if k not in out_now]
