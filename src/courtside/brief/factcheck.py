"""Fact-check generated prose against the facts JSON it was written from.

    idx = FactIndex.build(facts)            # numbers, score pairs, clocks, name vocabulary
    res = check(text, idx, kind="game_story")
    res.ok, res.violations, res.warnings

What is checked (violations make the writer retry once, then fall back to templates):

* **numbers**   every numeral in the text must be traceable to a number in the facts:
  exact, rounded to the displayed precision (``64.9%`` = ``0.649`` = ``65%``; ``44`` = ``43.5``),
  or a simple derivation (a game's final-score sum/difference, a quarter-range sum, the gap
  between two scores in a snapshot). Spelled numbers 10-99 ("twenty-four") are read too;
  below ten they're left alone ("two-time").
* **pairs**     ``125-124`` / ``14-2`` / ``35-20`` / ``10-of-19`` / ``10 for 19`` must match a pair
  in the facts as written ("13-4", "10-19 FG", a record) or the two sides of a score (either
  order). Three-part lines (``33-8-5``) are checked number by number.
* **times**     ``3:52`` must be a clock in the facts.
* **names**     every capitalised word that isn't sentence-initial must appear in the facts
  (player/team tokens, plus team names for any tricode in the facts). Accents are ignored
  (Jokic = Jokic). A sentence-initial single word can't be told from a capitalised common word,
  so only sentence-initial runs of 2+ unknown capitalised words are flagged. Tricodes in prose
  are accepted but warned about.
* **betting**   banned tipster language (see docs/STYLE.md).
* **length**    only gross overruns (more than ~30% over the target) are violations.
* **units**     when the facts carry ``unit_values`` (``{"dk": [...], "minutes": [...],
  "points": [...], ...}``, e.g. the Brief's player stories), a number written with a unit
  ("52.3 DK points", "41 minutes", "6 threes") must be one of that unit's values, so a
  minutes figure can't be passed off as DK points.
* **player_story** (kind) DFS-tipster words are banned on top of the betting list ("edge",
  "lean", "value", "chalk", "cover", "lock", "smash", and the betting phrases "the over", "hit the
  over", "over/under", "o/u"; plain "over"/"under" are allowed), and when the facts carry ``direction``
  (``above``/``below`` our projection) the text may not say the opposite.

Warnings (recorded, never trigger a retry): clichés, exclamation marks, markdown, tricodes.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from courtside.teams import TEAMS

_TRICODES = {t.tricode for t in TEAMS}
_MARGIN_KEYS = ("lead", "deficit", "margin", "spread", "edge", "gap", "swing", "change")
_ID_KEYS = {"id", "player_id", "game_id", "team_id", "schema", "key"}
_EPS = 1e-6


# ---------------------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Violation:
    kind: str        # number | pair | time | name | betting | length | format
    token: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.kind} '{self.token}'" + (f": {self.detail}" if self.detail else "")


@dataclass
class CheckResult:
    violations: list[Violation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def messages(self) -> list[str]:
        return [str(v) for v in self.violations]


# kind -> (target words, hard max words, target sentences, hard max sentences)
LIMITS: dict[str, tuple[int | None, int | None, int | None, int | None]] = {
    "game_headline": (10, 13, 1, 2),
    "game_story": (None, 130, 4, 5),
    "pulse_headline": (10, 13, 1, 2),
    "blurb": (None, 75, 2, 3),
    "tonight": (None, 75, 2, 3),
    "brief_headline": (12, 15, 1, 2),
    "player_story": (None, 60, 2, 3),
}


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------

def norm(s: str) -> str:
    """Accent- and case-insensitive form (Jokic == jokic), curly quotes straightened."""
    s = unicodedata.normalize("NFKD", s.replace("’", "'").replace("‘", "'"))
    return "".join(c for c in s if not unicodedata.combining(c)).casefold()


_CAP_TOKEN = re.compile(r"[A-ZÀ-Þ][\w'’.\-]*")


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _decimals(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


_NUMBER_RE = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_CHAIN_SEP = r"(?:\s*[-–−]\s*(?:of|for)\s*[-–−]\s*|\s+(?:of|for)\s+|\s*[-–−]\s*)"
_CHAIN_RE = re.compile(rf"(?<![\w.]){_NUMBER_RE}(?:{_CHAIN_SEP}{_NUMBER_RE})+(?![\w]|\.\d)")
_CHAIN_NUM = re.compile(_NUMBER_RE)
_CLOCK_RE = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?:\.\d)?(?![\d:])")
_ORD_RE = re.compile(r"(?<![\w.])(\d+)(?:st|nd|rd|th)\b", re.I)
_SEASON_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*[-–]\s*(\d{2})(?!\d)")
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_MONTH_DAY_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?\b")
_MARGIN_CLAIM_RES = [
    re.compile(rf"\b(?:led|trailed|trailing|leading|down|up|ahead|behind|won|win|wins|lost|lose|"
               rf"loses|beat|beats|favou?rites?|favou?red|favou?ring)\s+(?:by|of)\s+({_NUMBER_RE})(?![\d:%])", re.I),
    re.compile(rf"(?<![\w.])({_NUMBER_RE})[- ]points?\s+(?:lead|deficit|hole|comeback|win|loss|margin|"
               rf"blowout|advantage|cushion|favou?rites?|underdogs?)\b", re.I),
]
_SHOT_RE = re.compile(r"(?<![\w.])[23]-?(?:point(?:er)?s?|pt|pters?|P|PM)\b", re.I)
_NUM_RE = re.compile(rf"(?<![A-Za-z\d.,])({_NUMBER_RE})(\s*%|\s+percent\b)?(?![A-Za-z]|\d)")

_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_ONES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9}
_SPELLED_RE = re.compile(
    r"\b(" + "|".join(list(_TEENS) + [rf"(?:{t})(?:[-\s](?:{'|'.join(_ONES)}))?" for t in _TENS]) + r")\b",
    re.I)


def _spelled_value(w: str) -> int:
    w = w.lower().replace("-", " ").split()
    if len(w) == 1:
        return _TEENS.get(w[0]) or _TENS[w[0]]
    return _TENS[w[0]] + _ONES[w[1]]


# ---------------------------------------------------------------------------------------
# fact index
# ---------------------------------------------------------------------------------------

def _is_scoreboard(d: dict) -> bool:
    vals = list(d.values())
    return (len(d) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in vals)
            and all(isinstance(k, str) and k in _TRICODES for k in d))


class FactIndex:
    """Everything the facts document licenses a writer to say."""

    def __init__(self) -> None:
        self.vals: set[float] = set()          # every number (+ x100 for fractions), abs value
        self.pairs: set[tuple[float, float]] = set()
        self.clocks: set[str] = set()
        self.seasons: set[str] = set()
        self.vocab: set[str] = set()           # normalised capitalised tokens from the facts
        self.tricodes: set[str] = set()
        self.margins: set[float] = set()       # leads, deficits, margins, spreads
        self.words: set[str] = set()           # every lowercase word in the facts' strings
        self.units: dict[str, set[float]] = {}  # unit -> allowed values (facts' `unit_values`)
        self.direction: str | None = None       # 'above' / 'below' our projection (stories)

    # -- building ----------------------------------------------------------------------
    @classmethod
    def build(cls, facts: Any) -> "FactIndex":
        if isinstance(facts, FactIndex):
            return facts
        ix = cls()
        ix._walk(facts, None)
        for code in list(ix.tricodes):
            t = next(t for t in TEAMS if t.tricode == code)
            for part in f"{t.city} {t.name}".split():
                ix._add_vocab(part)
        return ix

    def _add_val(self, x: float) -> None:
        x = abs(float(x))
        self.vals.add(x)
        if 0 < x <= 1:
            self.vals.add(x * 100)

    def _add_vocab(self, token: str) -> None:
        t = norm(token).strip(".,;:!?()\"'")
        if not t:
            return
        self.vocab.add(t)
        for part in re.split(r"[-.]", t):
            if part:
                self.vocab.add(part)

    def _add_pair(self, a: float, b: float, both: bool = False) -> None:
        self.pairs.add((a, b))
        if both:
            self.pairs.add((b, a))

    def _scan_string(self, s: str) -> None:
        self.words.update(re.findall(r"[a-z]+(?:-[a-z]+)*", s.lower()))
        for m in _SEASON_RE.finditer(s):
            self.seasons.add(f"{m.group(1)}-{m.group(2)}")
        for m in _CLOCK_RE.finditer(s):
            self.clocks.add(f"{int(m.group(1))}:{m.group(2)}")
        rest = _ISO_DATE_RE.sub(" ", s)
        for m in _CHAIN_RE.finditer(rest):
            nums = [_num(x) for x in _CHAIN_NUM.findall(m.group(0))]
            for x in nums:
                self._add_val(x)
            if len(nums) == 2:
                self._add_pair(nums[0], nums[1])
        for m in re.finditer(r"\d+(?:\.\d+)?", rest):
            self._add_val(float(m.group(0)))
        # "20 pts (...), 10 reb" -> allows the "20-10" shorthand
        m = re.search(r"(\d+) pts\b.*?(\d+) (?:reb|ast)\b", s)
        if m:
            self._add_pair(float(m.group(1)), float(m.group(2)))
        for tok in _CAP_TOKEN.findall(s):
            self._add_vocab(tok)
        for tok in re.findall(r"\b[A-Z]{3}\b", s):
            if tok in _TRICODES:
                self.tricodes.add(tok)

    def _walk(self, obj: Any, key: str | None) -> None:
        if isinstance(obj, bool) or obj is None:
            return
        if isinstance(obj, (int, float)):
            if key not in _ID_KEYS:
                self._add_val(obj)
                if key and any(w in key for w in _MARGIN_KEYS):
                    self.margins.add(abs(float(obj)))
                if key and "prob" in key and 0 < obj < 1:
                    self._add_val(1 - obj)
            return
        if isinstance(obj, str):
            if key not in _ID_KEYS:
                self._scan_string(obj)
            return
        if isinstance(obj, (list, tuple)):
            if key == "periods":
                pass  # handled with its game dict in _derive_game
            for v in obj:
                self._walk(v, key)
            return
        if isinstance(obj, dict):
            uv = obj.get("unit_values")
            if isinstance(uv, dict):
                for unit, vals in uv.items():
                    bucket = self.units.setdefault(str(unit), set())
                    for v in vals if isinstance(vals, (list, tuple)) else [vals]:
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            bucket.add(abs(float(v)))
            if obj.get("direction") in ("above", "below") and self.direction is None:
                self.direction = obj["direction"]
            if _is_scoreboard(obj):
                a, b = list(obj.values())
                self.margins.add(float(abs(a - b)))
                self._add_val(abs(a - b))
                self._add_val(a + b)
                self._add_pair(a, b, both=True)
                self.tricodes.update(obj)
            elif ("home" in obj and "away" in obj and isinstance(obj["home"], dict)
                  and isinstance(obj["away"], dict)):
                self._derive_game(obj)
            for k, v in obj.items():
                if isinstance(k, str) and k in _TRICODES:
                    self.tricodes.add(k)
                self._walk(v, k if isinstance(k, str) else None)

    def _derive_game(self, g: dict) -> None:
        h, a = g["home"], g["away"]
        hp, ap = h.get("periods"), a.get("periods")
        if isinstance(h.get("score"), int) and isinstance(a.get("score"), int):
            self._add_val(abs(h["score"] - a["score"]))
            self.margins.add(float(abs(h["score"] - a["score"])))
            self._add_val(h["score"] + a["score"])
            self._add_pair(h["score"], a["score"], both=True)
        if isinstance(hp, list) and isinstance(ap, list) and len(hp) == len(ap):
            n = len(hp)
            for i in range(n):
                for j in range(i + 1, n + 1):
                    sh, sa = sum(hp[i:j]), sum(ap[i:j])
                    self._add_val(sh)
                    self._add_val(sa)
                    self._add_val(abs(sh - sa))
                    self.margins.add(float(abs(sh - sa)))
                    self._add_pair(sh, sa, both=True)

    # -- queries -----------------------------------------------------------------------
    def supports(self, v: float, decimals: int = 0, *, is_int_text: bool = True) -> bool:
        v = abs(v)
        tol = 0.5 * 10 ** (-decimals) + _EPS
        for f in self.vals:
            if abs(f - v) < _EPS:
                return True
            # rounding to the displayed precision applies only when the fact isn't already
            # that coarse (so "44" matches 43.5, but "43.5" never matches 43.9)
            if abs(f - round(f, decimals)) > _EPS and abs(f - v) <= tol:
                return True
        return False

    def supports_margin(self, v: float, decimals: int = 0) -> bool:
        tol = 0.5 * 10 ** (-decimals) + _EPS
        return any(abs(m - abs(v)) <= tol for m in self.margins)

    def supports_unit(self, unit: str, v: float, decimals: int = 0) -> bool:
        """`v` (as written, `decimals` places) is one of `unit`'s values, exactly or rounded
        to the displayed precision. True when the facts declare no values for `unit`."""
        vals = self.units.get(unit)
        if not vals:
            return True
        tol = 0.5 * 10 ** (-decimals) + _EPS
        return any(abs(f - abs(v)) < _EPS or abs(f - abs(v)) <= tol for f in vals)

    def supports_pair(self, a: float, b: float) -> bool:
        return any(abs(a - x) < _EPS and abs(b - y) < _EPS for x, y in self.pairs)


# ---------------------------------------------------------------------------------------
# name checking
# ---------------------------------------------------------------------------------------

COMMON_CAPS = {norm(w) for w in """
I A The An And But Or So Yet For Nor With Without Within After Before During While When Where
Which Who What How Why That This These Those There Here It Its He She They We You His Her Their
Our Your My Not No Yes If Then Than As At By In Of On To Up Out Off Over Under Into From About
Against Between Through Behind Beyond Despite Still Even Just Also Only Both Each Every All Any
Some Most More Less Much Many One Two Three Four Five Six Seven Eight Nine Ten First Second Third
Fourth Fifth Last Next Now Meanwhile Tonight Today Yesterday Tomorrow Monday Tuesday Wednesday
Thursday Friday Saturday Sunday January February March April May June July August September
October November December Jan Feb Mar Apr Jun Jul Aug Sep Sept Oct Nov Dec Mon Tue Tues Wed Thu
Thur Thurs Fri Sat Sun NBA MVP OT OT2 OT3 TS FG FT PM ORtg DRtg CI Q1 Q2 Q3 Q4 Q5 PT PTS REB AST
STL BLK TOV Eastern Western East West Conference Finals Playoffs Playoff All-Star All-NBA
All-Defensive Christmas Cup Rookie Year Season Regular Game Games Courtside Brief
Pulse Power Rankings No Mr Dr Jr Sr II III IV St Vs Per Net Rating Ratings DraftKings DK
""".split()}
_KNOWN_ABBR = re.compile(r"\b(Jr|Sr|vs|St|Mr|Dr|No|P\.J|T\.J|C\.J|R\.J|A\.J|D\.J|J\.J|O\.G)\.")


def _sentence_initial_positions(text: str) -> set[int]:
    pos = {m.end() for m in re.finditer(r"[.!?:]\s+(?=[\"'“‘(]?)|\n+", _KNOWN_ABBR.sub(
        lambda m: m.group(0).replace(".", "_"), text))}
    pos.add(0)
    # a quote or bracket may sit between the terminator and the word
    return pos


def _clean_tok(tok: str) -> str:
    t = tok.replace("\u2019", "'").replace("_", ".")
    t = re.sub(r"'s?$", "", t)           # possessive
    return t.rstrip(".,;:!?")


def _known(tok: str, ix: FactIndex) -> bool:
    t = norm(_clean_tok(tok))
    if not t:
        return True
    if t in ix.vocab or t in COMMON_CAPS:
        return True
    if _clean_tok(tok).isupper():                                 # acronym / tricode
        if _clean_tok(tok) in _TRICODES:
            return _clean_tok(tok) in ix.tricodes
        return len(_clean_tok(tok)) <= 5
    parts = [p for p in re.split(r"[-.]", t) if p]
    return bool(parts) and all(p in ix.vocab or p in COMMON_CAPS for p in parts)


def check_names(text: str, ix: FactIndex) -> tuple[list[Violation], list[str]]:
    vs: list[Violation] = []
    warns: list[str] = []
    protected = _KNOWN_ABBR.sub(lambda m: m.group(0).replace(".", "_"), text)
    starts = _sentence_initial_positions(text)
    toks = [(m.start(), m.group(0)) for m in _CAP_TOKEN.finditer(protected)]
    # runs of adjacent capitalised tokens (separated by single spaces)
    runs: list[list[tuple[int, str]]] = []
    prev_end = -10
    for pos, tok in toks:
        if runs and pos - prev_end <= 1 and protected[prev_end:pos].strip() == "":
            runs[-1].append((pos, tok))
        else:
            runs.append([(pos, tok)])
        prev_end = pos + len(tok)

    seen: set[str] = set()
    for run in runs:
        first_pos = run[0][0]
        initial = first_pos in starts
        flags = [not _known(tok, ix) for _, tok in run]
        for i, ((pos, tok), bad) in enumerate(zip(run, flags)):
            if not bad:
                continue
            if i == 0 and initial and not all(flags):
                continue           # "Tonight Boston ...": a capitalised common word
            if i == 0 and initial and len(run) == 1:
                continue           # lone sentence-initial word: can't tell name from word
            label = _clean_tok(tok)
            if label not in seen:
                seen.add(label)
                vs.append(Violation("name", label, "not in the facts"))
    for m in re.finditer(r"\b[A-Z]{3}\b", text):
        if m.group(0) in _TRICODES:
            warns.append(f"tricode in prose: {m.group(0)}")
    return vs, warns


# ---------------------------------------------------------------------------------------
# number checking
# ---------------------------------------------------------------------------------------

def check_numbers(text: str, ix: FactIndex) -> list[Violation]:
    vs: list[Violation] = []
    seen: set[str] = set()

    def add(kind: str, token: str, detail: str) -> None:
        key = f"{kind}:{token}"
        if key not in seen:
            seen.add(key)
            vs.append(Violation(kind, token, detail))

    work = text.replace("−", "-")
    work = _ISO_DATE_RE.sub(" ", work)
    work = _MONTH_DAY_RE.sub(" ", work)

    for m in _SEASON_RE.finditer(work):
        raw = f"{m.group(1)}-{m.group(2)}"
        if raw not in ix.seasons:
            add("number", raw, "season not in the facts")
    work = _SEASON_RE.sub(" ", work)

    def clocks(m: re.Match) -> str:
        c = f"{int(m.group(1))}:{m.group(2)}"
        if c not in ix.clocks:
            add("time", m.group(0), "clock time not in the facts")
        return " "
    work = _CLOCK_RE.sub(clocks, work)
    if ix.units:
        for v in check_units(work, ix):
            add(v.kind, v.token, v.detail)

    def ords(m: re.Match) -> str:
        n = int(m.group(1))
        if n > 4 and not ix.supports(n):
            add("number", m.group(0), "ordinal not in the facts")
        return " "
    work = _ORD_RE.sub(ords, work)

    def chains(m: re.Match) -> str:
        raw = m.group(0)
        nums = [_num(x) for x in _CHAIN_NUM.findall(raw)]
        if len(nums) == 2:
            if not ix.supports_pair(nums[0], nums[1]):
                add("pair", raw.strip(), "this pairing/score/record is not in the facts")
        else:
            for x, s in zip(nums, _CHAIN_NUM.findall(raw)):
                if not ix.supports(x, _decimals(s)):
                    add("number", s, "not in the facts")
        return " "
    work = _CHAIN_RE.sub(chains, work)
    work = _SHOT_RE.sub(" ", work)

    if ix.margins:
        for rx in _MARGIN_CLAIM_RES:
            for m in rx.finditer(work):
                x = m.group(1)
                if ix.supports(_num(x), _decimals(x)) and not ix.supports_margin(_num(x), _decimals(x)):
                    add("number", x, "no lead/deficit/margin of that size in the facts")
    for m in _NUM_RE.finditer(work):
        s = m.group(1)
        if not ix.supports(_num(s), _decimals(s)):
            add("number", s + ("%" if m.group(2) else ""), "not in the facts")
    for m in _SPELLED_RE.finditer(work):
        v = _spelled_value(m.group(1))
        if not ix.supports(v):
            add("number", m.group(1), f"{v} is not in the facts")
    return vs


# ---------------------------------------------------------------------------------------
# units (player stories)
# ---------------------------------------------------------------------------------------

UNIT_WORDS = {
    "points": "points", "point": "points", "rebounds": "rebounds", "rebound": "rebounds",
    "boards": "rebounds", "assists": "assists", "assist": "assists", "threes": "threes",
    "three": "threes", "three-pointers": "threes", "three-pointer": "threes",
    "steals": "steals", "steal": "steals", "blocks": "blocks", "block": "blocks",
    "turnovers": "turnovers", "turnover": "turnovers", "minutes": "minutes", "minute": "minutes",
}
_DK_UNIT_RE = re.compile(rf"(?<![\w.])({_NUMBER_RE})\s*(?:more\s+|fewer\s+|extra\s+)?"
                         rf"(?:DK|DraftKings|fantasy)(?:\s+fantasy)?\s+points?\b", re.I)
_STAT_UNIT_RE = re.compile(
    rf"(?<![\w.\-])({_NUMBER_RE})\s+(?:more\s+|fewer\s+|extra\s+|made\s+)?("
    + "|".join(sorted((re.escape(k) for k in UNIT_WORDS), key=len, reverse=True))
    + r")\b", re.I)


def check_units(text: str, ix: FactIndex) -> list[Violation]:
    """Numbers written with a unit must be one of that unit's values (facts' `unit_values`).
    "52.3 DK points" is checked against `dk`, "41 minutes" against `minutes`, "6 threes" against
    `threes`. A bare "points" means real points; "DK points" never reads as real points."""
    vs: list[Violation] = []
    dk_spans = []
    for m in _DK_UNIT_RE.finditer(text):
        dk_spans.append((m.start(), m.end()))
        x = m.group(1)
        if not ix.supports_unit("dk", _num(x), _decimals(x)):
            vs.append(Violation("number", m.group(0).strip(),
                                "not one of the DK-point figures in the facts"))
    for m in _STAT_UNIT_RE.finditer(text):
        if any(a <= m.start() < b for a, b in dk_spans):
            continue
        unit = UNIT_WORDS[m.group(2).lower()]
        x = m.group(1)
        if not ix.supports_unit(unit, _num(x), _decimals(x)):
            vs.append(Violation("number", m.group(0).strip(),
                                f"not one of his {unit} figures in the facts"))
    return vs


# DFS / tipster vocabulary banned in player stories (STYLE.md). Plain "over" and "under" are
# basketball English ("turned it over", "over 30 minutes", "under 20 minutes"); only the betting
# phrases are banned ("the over", "hit the over", "over/under", "o/u", "cover").
STORY_BANNED_RE = re.compile(
    r"(?<![\w/])(?:o/u|over\s*/\s*under|under\s*/\s*over)(?![\w/])|"
    r"\b(?:(?:hit|hits|hitting|hammer|take|took|play|played|bet)\s+)?the\s+(?:over|under)\b"
    r"(?![\s-]+(?:the|a|an|his|her|their|its|\d)\b)|"
    r"\b(?:edges?|lean(?:s|ed|ing)?|value|chalk|sleeper|smash(?:ed|es|ing)?|fade[sd]?|fading|"
    r"locks?|locked|bust|boom|punt|pivot|leverage|ownership|lineups?|salary|cash(?:ed|ing)?|GPP|"
    r"must-play|exposure|stack(?:ed|ing)?|covers?|covered|covering)\b", re.I)
_STORY_ABOVE_RE = re.compile(
    r"\b(?:outperform\w*|outscor\w+ (?:our|his|the) (?:number|projection)|exceed\w*|surpass\w*|"
    r"beat (?:our|his|the) (?:number|projection)|above (?:our|his|the) (?:number|projection))\b",
    re.I)
_STORY_BELOW_RE = re.compile(
    r"\b(?:underperform\w*|fell short|falls short|disappoint\w*|missed (?:our|his|the) "
    r"(?:number|projection)|below (?:our|his|the) (?:number|projection))\b", re.I)


def check_story(text: str, ix: FactIndex) -> list[Violation]:
    """Player-story rules: no DFS-tipster words, and no claim in the wrong direction."""
    vs: list[Violation] = []
    seen: set[str] = set()
    for m in STORY_BANNED_RE.finditer(text):
        w = m.group(0).lower()
        if w not in seen:
            seen.add(w)
            vs.append(Violation("betting", m.group(0), "DFS/betting language is banned"))
    if ix.direction == "below":
        m = _STORY_ABOVE_RE.search(text)
        if m:
            vs.append(Violation("direction", m.group(0), "he finished below our projection"))
    elif ix.direction == "above":
        m = _STORY_BELOW_RE.search(text)
        if m:
            vs.append(Violation("direction", m.group(0), "he finished above our projection"))
    return vs


# ---------------------------------------------------------------------------------------
# style lint
# ---------------------------------------------------------------------------------------

BETTING_RE = re.compile(
    r"\b(?:bet|bets|betting|bettors?|wager\w*|parlays?|fade|fading|smash|hammer the|"
    r"sharp money|best bet|free pick|value play|play the (?:over|under)|lay the points|"
    r"take the (?:home|away|over|under|points|[A-Z]\w+)|(?:covers?|covered|covering) the spread|"
    r"moneyline|ATS|lock of the|a lock|locks? it in|units?\b(?= on| at)|back the (?:home|away|[A-Z]\w+))\b",
    re.I)

# Outside-knowledge words: fine if the facts use them, a violation if the writer brings them.
# Positions and injuries are the classic invented "facts"; quotes and reporting verbs imply sources.
RISKY_WORDS = re.compile(
    r"\b(?:point guard|shooting guard|small forward|power forward|big man|guard|forward|center|"
    r"centre|wing|rookie|veteran|all-star|mvp|seeds?|playoffs?|contract|coach|"
    r"injured|injury|injuries|hurt|sprain(?:ed)?|strain(?:ed)?|ankle|knee|hamstring|concussion|"
    r"illness|soreness|sore|day-to-day|surgery|torn|said|told|according|reportedly)\b", re.I)

CLICHES = [
    "took over", "when it mattered most", "stepped up", "all the right moves", "gutsy",
    "in the blink of an eye", "leave it all on the", "dig deep", "dug deep", "wire to wire",
    "statement win", "put on a clinic", "tale of two halves", "sent a message", "the rest is history",
    "refused to quit", "never say die", "clutch gene", "heart of a champion", "kept it close",
    "at the end of the day", "weather the storm", "firing on all cylinders", "caught fire",
    "got hot", "night to remember", "make no mistake", "in a thrilling",
]
_CLICHE_RE = re.compile("|".join(re.escape(c) for c in CLICHES), re.I)


def count_sentences(text: str) -> int:
    t = _KNOWN_ABBR.sub(lambda m: m.group(0).replace(".", "_"), text.strip())
    t = re.sub(r"\b(?:[A-Z]\.){1,3}", lambda m: m.group(0).replace(".", "_"), t)  # P.J., T.J.
    t = re.sub(r"(?<=\d)\.(?=\d)", "_", t)
    parts = [p for p in re.split(r"(?<=[.!?])[\"')”]*\s+", t) if p.strip()]
    return len(parts)


def count_words(text: str) -> int:
    return len(re.findall(r"[^\s]+", text.strip()))


def check_style(text: str, kind: str, ix: FactIndex | None = None) -> tuple[list[Violation], list[str]]:
    vs: list[Violation] = []
    warns: list[str] = []
    if ix is not None:
        seen_r: set[str] = set()
        for m in RISKY_WORDS.finditer(text):
            w = m.group(0).lower()
            if w not in seen_r and not all(p in ix.words for p in w.split()):
                seen_r.add(w)
                vs.append(Violation("claim", m.group(0), "not in the facts: no positions, injuries, "
                                    "outside context or reported speech"))
        if len(re.findall(r"[\"\u201c\u201d]", text)) >= 2:
            vs.append(Violation("claim", "quotation marks", "never quote anyone"))
    m = BETTING_RE.search(text)
    if m:
        vs.append(Violation("betting", m.group(0), "betting language is banned"))
    for c in sorted({c.lower() for c in _CLICHE_RE.findall(text)}):
        warns.append(f"cliche: {c}")
    if "!" in text:
        warns.append("exclamation mark")
    if re.search(r"[*#`_]{1,}\w", text) or "```" in text:
        warns.append("markdown")
    limits = LIMITS.get(kind)
    if limits:
        _, hard_w, tgt_s, hard_s = limits
        w, s = count_words(text), count_sentences(text)
        if hard_w and w > hard_w:
            vs.append(Violation("length", f"{w} words", f"at most about {limits[0] or hard_w - 20}"))
        if hard_s and s > hard_s:
            vs.append(Violation("length", f"{s} sentences", f"at most {tgt_s}"))
        if limits[0] and w > limits[0]:
            warns.append(f"long: {w} words (target {limits[0]})")
        if tgt_s and s > tgt_s:
            warns.append(f"long: {s} sentences (target {tgt_s})")
    return vs, warns


# ---------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------

def check(text: Any, facts: Any, *, kind: str = "game_story") -> CheckResult:
    """Check one generated string against the facts (a dict/list or a prebuilt FactIndex)."""
    if not isinstance(text, str) or not text.strip():
        return CheckResult([Violation("format", "", "empty or non-string text")])
    ix = FactIndex.build(facts)
    res = CheckResult()
    res.violations += check_numbers(text, ix)
    if kind.endswith("headline") and looks_title_case(text, ix):
        # unknown capitalised words can't be told from names here: ask for sentence case first
        res.violations.append(Violation("format", text[:40], "headline is in Title Case: use sentence "
                                        "case (capitalise only the first word and names)"))
    else:
        nv, nw = check_names(text, ix)
        res.violations += nv
        res.warnings += nw
    sv, sw = check_style(text, kind, ix)
    res.violations += sv
    res.warnings += sw
    if kind == "player_story":
        res.violations += check_story(text, ix)
    return res


# ---------------------------------------------------------------------------------------
# headline style
# ---------------------------------------------------------------------------------------

def looks_title_case(text: str, ix: FactIndex | None = None) -> bool:
    """Most of the words that aren't names (or the first word) start with a capital."""
    toks = re.findall(r"[A-Za-z\u00C0-\u00FF][\w'\u2019\-]*", text)[1:]
    words = [w for w in toks if len(w) >= 4 and not (ix is not None and _known(w, ix))]
    return len(words) >= 2 and sum(w[0].isupper() for w in words) / len(words) >= 0.75


def normalize_headline(text: str) -> str:
    """Strip wrapping quotes and the trailing period (house style: sentence case, no period)."""
    return text.strip().strip("\"'\u201c\u201d").strip().rstrip(".").strip()
