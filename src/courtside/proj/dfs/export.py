"""DraftKings upload CSV and "copy lineup" text (docs/PLAYERS.md §2.7).

A lineup here is the parsed ``dfs_lineups.slots_json``: a list of
``{"slot", "player_id", "name", "team", "salary", "mean", "draftable_id"}`` in DK slot order.

**Format, and what is not yet verified.** The CSV has one header row of slot names
(Classic ``PG,SG,SF,PF,C,G,F,UTIL``; Showdown ``CPT,UTIL,UTIL,UTIL,UTIL,UTIL``) and one row per
lineup. Each cell is the draftable id of the player in that slot (``style="id"``, the default:
what DK's bulk-upload template takes). DK's own entry export writes ``Name (ID)`` in these cells
(``style="name_id"``). I could not run DK's uploader from here, so the owner must confirm with
one manual upload of a one-lineup file before anyone relies on it (T1.7 acceptance). If DK
rejects the numeric form, switch the route's default to ``name_id``: it is a single argument.

Only lineups with a ``draftable_id`` in every slot can be exported; `ExportError` names the
first slot that has none. Nothing here logs in to, or enters, DraftKings.
"""

from __future__ import annotations

import csv
import io
from typing import Iterable, Mapping, Sequence

CLASSIC_HEADER = ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL")
SHOWDOWN_HEADER = ("CPT", "UTIL", "UTIL", "UTIL", "UTIL", "UTIL")
STYLES = ("id", "name_id")


class ExportError(ValueError):
    """A lineup cannot be written as a DK row (missing draftable id, wrong slots)."""


def header_for(lineup: Sequence[Mapping]) -> list[str]:
    """Slot names of a lineup, in order (matches the DK template for Classic and Showdown)."""
    return [str(s.get("slot")) for s in lineup]


def default_header(showdown: bool = False) -> list[str]:
    return list(SHOWDOWN_HEADER if showdown else CLASSIC_HEADER)


def dk_cell(slot: Mapping, style: str = "id") -> str:
    did = slot.get("draftable_id")
    if did in (None, ""):
        raise ExportError(f"{slot.get('name') or slot.get('player_id')} has no draftable id "
                          f"for slot {slot.get('slot')}")
    if style == "name_id":
        return f"{slot.get('name') or ''} ({did})"
    if style != "id":
        raise ExportError(f"unknown style {style!r}")
    return str(did)


def dk_row(lineup: Sequence[Mapping], style: str = "id") -> list[str]:
    return [dk_cell(s, style) for s in lineup]


def to_csv(lineups: Iterable[Sequence[Mapping]], *, style: str = "id",
           showdown: bool = False) -> str:
    """The upload file: header row + one row per lineup. The header is the lineups' own slot
    order (they are stored in DK order); if there are no lineups it is the template header."""
    lineups = [list(lu) for lu in lineups]
    header = header_for(lineups[0]) if lineups else default_header(showdown)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for lu in lineups:
        if header_for(lu) != header:
            raise ExportError(f"lineup slots {header_for(lu)} do not match header {header}")
        w.writerow(dk_row(lu, style))
    return buf.getvalue()


EDIT_HEADER = ("Entry ID", "Contest Name", "Contest ID", "Entry Fee")


def to_edit_csv(entries: Iterable[Mapping], *, header: Sequence[str] | None = None,
                style: str = "name_id", showdown: bool = False) -> str:
    """DK's edit-entries upload (late swap, PLAYERS.md §2.5): ``Entry ID, Contest Name,
    Contest ID, Entry Fee`` then the slot columns, one row per entry. Each entry is
    ``{entry_id, contest_name, contest_id, entry_fee, slots}``; a slot is a lineup slot dict
    (written with `dk_cell` in `style`) or ``{"cell": text}`` to write DK's own cell back as it
    was (locked and unchanged slots). Same caveat as `to_csv`: the owner uploads it by hand."""
    entries = [dict(e) for e in entries]
    cols = list(header) if header else (
        header_for([s for s in entries[0]["slots"]]) if entries else default_header(showdown))
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(list(EDIT_HEADER) + cols)
    for e in entries:
        slots = list(e.get("slots") or [])
        if len(slots) != len(cols):
            raise ExportError(f"entry {e.get('entry_id')} has {len(slots)} slots, the header "
                              f"{len(cols)}")
        cells = [s["cell"] if "cell" in s else dk_cell(s, style) for s in slots]
        w.writerow([str(e.get(k) or "") for k in ("entry_id", "contest_name", "contest_id",
                                                  "entry_fee")] + cells)
    return buf.getvalue()


def copy_text(lineup: Sequence[Mapping], *, salary: int | None = None,
              mean: float | None = None, title: str | None = None) -> str:
    """Plain text for pasting into a note or chat, one slot per line::

        Cash lineup
        PG  Jalen Brunson (NYK)  $9,800  44.1
        ...
        Salary $49,800 | Projection 312.4
    """
    lines = [title] if title else []
    for s in lineup:
        sal = s.get("salary")
        mean_s = s.get("mean")
        parts = [f"{str(s.get('slot')):<5}{s.get('name') or s.get('player_id')}"
                 + (f" ({s.get('team')})" if s.get("team") else "")]
        if sal is not None:
            parts.append(f"${int(sal):,}")
        if mean_s is not None:
            parts.append(f"{float(mean_s):.1f}")
        lines.append("  ".join(parts))
    tail = []
    total = salary if salary is not None else sum(int(s.get("salary") or 0) for s in lineup)
    tail.append(f"Salary ${int(total):,}")
    if mean is not None:
        tail.append(f"Projection {float(mean):.1f}")
    lines.append(" | ".join(tail))
    return "\n".join(lines)
