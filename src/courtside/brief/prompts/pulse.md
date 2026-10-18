{{STYLE}}

TASK
Write the Pulse section: a short headline, one blurb per team, one blurb per player. Each blurb is 1 to 2 sentences and about its own subject only. Do not mention other teams or players in a blurb (the checker scopes each blurb to its own entry), except the team a player plays for.

- "headline": at most 10 words; the one thing a reader should take from the pulse (e.g. a team surging, a collapse, a notable run of form).
- Team blurbs: say what changed and why, using the entry's `reasons`, `off`/`def` shifts, `sos_last10` (opponent difficulty), and `key_absences` where relevant. Hedge: these are 10-game signals; use the validity numbers in `method` honestly and do not overstate. A team streak or power move is a plain fact.
- Hedge in at most one short clause ("a 10-game signal, real but modest"); never stack caveats or contradict yourself.
- Player blurbs: say what he has been doing against his own baseline (`recent` vs `season`), or the role change in plain English, or the streak/career high. Quote at most two or three numbers. Hedge hot/cold with `hot_cold_validity` when you forecast anything; otherwise just describe.
- `early_season` true or non-empty `notes`: the sample is small; say so briefly in the headline-adjacent blurbs where relevant.

METHOD AND VALIDITY (how much to trust each signal)
{{METHOD}}

TEAM ENTRIES (key = tricode)
{{TEAMS}}

PLAYER ENTRIES (key = player key)
{{PLAYERS}}

OUTPUT FORMAT
Return ONLY this JSON object, no code fences, no commentary:
{"headline": "...", "teams": {"<team key>": "..."}, "players": {"<player key>": "..."}}
Include every key from the entries above exactly once, using the exact key strings.
