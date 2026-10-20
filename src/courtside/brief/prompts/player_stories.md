{{STYLE}}

TASK
Write the "Player stories" lines for this morning's Brief: last night's biggest gaps between a player's DraftKings fantasy points and our pre-game projection, with the reason in box-score English. One or two sentences per player, at most 45 words.

Model sentence: "Jalen Duren scored 61.8 DK points vs 34.0 projected: he played 38 minutes (29 expected) and grabbed 21 rebounds (8.5 projected)."

HOW TO READ AN ENTRY
- `dk.actual` / `dk.expected` are DK fantasy points: what he scored and our pre-game projection. `dk.diff` = actual minus expected. `direction` is "above" or "below" our projection.
- `minutes.actual` vs `minutes.expected` is playing time. `line` is his box score (`fg` like "12-18 FG" is field goals made-attempted, `fg3` is threes). `projected` is our full-game projection of each stat.
- `reasons` are the parts of the gap worth naming, biggest first, already checked to point the right way: "minutes" = playing time, "points"/"rebounds"/"assists"/"threes"/"steals"/"blocks"/"turnovers" = that stat against its projection, "double_double"/"triple_double" = he earned that DK bonus. `dk_points` is how many DK points that reason moved him (positive pushed him up, negative pulled him down). Lead with the first reason; use at most three.
- `context` facts you may add when they plainly fit the minutes story: `blowout_margin` (a blowout), `overtime` (extra periods), `fouled_out`, `ejected`, `teammates_not_playing` (regulars who did not play). Never claim a cause the facts do not state.
- `did_not_play` true: say he did not play and what we had projected; give no reason.

RULES
- Always write the unit as "DK points" (never just "points" for fantasy points, which means real points). Real stats use their own words: "34 points", "21 rebounds", "6 threes", "41 minutes".
- Use only numbers from the entry, exactly as given; you may round a decimal to a whole number. Do not compute new numbers.
- Never use DFS or betting words: no "edge", "lean", "value", "chalk", "smash", "fade", "lock", "cover", "bust", "boom", "lineup", "salary", "ownership", and never the betting phrases "the over", "the under", "hit the over", "over/under", "o/u". Plain "over" and "under" are fine in basketball sense ("turned it over", "over 30 minutes"). Say "above our projection" / "below our projection" or "vs 38.0 projected".
- Never say he beat our projection when `direction` is "below", or fell short when it is "above".
- Never call anyone injured, never guess why someone did not play, no positions, no quotes.
- Name: full name on first mention. No tricodes. No clichés, no exclamation marks, no markdown.

ENTRIES (JSON, one object per player; `id` is the key you must answer under)
{{FACTS}}

OUTPUT FORMAT
Return ONLY this JSON object, no code fences, no commentary:
{"stories": {"<id>": "<one or two sentences>", ...}}
Include every id exactly once, using the exact id strings from the entries.
