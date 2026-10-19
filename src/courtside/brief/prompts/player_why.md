{{STYLE}}

TASK
Rewrite the post-game "why" line for each player below. Each entry is one player-game with our pregame projection decomposed EXACTLY into signed DK-point contributions: `dk.diff` = `dk.actual` - `dk.expected` = the sum of the `contributions`. Write ONE or TWO short sentences (at most 45 words) that say why he beat or missed our number, leading with the biggest contribution.

HOW TO READ AN ENTRY
- `dk` is DraftKings fantasy points: `actual`, `expected` (our pregame number) and `diff` (actual minus expected; positive = he beat our number).
- `contributions` are signed DK points, already sorted by size. A POSITIVE `dk_points` pushed him above our number; a NEGATIVE `dk_points` pulled him below it. They sum to `dk.diff`. `source` "minutes" is playing time (compare `actual` vs `expected` minutes); "points", "threes", "rebounds", "assists", "steals", "blocks" and "turnovers" are production per minute compared with his usual rate (`actual` is what he finished with; `expected_in_his_minutes` is what our rates implied for the minutes he actually played, NOT a full-game projection: a short night has a lower figure, so never compare `actual` with it as if it were his pregame projection, and do not call it a "projection" or "projected"); "bonus" is the double-double / triple-double bonus.
- `context` holds the game facts you may add: `overtime` (number of extra periods), `blowout_margin`, `started`, `teammates_not_playing` (regulars of the previous games who did not play).
- Mention only the one to three largest contributions (an absolute size of 1.0 or more), plus at most one context fact when it plainly explains a minutes contribution (for instance a blowout or a teammate not playing). Say what happened; never claim a cause the facts do not state, and never call anyone injured.

SIGN RULES (machine-checked)
- Never describe a negative contribution as helping, adding or boosting; never describe a positive one as costing or dragging. A negative contribution is a cost: "cost him 3.2 points", "3.2 fewer from rebounds". A positive one is a gain: "added 9.0 from rebounds".
- When you write a contribution as a signed figure, keep its sign: "+9.0 from rebounds", "-3.2 from assists".
- If `dk.diff` is positive he beat our number; if negative he fell short. Do not say he beat it when he fell short or the reverse.
- If he did not play, say so plainly and do not invent a reason.

NUMBERS
- Use only numbers from the entry, exactly as given or rounded to a whole number. Do not add, subtract or derive anything new. Do not state a percentage.
- Call the projection "our number" or "our projection". Write "DK points" or "fantasy points" for the unit. Use the team nickname if you need the team (the `team` code is a tricode, never write it); use surnames after the first mention.
- No betting words, no clichés, no exclamation marks, no markdown, no quotes.

ENTRIES (JSON, one object per player; `id` is the key you must answer under)
{{FACTS}}

OUTPUT FORMAT
Return ONLY this JSON object, no code fences, no commentary:
{"stories": {"<id>": "<one or two sentences>", ...}}
Include every id exactly once, using the exact id strings from the entries.
