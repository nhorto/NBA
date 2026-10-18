{{STYLE}}

TASK
Write the recap for each of last night's games below. One entry per game, in any order.

For each game:
- "headline": at most 10 words, verb-driven, no trailing period, does not just repeat the score.
- "story": 2 to 4 sentences. Sentence 1 is the story of the game (not the score). Then the turning point, then the unsung hero or standout performer with the why. Add an in-game absence/early exit or ejection only if the facts list one.

Use only each game's own facts. Do not mention a team or player from a different game.

FIELD NOTES FOR THE FACTS
- `score` is the final. `periods` are points per quarter/OT.
- `turning_point.run` like "13-4" means that team outscored the other 13-4 over `minutes` between the two `score` snapshots; `reason` says whether it flipped the lead, created the winning margin, or leveled the game.
- `runs` are the game's biggest stretches; "includes_unanswered" is an uninterrupted burst inside the run.
- `unsung_hero[].why` are plain-English reasons you may quote or paraphrase; `winner` says whether he was on the winning team.
- `clutch.key_plays` are late go-ahead/tying scores; `decisive` marks the winning basket.
- `context.our_pregame_model` (sometimes present) is our own model's pre-tip pick: `team`, `win_prob`, `confidence`, `correct`. Mention it in at most half a sentence, and only when it makes the story (a missed upset, a confident call that held up); skip it for toss-ups and when absent.
- `garbage_time_seconds` present means the starters sat for a decided game: say the game was over, do not credit late stats.

FACTS (JSON, last night's games)
{{FACTS}}

OUTPUT FORMAT
Return ONLY this JSON object, no code fences, no commentary:
{"games": {"<game_id>": {"headline": "...", "story": "..."}, ...}}
Include every game_id exactly once, using the exact id strings from the facts.
