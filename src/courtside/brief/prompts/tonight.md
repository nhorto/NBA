{{STYLE}}

TASK
Write the "Tonight" preview line for each game: at most 2 sentences, present/future tense. Say who is favored in our model and by how much, and the one or two factors that matter most from `reasons` and `injuries` (rest, travel, a key absence, form). When our number differs from the market line (`market_home_spread`), say so plainly as a gap between "our number" and "the market line". Never advise anyone to back a side. If no market line is given, do not mention a market. If a game has no reasons or injuries, say only who is favored and by how much.

Phrasing: "Boston is a 71% favorite in our model, projected to win by 7. The market line is closer, Boston by 4.5." Say "our model" for the model and "the market line" for the market; avoid the jargon "our number". When the model and the market agree within a point or so, say they agree. Do not state the same margin twice. The gap field `our_number_minus_market_pts` is positive when we like the home team more than the market does.

`model_prices_injuries` false means our model does NOT account for injuries: present absences as context only and never say they did or did not move our number or the line. (null means unknown: say nothing about it.)

Sign convention: `home_margin` is the model's projected home margin (positive = home wins by that much). `market_home_spread` is the market home spread in betting convention (negative = home favored by that many). `home_win_prob` is the model's probability the home team wins (a fraction). Translate these into plain English yourself ("Boston by 7", "a 71% favorite"), and never show raw signs.

FACTS (JSON, tonight's slate)
{{FACTS}}

OUTPUT FORMAT
Return ONLY this JSON object, no code fences, no commentary:
{"games": {"<game_id>": "<preview text>", ...}}
Include every game_id exactly once, using the exact id strings from the facts.
