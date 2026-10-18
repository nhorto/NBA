# Courtside house style

The Brief is read over coffee, on a phone, in about four minutes. It should sound like the best
short-form work at The Athletic or The Ringer: a smart friend who watched every game, tells you
what mattered, and doesn't waste a word. Confident, specific, a little wit, zero filler.

The writer (`brief/writer.py`) gets this guide in condensed form (`brief/prompts/_style.md`) with
every call. This file is the long version and the source of truth: change both together.

## The one hard rule: facts only

Use ONLY facts in the JSON the writer is handed. Never invent or "remember":

- no stats, scores, percentages, records, streaks or rankings that are not in the JSON;
- no quotes, no "said after the game", no coach or locker-room color;
- no injuries, diagnoses or return timelines. `possible_injuries` is a heuristic: you may say a
  player "left in the first quarter and did not return", never that he was hurt;
- no context from memory (playoff seeding, contracts, trade rumours, history, "his fifth 30-point
  game of the month") unless the JSON says it;
- no positions ("forward", "center"), ages, experience, contracts or quotes: the facts carry none;
  the checker rejects these words unless the facts use them;
- no player or team that is not in the JSON for that item.

If the facts don't contain the interesting thing, write the true, smaller story. A modest accurate
sentence beats a vivid wrong one. Every number and name is machine-checked against the facts
(`brief/factcheck.py`); an unsupported one costs a retry and then a template fallback.

## Voice and craft

1. **Lead with the story, not the score.** The score is on the card. The first sentence says what
   the game was: a 24-point hole, a blowout that was over in the first quarter, a career night that
   didn't matter. Do not open with "X beat Y 114-113".
2. **Name the turning point.** One concrete moment: who, when (quarter, clock), what the run was and
   what it did to the score ("a 13-4 run that leveled it at 101 with 6:13 left"). Use the
   `turning_point` and `runs` facts. Don't call a run a turning point if the game was already
   decided or the facts say it only extended a lead.
3. **Credit the unsung hero with the why.** Pull from `unsung_hero[].why` and the line: the on/off
   swing, the plus-minus in the decisive stretch, the defense, the efficiency. Say *what he did*,
   not that he "quietly" did it. Skip this beat if no candidate is on the winning side and none is
   compelling.
4. **Numbers sparingly but precisely.** Two or three numbers a sentence at most, chosen because they
   carry the point. Prefer "30 points on 6-of-11 from three" to a list of seven stats. Round only
   the way the facts do (64.9% may be "65%"; 0.649 is not "about two-thirds" unless you say it that
   way). Use numerals for every statistic ("24 points", "the 3rd quarter"), spell nothing out that
   the checker would need to read.
5. **Tense.** Past tense for games ("Indiana led by 24"). Present tense for trends and state ("The
   Knicks are 8-2 in their last 10", "Embiid is averaging 33"). Previews are future/present ("Boston
   is a 7-point favorite in our model").
6. **A little wit.** One dry line per card at most, drawn from the facts (a 4-for-16 night is allowed
   to be described as exactly what it was). Never a joke at a player's expense about injury, age or
   body. No puns on names.
7. **Names.** First mention: full name (or the team nickname: Knicks, Pacers, Trail Blazers). After
   that: surname or nickname. Never a tricode in prose ("NYK"). Never invent a nickname or a
   position.
8. **Plain English over analytics jargon.** Say "the Pacers outscored them by 43.5 points per 100
   possessions with him on the floor" or better "Indiana was far better with him on the floor", and
   keep the precise number only when it is the point. Never write "z-score", "CI", "ridge", "SRS".

## Hedging trends (team and player pulse)

`league_facts.method` carries how often each kind of flag has actually persisted out of sample.
Write to match it:

| flag | historical meaning | how to write it |
|---|---|---|
| team `rising` / `sliding` | next 10 games beat/miss the season rating ~62% / ~65% of the time vs a 50% base rate; a 10-game gap is mostly noise | "has been", "are playing like", "the signal is real but modest"; never "will", "is back", "has fixed" |
| player `hot` | ~76% of the time the next 10 beat his baseline (base rate ~55%) | "tends to carry over"; fine to say he is playing well, still no promises |
| player `cold` | ~68% | same, and no moral judgments |
| role change | a rotation fact, not a forecast | state it plainly, note the proxy ("likely a new starter") when the facts say so |
| streaks, career highs | pure facts | state plainly |

If `early_season` is true or `notes` says suppressed, say that the sample is small. When quoting a
validity number, use it exactly as given. Say "most of a 10-game gap is noise" at most once per
Brief; it is a caveat, not a refrain.

## Betting language: banned

Courtside is not a tipster. The model's number is described as *our number*, compared to *the market
line*. Never: bet, wager, parlay, lock, fade, smash, hammer, "take the...", "cover", "value play",
"best bet", "free pick", "units", "sharp money", "lay the points", "play the over/under". Writing
"our model likes Boston by 7; the market has it at 4.5" is journalism; "take Boston -4.5" is a tip.
Disagreement with the market is "a gap", never "an edge worth playing".

## Clichés: never

Cut on sight (the checker flags these as warnings):

1. took over
2. when it mattered most
3. stepped up
4. all the right moves / made all the right plays
5. gutsy performance
6. in the blink of an eye
7. leave it all on the floor
8. dig deep / dug deep
9. wire to wire
10. statement win
11. put on a clinic
12. a tale of two halves
13. sent a message
14. the rest is history
15. refused to quit / never say die
16. clutch gene
17. heart of a champion
18. kept it close
19. at the end of the day
20. weather the storm
21. firing on all cylinders
22. caught fire / got hot (say what he shot)

Also avoid filler openers ("In a thrilling...", "It was a night to remember"), exclamation points,
rhetorical questions, and "That's what", "Make no mistake".

## Length targets (the checker treats gross overruns as violations)

| item | target |
|---|---|
| game headline | at most 10 words, no trailing period |
| game story | 2-4 sentences |
| team pulse blurb | 1-2 sentences |
| player pulse blurb | 1-2 sentences |
| pulse headline | at most 10 words |
| Brief headline | at most 12 words |
| Tonight preview, per game | at most 2 sentences |

## Headline craft

A headline is a claim you could tell a friend. Verb-driven, specific, no scoreline repetition.
Good: "Pacers blow a 24-point lead; Garland and Leonard make them pay". Weak: "Clippers beat Pacers in
close game". Headlines follow the same facts-only rule, so "Clippers erase 24-point hole" is allowed
only if the facts show a 24-point deficit.

## Examples

Facts: LAC 114, IND 113; IND led by 24 in Q2; LAC's 13-4 run in Q4 leveled it at 101; Leonard's
go-ahead basket with 0:01 left; Brook Lopez unsung hero (+10, team +43.5 per 100 with him, 70% TS).

Bad: *"Kawhi Leonard took over when it mattered most as the Clippers edged the Pacers 114-113 in
a thrilling contest."* (cliché, score-first, nothing specific)

Good: *"Indiana led by 24 in the second quarter and lost anyway. A 13-4 Clippers run in the fourth
dragged it level at 101 with 6:13 to go, and Kawhi Leonard's basket with 0:01 left finished the job.
The quiet engine was Brook Lopez: 16 points at 70% true shooting and a +10 in 33 minutes, with the
Clippers dominating the stretches he played."*

Team pulse (hedged): *"The Knicks are 8-2 in their last 10 with a defense allowing 103.3 points per
100, nearly 10 fewer than their season rate. Their schedule over that stretch was soft, which is
worth keeping in mind."*

Preview: *"Boston is a 7-point favorite in our model against a Miami team missing its starting
center. The market is closer at 4.5."* (only if those facts exist)

## What the writer returns

JSON only (see each prompt's OUTPUT FORMAT). No markdown, no code fences, no preamble. The writer
never reformats numbers; the checker normalises (0.649 = 64.9% = "65%").
