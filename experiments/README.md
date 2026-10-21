# Counter-link experiments: summary (updated 2026-10-07)

How we got from noisy claim-level `CONTRADICTS` links to the graded passage check now in stage 5.
Details, scripts and data are in each folder's README.

## The problem

Stage 5 used to find counter links claim by claim: nearest claim pairs → small NLI model on CPU →
qwen3:8b verifies. The links were mostly not two sides of an argument, and NLI cost ~15 min of CPU
per rebuild. 15% of claims are about the text itself ("The passage…") and they sat in 26% of the links.

## What we tried

| # | Experiment | Result | Verdict |
|---|---|---|---|
| 1 | Issue + stance per claim, grouped by issue (8b) — `stance/` | 6/25 real counters, same as the old links | ✗ groups by topic, forced yes/no invents disagreement |
| 2 | Strict 3-way check on claims — `stance/` | removed 2/3 of pairs, real ones too | ✗ one-sentence claims lose the argument |
| 3 | Same check on whole passages (8b) — `stance/` | only 4/30 of its "opposed" held up on full reading | ✗ |
| 4 | Judge comparison, 619 pairs, 60 labelled — `stance/` | graded 1–5 scale, qwen3:14b, mean of both orders: AUC 0.85 | ✓ bigger model + graded scale |
| 5 | Labelling rule changed: a view counts whoever holds it, incl. one an author describes to reject | counters 5 → 19 of 60; graded 14b on passages AUC 0.90 | ✓ the definition mattered most |
| 6 | Positions with voice/holder + `meta` role (14b) — `positions/` | judging bare statements: AUC 0.79, flags 50/60; `meta` caught credits/intros | ✗ for judging, ✓ `meta` role |
| 7 | Hybrid: positions find candidates, passage check decides — `positions/` | 18/30 flagged pairs real, 3 cross-source; proposes only 8/19 known counters | ~ good precision, poor recall |
| 8 | **Representative test set**: 80 neighbouring pairs, all 9 sources, method-neutral — `testset/` | 10 counters (3 cross-source), 15 junk pairs. Old links AUC **0.51**; graded 14b **0.93**; graded 8b 0.91 (2.5× the flags); strict 3-way 0.65 | ✓ confirms the graded check |
| 9 | Cut-off for the graded score — `testset/` | 0.5: recall 0.51 · **0.45: recall 0.93, precision 0.64** (one order "qualifies", the other "compatible" averages ≈0.48) | ✓ 0.45 (tuned on the same 80 pairs) |
| 10 | **Built into stage 5** (`DIP_COUNTER_*`), full rebuild | 11,214 judgements in 75 min (2.7/s); 767 of 5,607 neighbouring pairs linked, 151 cross-source. Live links on the test set: **precision 0.62, recall 0.93, AUC 0.91** | ✓ in production |
| 11 | Veto pairs where either order is ≥50% "different question" | recall 0.93 → 0.64, precision no better; 133/767 live links have the pattern | ✗ not adopted |

## What we learned

1. Define "counter" first. Judging positions, not authors, changed results more than any model or prompt.
2. Judge whole passages. Every attempt on short claims or statements over-flagged or missed real debates.
3. qwen3:14b + graded scale + both orders + digit probabilities is the best judge found; NLI isn't needed for counters.
4. Real counters are rare (~1 in 8 neighbouring pairs) and mostly inside one text: an author stating an objection and answering it (Pasek's "to be fair…", "voting with your dollar").
5. Near the cut-off it is unreliable both ways: false links when one order says "different question" (501 ↔ 514 caption, 501 ↔ 699), misses when one order says "compatible" (357 ↔ 839, score 0.27).

## Caveats

- All labels are Claude's; 10 counters in the test set, so each is ~10 points of recall; the 0.45 cut-off was tuned on the same pairs.
- Recall is measured only *within* each passage's 10 nearest neighbours. Counters worded too differently to be neighbours are never candidates and aren't counted.
- Only Counter was evaluated. `SUPPORTS`, `ELABORATES` and `EXAMPLE_OF` still come from NLI + qwen3:8b on claims and have never been tested the same way.

## Options for what next

- **A. Your labels** (`python stance/label.py`, ~20–30 min): check Claude's reading; re-score with `python testset/run.py --score-only`.
- **B. Test the other link types** (Evidence, Zoom): the same test-set approach for supports/example/elaborates, which still use the old pipeline.
- **C. Candidate recall beyond neighbours:** at ingest, have the LLM write each passage's strongest opposing view and search near it (the query-time fallback, moved to ingest); measure on pairs outside the 10-NN pool.
- **D. Stage 3 changes:** `meta` role (credits, contents, reading lists, e.g. 514) excluded from Counter; voice/holder so in-text objections show as "the objection the author answers".
- **E. A better "why":** graded links carry no claims, so Counter results have no explanation; asking the judge for the question the passages disagree on would supply one.
- **F. Sources that disagree:** an industry sustainability report or a pro-scaling essay, to test cross-source counters where they should exist.
