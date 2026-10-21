# Counter-link experiment: issue/stance vs NLI (2026-10-07)

Question: does issue-and-stance extraction find better Counter pairs than the current
NLI-propose / LLM-verify pipeline (stage 5)? Run on 6 of the 8 sources (items 2,3,4,6,7,8;
373 passages) with qwen3:8b. Reads `data/dip.sqlite` read-only; nothing in the main DB changes.

## Scripts (run in order, with the `dip` virtualenv's python)

| Script | What it does | Output |
|---|---|---|
| `extract.py` | per passage: 0–3 positions as (claim, neutral yes/no issue, stance) | `positions.json` |
| `group.py 0.25` | embed issues, agglomerative clustering (cosine distance 0.25), then one LLM call per group writes a shared question and re-labels each stance against it; yes × no across passages = counter pair | `groups.json`, `pairs.json` |
| `compare.py` | counts vs current `CONTRADICTS` links, anchor passages, blind 50-pair sample | `sample.txt`, `sample_key.json` |
| `verify.py` | strict 3-way check (opposed / compatible / different_question) on one-sentence claims, applied to both candidate sets | `verified.json` |
| `verify_passages.py` | same strict check, but shown the two full passages | `verified_passages.json` |

`judged.json` holds the blind judgements of the 50-pair sample.

## Results

| | Current links | Issue/stance |
|---|---|---|
| Passage pairs proposed | 291 | 355 (only 27 also in current) |
| Blind sample: real or arguable counters | 6 / 25 | 6 / 25 |
| After strict check on claims | 101 kept | 93 kept |
| After strict check on passages | 81 kept | 62 kept |

Pairs kept by the passage-level check, judged by hand (29 = all 17 cross-source + 12 random
within-source): 9 clear counters + 4 arguable ≈ 45%, up from ≈ 24% before the check.

Cost: extraction 1.6 s/passage wall (4 parallel), grouping 124 calls / 82 s, passage-level check
2.2 pairs/s (619 pairs / 284 s). NLI xsmall on this laptop's CPU: 56 pairs/s, ≈15 min for a full
corpus rebuild.

## Findings

1. Issue/stance extraction on its own is no better than NLI + LLM: the same ~1 in 4 precision, just
   on different pairs. Grouping issue embeddings collects questions on the same topic, not the same
   question, and forcing a yes/no answer invents opposition (a claim describing a problem gets
   labelled as disagreeing with a claim that criticises it).
2. The verifier, not the candidate generator, is the lever. Letting it answer "compatible" or
   "different question", and showing it full passages instead of one-sentence claims, roughly doubles
   precision for either candidate set. Judging claims alone also drops real debates, because the
   claims leave out the argument.
3. NLI adds nothing that embedding-nearest candidates plus the passage-level check don't give, so it
   can be dropped.
4. The corpus has few real two-sided debates: most sources are critical of big tech and AI. Even
   GCC "Digital" vs Data Center Fights (839 vs 361), the best example on the artifact, is judged
   "compatible" at passage level, which is defensible. Counter should be allowed to return nothing.
5. The passage-level check also returns the question the two passages disagree on, which could be
   shown as the "why" on each Counter result.

Caveats: one judge (Claude), small samples (25 + 25, then 29), one 8B model, prompts not tuned.

## Judge comparison on the 619 pairs (2026-10-07, part 2)

Scripts: `judges.py <strict3|graded> <model>` → `results/`, `eval.py` scores them against `gold.json`
(60 pairs, stratified 30/30, labelled blind from full passages by Claude; `label.py` lets a person
relabel, and their labels win). Only 5 of 60 gold pairs are counter-worthy, all "qualifies", so the
numbers below are very noisy.

| Judge | AUC | Precision @0.5 | Recall @0.5 | Flags (of 619) | Time |
|---|---|---|---|---|---|
| graded 1–5, qwen3:14b, mean of both orders | **0.85** | 0.24 | 0.86 | 117 | 805 s |
| graded 1–5, qwen3:8b, mean of both orders | 0.66 | 0.13 | 1.00 | — | 267 s |
| strict 3-way, qwen3:8b | 0.72 | 0.24 | 0.55 | — | 231 s |
| strict 3-way, qwen3:14b | 0.48 | 0.00 | 0.00 | 38 | 213 s |

- Gold is "qualifies" (partial pushback); the strict 3-way scheme has no slot for it, and 14b
  sorts those into "compatible". 14b's "opposed" calls are mostly "A describes the view B argues
  against" pairs (e.g. 539 universalist assumptions vs 567 placefulness), which gold labels compatible.
- 8b overuses digit 2 (420 of 1,238 judgements) and almost never says 5 (supports); 14b uses the
  whole scale (5-way accuracy 0.52 vs 0.25).
- Requiring both orders to agree loses the real counters: the two orders often disagree.
