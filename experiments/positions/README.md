# Positions experiment (2026-10-07)

Follows `../stance/`. Idea: extract *positions* with their *voice* (does the text argue for it,
report it, or set it up to reject it?) and *holder*, so a view an author quotes or argues against can
serve as a counter without being attributed to that author. Same 6 sources (items 2,3,4,6,7,8; 375
passages), qwen3:14b throughout, `data/dip.sqlite` read-only.

## Scripts

| Script | What it does | Output |
|---|---|---|
| `extract.py` | per passage: role (now incl. `meta`) + 0–3 positions {statement, voice, holder}. Context: the previous and next passage and the document's stage-3 summary | `positions.json` |
| `judge.py` | each position's 6 nearest positions in other passages (skipping same-author argued/argued pairs) plus every position pair inside the 60 gold pairs; graded 1–5 judge on the two statements alone, both orders | `judged__qwen3-14b.json` |
| `eval.py [--top N]` | passage pair score = max over its position pairs; compares with the passage judges in `../stance/results/` | |

`gold.json` = the 60 stance gold pairs relabelled under the position rule: counter-worthy if one
passage's main content is a position (argued, reported, or described in order to reject it) that the
other argues against; two passages rejecting the same view are compatible. 19 of 60 are
counter-worthy (5 before); 10 of the 14 "opposed" come from Designing Friction.

## Results

Extraction: 375 passages in 15 min (3.4 s/passage wall); 936 positions (716 argued, 135 reported,
85 rejected); 14 passages tagged `meta` (series intro 792, Data Center Fights credits 343–347,
acknowledgements). Judging: 4,917 position pairs in 47 min.

| Approach (on the relabelled gold) | AUC | Precision | Recall | Flags of 60 |
|---|---|---|---|---|
| passages, graded qwen3:14b (from ../stance) | **0.90** | 0.55 | 0.55 | 19 |
| passages, graded qwen3:8b | 0.90 | 0.46 | 1.00 | 35 |
| passages, strict 3-way qwen3:8b | 0.73 | 0.76 | 0.50 | 17 |
| **positions**, graded qwen3:14b on statements | 0.79 | 0.27 | 1.00 | 50 |

## Findings

1. The labelling rule mattered more than the method. With the rule changed, the passage-level
   graded judge from `../stance` is the best (AUC 0.90): when it sees the full passages it already
   recognises "a passage describing a view" vs "a passage arguing against it".
2. Judging bare position statements is worse. Out of context, the statements read as more opposed
   than they are: 982 of 4,504 nearest-neighbour pairs score ≥ 0.8. And scoring a passage pair by
   its best position pair flags almost everything (50 of 60).
3. Voice is what makes position pairs readable. Of the high-scoring pairs, 458 are argued +
   rejected: mostly a text against the view it sets up (Small AI's "universalist models" against its
   own placefulness, many times over), or two texts that reject the same view and so *agree*. Only
   argued + argued cross-source pairs are real disagreements between texts; about 1 in 3 of the top
   24 hold up (e.g. 355 vs 830 individual action, 348 vs 822/823/824 how much digital emissions matter).
4. The `meta` role works and is cheap: it removes series intros, credits and acknowledgements from
   the claim graph.
5. Bad extractions leak through: an empty-holder "Data centers have remained unchanged over the
   years" produced 4 of the top 30 pairs.

Caveats: one labeller (Claude), 60 gold pairs with a Designing Friction cluster, no tuning.

## Hybrid: positions find candidates, the whole-passage judge decides (`hybrid.py`)

Candidates: passage pairs behind any nearest-neighbour position pair scoring ≥ 0.8, except
rejected + rejected → 771 passage pairs, only 91 of which were among the 619 from claim search.
Judged with `../stance/judges.py`'s graded prompt, qwen3:14b, both orders (680 new pairs, 459 s).

- 100 of 771 flagged (mean score ≥ 0.5); 28 of those were also in the claim-search set.
- Blind hand check of 30 flagged pairs (`hybrid_sample_judged.json`, position rule): **18 / 30
  counter-worthy**, but only 3 of those are cross-source. Real finds the claim search never proposed:
  359 vs 361 and 354 vs 359 (Pasek states the "voting with your dollar" objection and then answers it),
  413 vs 414 ("to be fair… green power purchasing agreements"), 589 vs 673 (user-side controls vs
  "putting the onus on the user sidesteps the problem"), 558 vs 559 (colonial pines that "take on new
  meanings").
- Candidate recall is the weak point: of the 19 counter-worthy gold pairs, position search proposes
  only 8; the other 11 are never anyone's 6 nearest neighbours (e.g. 862 vs 871, worded too
  differently). Claim search and position search find largely different pairs.
- A few passages dominate: 539 (universalist assumptions) is in 15 of the 100 flagged pairs,
  858 (frictionless trend) and 414 (green power deals) in 10 each.
