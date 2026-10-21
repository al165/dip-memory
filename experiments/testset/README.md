# Representative Counter test set (2026-10-07)

Built after ingesting the DataCenterDynamics orbital data centres article (9 sources, 759 passages).
The Medium "history of privacy" URL failed to fetch (Medium blocks automated requests). The
pre-ingest database and wiki are backed up as `data/dip.sqlite.bak-2026-10-07` and
`data/wiki.bak-2026-10-07`; the ingest log is `ingest-2026-10-07.log`.

## How it was built (`build.py`)

- **Pool, independent of any method:** every passage with its 10 nearest passages by embedding
  (link/navigation passages excluded) = 5,607 pairs (1,215 cross-source, 4,392 within-source).
- **Sample:** 40 cross-source + 40 within-source pairs at random, each passage used at most once,
  each source capped at 15% of pair ends. All 9 sources appear (7–24 pair ends each).
- **Labels:** `gold.json`, labelled blind by Claude from the full passages under the position rule
  (stored in the file). `junk_side` marks the 15 pairs where one side is credits, contents, a
  reference list or a fragment. Add your own labels with `python ../stance/label.py` (they override Claude's in `run.py`).

Labels: 34 different question, 25 compatible, 11 supports, 6 qualifies, 4 opposed, so 10 of 80 are
counter-worthy (7 within-source, 3 cross-source).

## Results (`run.py`; metrics re-weighted to the pool)

| Judge | Precision | Recall | AUC | Flags of 80 | Time for 80 |
|---|---|---|---|---|---|
| current stage 5 (NLI + qwen3:8b verify) | 0.18 | 0.13 | 0.51 | 7 | – |
| **graded, qwen3:14b, whole passages** | **0.61** | 0.51 | **0.93** | 8 | 174 s |
| graded, qwen3:8b | 0.47 | 0.80 | 0.91 | 20 | 35 s |
| strict 3-way, qwen3:8b | 1.00 | 0.29 | 0.65 | 3 | 14 s |

- The current stage-5 links are no better than chance on a representative sample (AUC 0.51).
- The graded whole-passage judge ranks well with either model (AUC 0.91–0.93); 14b is more precise,
  8b flags 2.5× as many pairs.
- No judge flagged any of the 15 junk pairs.
- Caveats: one labeller, 10 positives, so each positive is ~10 points of recall.
