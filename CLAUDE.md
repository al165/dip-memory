# dip-memory

Local-first document memory for exploring a corpus through seven moves: Neighbours, Counter, Evidence, Zoom in, Zoom out, Images, Question. Ingest CLI → single SQLite file + generated markdown wiki → FastAPI server with a test UI. No cloud APIs: LLM and embeddings go through Ollama, the NLI cross-encoder runs on CPU via sentence-transformers.

Deeper docs (keep them in sync when behaviour changes):
- `../../02_Documentation/retrieval-techniques.md`: every technique and setting
- `../../02_Documentation/adding-data.md`: adding sources, re-runs, troubleshooting
- The "How DIP Memory Works" artifact (https://claude.ai/artifact/Q1vc3HW73rYaoN2ehxkHof) explains the system with real examples and numbers from `data/dip.sqlite`. If you change stages or rebuild, its figures may go stale.

## Commands

Python env: the `dip` pyenv virtualenv (set by `DIP/.python-version`). Run everything from this directory.

```bash
python -m pytest tests                      # unit tests, no models needed (fixture DB)
python -m tests.eval_gold                   # live eval against the ingested DB (needs Ollama)
python -m dip.ingest.run <files|dirs|URLs|.urls>   # ingest; reruns stages 4-6 over everything
python -m dip.ingest.run --stages 5,6       # rerun selected stages only
python -m dip.ingest.run --status           # print DB contents
python -m dip.ingest.run --lint             # wiki lint → data/wiki/lint.md
python -m dip.ingest.run --stages 7         # image library: 00_Data/images/ → data/wiki/images/*.md + vectors
python -m dip.ingest.run --concepts         # precompute passage concepts for the Images action (LLM)
uvicorn dip.api:app --port 8000             # UI at /, API docs at /docs
sqlite3 data/dip.sqlite                     # inspect directly (schema in dip/db.py)
```

Other flags: `--reset` (deletes DB and wiki), `--force`, `--dedupe [--dry-run]`, `--rescore-summaries`.

## Config

All settings are in `dip/config.py`. Each can be overridden by an env var of the same name or by `.env`. **`.env` overrides the code defaults:** it sets `DIP_LLM_MODEL=qwen3:14b`, `DIP_LLM_NUM_CTX=8192` and `OLLAMA_URL`, which points to the GPU machine on the LAN (192.168.0.15), not localhost. Code defaults say `qwen3:4b`; the current build uses 14b throughout. When running on the GPU machine itself, these URLs still resolve to the same host. The ingest refuses to start unless the embed server answers, so start it first: `cd embed_server && uvicorn server:app --host 0.0.0.0 --port 11500` (the `dip` env has CUDA torch; `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` puts it on the RTX 3070 and leaves the 5060 Ti to Ollama).

Builds: `data-embeddinggemma2/` is the current build: a fresh full ingest of `../../00_Data/ingest` on 2026-10-21 (10 sources, 808 passages, gemma vectors for clusters and edges too). Earlier builds (`data/` nomic, the old `data-embeddinggemma2/` re-embed, `data-qwen3-4b/`) are in `data-archive-2026-10-21/`, caches included; point `DIP_DATA_DIR` there to use one. The Medium URL in `reading-list.urls` can't be fetched (Medium blocks it); the same article is ingested from the local `.md`.

Embedding backend experiment: `DIP_EMBED_BACKEND=st` swaps Ollama/nomic for a local sentence-transformers model (default `google/embeddinggemma-2`, text + image in one space, CPU bf16, ~1 s per query). Image items then blend the image's own vector into their nodes (`common.blend_image_vectors`). `python -m dip.reembed` re-embeds an existing build with it (clusters/edges kept). Queries must use the same backend as the build: run a gemma build with `DIP_EMBED_BACKEND=st` or `remote`. `DIP_EMBED_BACKEND=remote` (+ `DIP_EMBED_URL`) gives the same vectors from `embed_server/` running on the GPU machine (see its README; Ollama's embeddinggemma-2 is MLX/Apple-only); `config.EMBED_MULTIMODAL` is true for both `st` and `remote`. Comparisons: `experiments/compare_embeddings.py`, `experiments/image_probe.py`, `experiments/gemma_alignment.py` (how the nomic-built clusters/sections hold up in gemma space → `gemma_alignment.md`), `experiments/image_metaphor/run.py` (Images action; run per backend). `.env` now makes the gemma build the default (`DIP_EMBED_BACKEND=remote`, `DIP_DATA_DIR=data-embeddinggemma2`) and pins the image pages to `DIP_IMAGE_WIKI_DIR=data-embeddinggemma2/wiki/images`, so every build shares that one set (it is not under `--reset`'s reach; move it along when archiving a build). Use the archived nomic build with `DIP_DATA_DIR=data-archive-2026-10-21/data DIP_EMBED_BACKEND=ollama`.

Offline demo: `DIP_OFFLINE=1` makes every LLM/embedding call raise `LLMError` at once (`llm._check_online`). Search falls back to BM25 (`retrieve.hybrid` with `vec=None`), Counter/Evidence return stored links with `fallback_error`, Images uses cached concepts + `concept_vectors` (filled by `--concepts`), Question returns 503. Image file/page paths stored by the build are looked up by name under `DIP_IMAGE_DIR` / `DIP_IMAGE_WIKI_DIR` when missing. The wiki editor (`/ui/editor.html`, `/wiki-files` endpoints in `api.py`) only edits pages listed in `wiki_pages` / `images`; `DIP_EDIT_TOKEN` guards saving. Stage 6 deletes unlocked pages of clusters/themes that no longer exist (`remove_stale_pages`); locked ones stay on disk, unindexed, and are named in the log.

## Architecture

Ingest stages (`dip/ingest/s1..s7`, orchestrated by `run.py`):
1. parse + chunk (`parse.py`, `chunk.py`): document › section › chunk
2. embed (nomic-embed-text) + FTS5 keyword index
3. annotate with the LLM: a context prefix per chunk (contextual retrieval: 1–2 sentences situating it in its document, stored in `nodes.context`, prepended for vectors and FTS only; `DIP_CHUNK_CONTEXT=0` turns it off; changing a prefix drops that chunk's vector so it is re-embedded; because the prefix names the document, it pulls each vector toward its own source: only 5% of 10-nearest-neighbour pairs cross sources (18% for the same texts without prefix), and clustering on prefixed vectors made 25 of 28 clusters single-source. So each chunk also gets a prefix-free `chunk_text` vector; stage 4 clusters on it and stage 5 takes counter candidates (10 nearest) from it. Search, Neighbours and the other query moves keep the prefixed `node` vectors), 1–3 claims per chunk, role, abstraction score (probability-weighted average over the digits 1–5, read from the model's logprobs), section/document summaries
4. k-means: chunks → clusters (k≈√n) → themes (k≈√clusters), LLM-named
5. claim graph: nearest-claim pairs → NLI proposes → LLM verifies → `SUPPORTS` / `ELABORATES` / `EXAMPLE_OF`; counter links (`CONTRADICTS`) come from a separate graded check: each passage and its 10 nearest passages rated 1–5 by qwen3:14b in both orders (`COUNTER_*` settings, `DIP_COUNTER_JUDGE=legacy` for the old claim-level links)
6. wiki: one markdown page per cluster and theme in `data/wiki/`
7. images: one page per file in `00_Data/images/` at `data/wiki/images/<slug>.md` (credit front matter from `<file>.source.txt`, Description, Subjects, Metaphors `- keyword: gloss`). Pages are the source of truth; missing ones are drafted (VLM, else the .source.txt description → LLM metaphors). Kept out of `nodes`. `--reset` keeps `wiki/images/`.

Stages 1–3 run only on new or changed sources. Stages 4–6 rebuild across the whole corpus every run. Stage 7 only drafts/embeds new or changed images. The `nli_cache`, `pair_cache` (claim text) and `counter_cache` (passage text + model) tables keep reruns cheap. Don't drop them casually: a full stage 5 costs hours of LLM calls (the first graded counter run alone is ≈11,000 judgements, ≈3 h with qwen3:14b).

Four index layers: L1 vectors + FTS, L2 hierarchy (`nodes.parent_id` = source tree, `nodes.cluster_id` = semantic tree), L3 claims + edges, L4 wiki (`wiki_pages` + files). Query side: `dip/query/retrieve.py` (hybrid RRF + MMR) and `dip/query/actions.py` (the seven moves; each result carries a `why`).

Terminology: "chunk" in code = "passage" in docs/UI. `level` (0/1/2) is tree position; `abstraction` (1–5) is content generality, a separate thing.

Edge semantics: rows read "src TYPE dst". `CONTRADICTS` is stored in both directions. Counter follows outgoing `CONTRADICTS`. Evidence follows incoming `SUPPORTS` and `EXAMPLE_OF`. Zoom in follows incoming `ELABORATES` and `EXAMPLE_OF`; Zoom out follows them outward. `method` is `nli+llm`, `llm`, `graded` (passage-level counter links: no claims attached, weight = graded score) or `wiki_lint`. Stage 5 deletes and rebuilds all edges except `wiki_lint` ones.

Wiki pages with `locked: true` in their front matter are not rewritten.

Images action: subject route (passage vector vs `image_desc` + `image_visual`, rank-fused) interleaved with metaphor route (LLM concepts of the passage, cached in `concept_cache`, vs `image_metaphor` lines; falls back to cluster/theme titles + claims). Every score is hubness-corrected (cosine minus the image vector's mean cosine to all passages); without it one generic image tops every list. Results carry `match`, `metaphor`, `concept`, `credit`/`source`/`license`; always show the credit. Reverse move `image_passages` (`GET /image-passages/{slug}`, "Passages for this image" in the UI's Images tab): subject = image description + picture vectors vs passages; metaphor = metaphor lines vs claim vectors (mapped to their passage). Hub correction is applied from the passage side at strength `REVERSE_CENTRE` (1.0 subject, 0.25 metaphor: full correction halved metaphor accuracy, none let junk passages like citations top many images).

## Counter-link research (2026-10-07)

- The old claim-level `CONTRADICTS` links (NLI + qwen3:8b) ranked real counters no better than chance (AUC 0.51) on `experiments/testset/`, an 80-pair, all-source test set labelled under the rule "a view counts whoever holds it, including one an author describes in order to argue against it". The graded whole-passage check now in stage 5 scored AUC 0.93. Re-score any change with `experiments/testset/run.py`.
- History and findings: `experiments/stance/` (issue/stance extraction, judge comparison), `experiments/positions/` (positions with voice/holder, `meta` role, hybrid candidates). Each has a README.
- Open: candidate recall (opposing passages worded differently aren't nearest neighbours); a `meta` role for credits/contents/methodology (prototyped, not in stage 3); only ~1 in 8 nearby pairs is counter-worthy, and few counters cross sources; all labels so far are Claude's.
- About 15% of claims are about the text ("The passage…"); NLI still runs for the other link types and is the slowest CPU step.
