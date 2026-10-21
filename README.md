# dip-memory

A local-first document memory for exploring a corpus through six moves: **Neighbours, Counter,
Zoom-in, Zoom-out, Evidence and Question**. It has ingest scripts, a single-file SQLite
database, an LLM-maintained markdown wiki, and a FastAPI server. All models run locally through
Ollama and sentence-transformers; nothing calls a cloud API. (The only network access besides
your Ollama server is fetching the web articles you ask it to ingest.)

## Quick start

```bash
# uses the `dip` pyenv virtualenv (set by DIP/.python-version)
pip install -r requirements.txt
ollama serve &                       # needs the GPU build (e.g. ollama-cuda) for reasonable speed
ollama pull qwen3:4b && ollama pull nomic-embed-text

python -m dip.ingest.run "../../00_Data/Small AI – text only.md"   # ~30-40 min on a GTX 1650
python -m dip.ingest.run paper.pdf https://example.org/essay       # PDFs and web articles too
python -m dip.ingest.run ../../00_Data/reading-list.urls           # or a list of URLs
uvicorn dip.api:app --port 8000      # test UI: http://localhost:8000/   API docs: /docs
```

## API

| Endpoint                                                         |                                                                    |
| ---------------------------------------------------------------- | ------------------------------------------------------------------ |
| `GET /health`                                                    | counts + model status                                              |
| `GET /search?q=`                                                 | hybrid keyword + semantic search over chunks                       |
| `GET /neighbours?node_id=`                                       | similar passages (diverse)                                         |
| `GET /counter?node_id=`                                          | passages that disagree                                             |
| `GET /evidence?node_id=`                                         | passages that support it (evidence, examples first)                |
| `GET /zoom-in?node_id=`                                          | more specific: details, examples, children                         |
| `GET /zoom-out?node_id=`                                         | broader: cluster, theme, section, wiki page, more general passages |
| `POST /question` `{question, node_id?}`                          | answer with `[id]` citations, wiki-routed                          |
| `GET /nodes?kind=chunk`, `GET /nodes/{id}`                       | browse nodes, with claims and edges                                |
| `GET /items`, `GET /wiki`, `GET /wiki/{slug}`                    | sources and wiki pages                                             |
| `GET /wiki-files`, `GET`/`PUT /wiki-files/{topic\|image}/{name}` | wiki editor: list, read, save (`{text, base_hash, lock?}`)         |

The wiki editor is at `/ui/editor.html` (linked from the test UI).

## Offline demo (no LLM, no embedding model)

To share the explorer from a server without Ollama or the embedding server:

```bash
python -m dip.ingest.run --concepts   # first, on the GPU machine: Images concepts + their vectors
# copy data-embeddinggemma2/ and 00_Data/images/ to the server, then:
DIP_OFFLINE=1 DIP_EDIT_TOKEN=<secret> \
DIP_DATA_DIR=data-embeddinggemma2 DIP_IMAGE_WIKI_DIR=data-embeddinggemma2/wiki/images \
DIP_IMAGE_DIR=/path/to/images uvicorn dip.api:app --host 0.0.0.0 --port 8000
```

Browsing, Neighbours, Counter, Evidence, Zoom in/out, Images and "Passages for this image" run
on stored vectors, links and cached concepts. Search matches keywords only (BM25), Counter and
Evidence skip their LLM fallback (stored links only), and Question is unavailable. Wiki edits are
saved to the files; a topic page's summary and citations update the database at once, while the
page's vector (and image page edits) wait for the next ingest run on the GPU machine. Without
`DIP_EDIT_TOKEN` anyone who can open the site can save edits.

Every action also accepts `text=` instead of `node_id=` (mapped to the closest chunk) and `k=`.
Each result has a `why` field naming the index layer that produced it.

## Layout

```
dip/
  config.py      all settings (env-overridable)
  db.py          SQLite schema, vector storage + in-memory cosine index
  llm.py         Ollama client (chat w/ JSON schema, embeddings)
  nli.py         NLI cross-encoder (entail / contradict / neutral)
  parse.py       file loaders: md (markdown-it), txt, pdf (Docling), html (trafilatura), images
  chunk.py       section detection + chunking
  ingest/        run.py (CLI) + stages s1..s6
  query/         retrieve.py (hybrid, RRF, MMR) + actions.py (the six actions)
  api.py         FastAPI app
data/            dip.sqlite + wiki/ (generated; safe to delete with --reset)
tests/           pytest suite (no models needed) + eval_gold.py (live DB evaluation)
```

## Tests

```bash
python -m pytest tests          # unit tests: chunker + action contracts on a fixture DB
python -m tests.eval_gold       # recall on provocations.json + relation sanity checks (needs an ingested DB)
```
