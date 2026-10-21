"""Central configuration. Every value can be overridden with an env var of the same name
(or a .env file in the project root)."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name, default, cast=str):
    val = os.getenv(name)
    return cast(val) if val is not None else default


# --- paths -----------------------------------------------------------------
DATA_DIR = Path(_env("DIP_DATA_DIR", str(ROOT / "data")))
DB_PATH = Path(_env("DIP_DB_PATH", str(DATA_DIR / "dip.sqlite")))
WIKI_DIR = Path(_env("DIP_WIKI_DIR", str(DATA_DIR / "wiki")))
SOURCES_DIR = Path(_env("DIP_SOURCES_DIR", str(DATA_DIR / "sources")))  # snapshots of fetched web articles
IMAGE_DIR = Path(_env("DIP_IMAGE_DIR", str(ROOT.parent.parent / "00_Data" / "images")))  # the image library
IMAGE_WIKI_DIR = Path(_env("DIP_IMAGE_WIKI_DIR", str(WIKI_DIR / "images")))  # one page per image (stage 7)

# --- models (all local) ----------------------------------------------------
# Offline demo: no LLM and no embedding model at query time. Calls fail at once instead of
# waiting for a timeout; actions fall back to stored vectors, links and cached concepts.
OFFLINE = _env("DIP_OFFLINE", False, lambda v: v.strip().lower() in ("1", "true", "yes"))
EDIT_TOKEN = _env("DIP_EDIT_TOKEN", "")  # if set, saving in the wiki editor needs this token
OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = _env("DIP_LLM_MODEL", "qwen3:4b")
VLM_MODEL = _env("DIP_VLM_MODEL", "qwen2.5vl:3b")  # only used for image items
EMBED_BACKEND = _env("DIP_EMBED_BACKEND", "ollama")  # ollama | st (local sentence-transformers) | remote
EMBED_MULTIMODAL = EMBED_BACKEND in ("st", "remote")  # gemma: images and text in one space
EMBED_MODEL = _env("DIP_EMBED_MODEL", "google/embeddinggemma-2" if EMBED_MULTIMODAL else "nomic-embed-text")
EMBED_URL = _env("DIP_EMBED_URL", "http://localhost:11500")  # remote backend: embed_server/ on the GPU machine
EMBED_TOKEN = _env("DIP_EMBED_TOKEN", "")  # remote backend: bearer token, if the server sets EMBED_TOKEN
EMBED_DIM = _env("DIP_EMBED_DIM", 768, int)  # st backend: below the model's native size truncates (Matryoshka)
EMBED_DEVICE = _env("DIP_EMBED_DEVICE", "cpu")  # st backend only; bf16 on CPU fits in ~1.3 GB RAM
NLI_MODEL = _env("DIP_NLI_MODEL", "cross-encoder/nli-deberta-v3-xsmall")
LLM_TIMEOUT = _env("DIP_LLM_TIMEOUT", 300.0, float)
LLM_NUM_CTX = _env(
    "DIP_LLM_NUM_CTX", 32768, int
)  # context window; 4096 lets qwen3:4b fit in 4 GB VRAM
LLM_MAX_TOKENS = _env(
    "DIP_LLM_MAX_TOKENS", 1500, int
)  # cap on generated tokens per call, so a model stuck repeating itself can't run to the timeout

# --- chunking ----------------------------------------------------------------
CHUNK_MIN_CHARS = _env("DIP_CHUNK_MIN", 300, int)
CHUNK_MAX_CHARS = _env("DIP_CHUNK_MAX", 800, int)
CHUNK_CONTEXT = _env("DIP_CHUNK_CONTEXT", 1, int)  # 1 = stage 3 writes a context prefix per chunk (0 = off)
CONTEXT_DOC_CHARS = _env(
    "DIP_CONTEXT_DOC_CHARS", max(4000, (LLM_NUM_CTX - 1500) * 3), int
)  # document text shown to the LLM per chunk; longer documents get an outline + the chunk's section

# --- retrieval -----------------------------------------------------------------
RRF_K = _env("DIP_RRF_K", 60, int)  # reciprocal rank fusion constant
MMR_LAMBDA = _env(
    "DIP_MMR_LAMBDA", 0.7, float
)  # 1.0 = pure relevance, 0.0 = pure diversity
DEFAULT_K = _env("DIP_DEFAULT_K", 5, int)
IMAGE_CONCEPTS = _env("DIP_IMAGE_CONCEPTS", 4, int)  # abstract concepts drawn from a passage for metaphor search

# --- claim graph -----------------------------------------------------------
EDGE_CANDIDATES = _env(
    "DIP_EDGE_CANDIDATES", 15, int
)  # claim neighbours checked per claim
NLI_THRESHOLD = _env(
    "DIP_NLI_THRESHOLD", 0.6, float
)  # NLI prob needed to *propose* an edge (the LLM then verifies it)
LLM_ESCALATE_SIM = _env(
    "DIP_LLM_ESCALATE_SIM", 0.75, float
)  # cosine above which pairs NLI didn't flag still go to the LLM
LLM_ESCALATE_MAX = _env(
    "DIP_LLM_ESCALATE_MAX", 150, int
)  # cap on those unflagged pairs
LLM_VERIFY_MAX = _env(
    "DIP_LLM_VERIFY_MAX", 2500, int
)  # cap on all LLM pair checks per run (strongest pairs first)
LLM_CONCURRENCY = _env(
    "DIP_LLM_CONCURRENCY", 4, int
)  # parallel LLM requests; match OLLAMA_NUM_PARALLEL on the Ollama host
FALLBACK_NLI_THRESHOLD = _env("DIP_FALLBACK_NLI", 0.5, float)

# --- counter links (CONTRADICTS) ---------------------------------------------------
COUNTER_JUDGE = _env(
    "DIP_COUNTER_JUDGE", "graded"
)  # graded = LLM rates whole passage pairs 1-5; legacy = NLI-proposed claim pairs the LLM verifies
COUNTER_MODEL = _env("DIP_COUNTER_MODEL", "qwen3:14b")  # judge model for graded counter links
COUNTER_NEIGHBOURS = _env(
    "DIP_COUNTER_NEIGHBOURS", 10, int
)  # nearest passages judged per passage; ~5,600 pairs for 760 passages at 10
COUNTER_THRESHOLD = _env(
    "DIP_COUNTER_THRESHOLD", 0.45, float
)  # mean over both orders of P(opposed) + P(qualifies) needed for a CONTRADICTS link. Often one
# order says "qualifies" and the other "compatible" (mean ≈0.48): 0.45 keeps those (test set:
# recall 0.93, precision 0.64), 0.5 drops them (recall 0.51)

# --- semantic tree -------------------------------------------------------------
THEME_COUNT = _env("DIP_THEME_COUNT", 0, int)  # 0 = auto (≈ sqrt(#clusters))
CLUSTER_COUNT = _env("DIP_CLUSTER_COUNT", 0, int)  # 0 = auto (≈ sqrt(#chunks))
