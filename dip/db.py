"""SQLite storage: one file holds items, the node tree, claims, typed edges, vectors and an
FTS5 keyword index.

Vectors are stored as float32 BLOBs and searched with numpy (brute-force cosine). That is
instant for tens of thousands of rows and avoids SQLite extension loading, which many
Python builds (pyenv, macOS system Python) do not support. Swap in sqlite-vec / LanceDB
if the corpus grows past ~100k chunks.
"""
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager

import numpy as np

from dip import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id          INTEGER PRIMARY KEY,
    path        TEXT UNIQUE NOT NULL,
    type        TEXT NOT NULL,            -- text | pdf | html | web | image
    title       TEXT,
    sha256      TEXT NOT NULL,
    ingested_at TEXT DEFAULT (datetime('now'))
);

-- One table for both trees:
--   structural: document -> section -> chunk   (parent_id)
--   semantic:   theme -> cluster -> chunk       (cluster nodes use parent_id = theme;
--                                                chunks point at their cluster via cluster_id)
CREATE TABLE IF NOT EXISTS nodes (
    id          INTEGER PRIMARY KEY,
    item_id     INTEGER REFERENCES items(id) ON DELETE CASCADE,  -- NULL for cluster/theme
    parent_id   INTEGER REFERENCES nodes(id) ON DELETE CASCADE,
    cluster_id  INTEGER REFERENCES nodes(id) ON DELETE SET NULL,
    kind        TEXT NOT NULL,            -- document | section | chunk | cluster | theme
    level       INTEGER NOT NULL,         -- 0 chunk, 1 section/cluster, 2 document/theme
    title       TEXT,
    text        TEXT NOT NULL,
    abstraction REAL,                     -- 1 concrete .. 5 general principle
    role        TEXT,                     -- claim|evidence|example|definition|provocation|question|caption|link
    topics_json TEXT,
    ord         INTEGER DEFAULT 0,
    context     TEXT                      -- chunks: LLM sentence(s) situating it in its document
                                          -- (contextual retrieval); NULL = not written yet
);
CREATE INDEX IF NOT EXISTS nodes_parent ON nodes(parent_id);
CREATE INDEX IF NOT EXISTS nodes_cluster ON nodes(cluster_id);
CREATE INDEX IF NOT EXISTS nodes_kind ON nodes(kind);

CREATE TABLE IF NOT EXISTS claims (
    id      INTEGER PRIMARY KEY,
    node_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    text    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS claims_node ON claims(node_id);

-- Directed, typed relations between chunks (derived from their claims).
--   SUPPORTS    src gives support/evidence for dst
--   CONTRADICTS src challenges dst (stored in both directions)
--   ELABORATES  src is a more detailed treatment of dst
--   EXAMPLE_OF  src is a concrete example of dst
CREATE TABLE IF NOT EXISTS edges (
    src_node  INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    dst_node  INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    src_claim INTEGER REFERENCES claims(id) ON DELETE CASCADE,
    dst_claim INTEGER REFERENCES claims(id) ON DELETE CASCADE,
    type      TEXT NOT NULL,
    weight    REAL NOT NULL,
    method    TEXT NOT NULL,              -- nli | llm | wiki_lint
    PRIMARY KEY (src_node, dst_node, type)
);
CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst_node, type);

-- Embeddings. owner = 'node' | 'claim' | 'wiki'; owner_id = that table's id.
-- 'chunk_text' (owner_id = nodes.id): a chunk's text without its context prefix, for clustering.
CREATE TABLE IF NOT EXISTS vectors (
    owner    TEXT NOT NULL,
    owner_id INTEGER NOT NULL,
    vec      BLOB NOT NULL,
    PRIMARY KEY (owner, owner_id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_nodes USING fts5(title, text);  -- rowid = nodes.id

CREATE TABLE IF NOT EXISTS wiki_pages (
    id           INTEGER PRIMARY KEY,
    slug         TEXT UNIQUE NOT NULL,
    title        TEXT NOT NULL,
    summary      TEXT,
    path         TEXT NOT NULL,
    source_node  INTEGER,                 -- the cluster/theme node it was written from
    node_ids_json TEXT NOT NULL           -- chunk ids cited on the page
);

-- Cache of LLM verdicts on claim pairs, keyed by text, so re-running stage 5 is cheap.
CREATE TABLE IF NOT EXISTS pair_cache (
    a_text  TEXT NOT NULL,
    b_text  TEXT NOT NULL,
    verdict TEXT NOT NULL,
    PRIMARY KEY (a_text, b_text)
);

-- Cache of NLI cross-encoder scores, keyed by text + model, so stage 5 only scores new pairs.
CREATE TABLE IF NOT EXISTS nli_cache (
    premise       TEXT NOT NULL,
    hypothesis    TEXT NOT NULL,
    model         TEXT NOT NULL,
    entailment    REAL NOT NULL,
    contradiction REAL NOT NULL,
    neutral       REAL NOT NULL,
    PRIMARY KEY (premise, hypothesis, model)
);

-- Cache of graded counter judgements on passage pairs, per order (how B relates to A), keyed
-- by text + model: the 1-5 digit probabilities, so stage 5 only judges new passage pairs.
CREATE TABLE IF NOT EXISTS counter_cache (
    a_text     TEXT NOT NULL,
    b_text     TEXT NOT NULL,
    model      TEXT NOT NULL,
    probs_json TEXT NOT NULL,
    PRIMARY KEY (a_text, b_text, model)
);

-- Image library (stage 7). Rebuilt from the image wiki pages, which are the source of truth.
-- Vectors: 'image_desc' (title + description + subjects), 'image_visual' (the picture itself,
-- st backend only), 'image_metaphor' (one per metaphor line, owner_id = image_metaphors.id).
CREATE TABLE IF NOT EXISTS images (
    id            INTEGER PRIMARY KEY,
    slug          TEXT UNIQUE NOT NULL,
    path          TEXT NOT NULL,          -- the image file
    page          TEXT NOT NULL,          -- its wiki page
    title         TEXT NOT NULL,
    description   TEXT,
    subjects_json TEXT,
    source        TEXT,                   -- credit: where it came from (URL)
    creator       TEXT,
    license       TEXT,
    credit        TEXT,                   -- the full credit line to show with the image
    file_sha      TEXT NOT NULL,
    page_sha      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS image_metaphors (
    id       INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    keyword  TEXT NOT NULL,
    gloss    TEXT
);
CREATE INDEX IF NOT EXISTS image_metaphors_image ON image_metaphors(image_id);

-- Abstract concepts the LLM drew from a passage (for metaphor search), keyed by text + model.
CREATE TABLE IF NOT EXISTS concept_cache (
    text          TEXT NOT NULL,
    model         TEXT NOT NULL,
    concepts_json TEXT NOT NULL,
    PRIMARY KEY (text, model)
);

-- Query vectors of those concepts, keyed by concept text + embedding model, so the Images action
-- also runs where no embedding model is available (DIP_OFFLINE demo).
CREATE TABLE IF NOT EXISTS concept_vectors (
    text  TEXT NOT NULL,
    model TEXT NOT NULL,
    vec   BLOB NOT NULL,
    PRIMARY KEY (text, model)
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def connect(path=None) -> sqlite3.Connection:
    path = path or config.DB_PATH
    if str(path) != ":memory:":
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    if "context" not in {r["name"] for r in conn.execute("PRAGMA table_info(nodes)")}:
        conn.execute("ALTER TABLE nodes ADD COLUMN context TEXT")  # DBs built before contextual retrieval
    return conn


@contextmanager
def tx(conn):
    try:
        yield conn
        conn.commit()
        bump_version(conn)
    except Exception:
        conn.rollback()
        raise


def bump_version(conn):
    """Mark the DB as changed. A random id (not a counter) so a reset DB never reuses an old
    version that a running server has cached."""
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('version', ?)", (uuid.uuid4().hex,))
    conn.commit()


def get_version(conn) -> str:
    row = conn.execute("SELECT value FROM meta WHERE key='version'").fetchone()
    return row["value"] if row else ""


# --- vectors -------------------------------------------------------------------

def to_blob(vec) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def put_vectors(conn, owner: str, ids, vecs):
    conn.executemany(
        "INSERT OR REPLACE INTO vectors(owner, owner_id, vec) VALUES (?,?,?)",
        [(owner, int(i), to_blob(v)) for i, v in zip(ids, vecs)],
    )


def delete_orphan_vectors(conn):
    conn.execute("DELETE FROM vectors WHERE owner IN ('node','chunk_text') AND owner_id NOT IN (SELECT id FROM nodes)")
    conn.execute("DELETE FROM vectors WHERE owner='claim' AND owner_id NOT IN (SELECT id FROM claims)")
    conn.execute("DELETE FROM vectors WHERE owner='wiki' AND owner_id NOT IN (SELECT id FROM wiki_pages)")
    conn.execute("DELETE FROM vectors WHERE owner IN ('image_desc','image_visual') "
                 "AND owner_id NOT IN (SELECT id FROM images)")
    conn.execute("DELETE FROM vectors WHERE owner='image_metaphor' AND owner_id NOT IN (SELECT id FROM image_metaphors)")


class VectorIndex:
    """In-memory, L2-normalised matrix of all vectors for one owner type.
    Reloaded automatically when the DB version changes (i.e. after any ingest)."""

    def __init__(self, owner: str):
        self.owner = owner
        self.version = None
        self.ids = np.zeros(0, dtype=np.int64)
        self.mat = np.zeros((0, config.EMBED_DIM), dtype=np.float32)
        self.pos = {}
        self._lock = threading.Lock()

    def refresh(self, conn):
        v = get_version(conn)
        if v == self.version:
            return self
        with self._lock:
            rows = conn.execute(
                "SELECT owner_id, vec FROM vectors WHERE owner=? ORDER BY owner_id", (self.owner,)
            ).fetchall()
            if rows:
                self.ids = np.array([r["owner_id"] for r in rows], dtype=np.int64)
                mat = np.stack([from_blob(r["vec"]) for r in rows])
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                self.mat = mat / np.clip(norms, 1e-9, None)
            else:
                self.ids = np.zeros(0, dtype=np.int64)
                self.mat = np.zeros((0, config.EMBED_DIM), dtype=np.float32)
            self.pos = {int(i): p for p, i in enumerate(self.ids)}
            self.version = v
        return self

    def get(self, owner_id: int):
        p = self.pos.get(int(owner_id))
        return None if p is None else self.mat[p]

    def search(self, query_vec, k=10, allowed: set | None = None, exclude: set | None = None):
        """Return [(owner_id, cosine)] best first."""
        if len(self.ids) == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / max(np.linalg.norm(q), 1e-9)
        sims = self.mat @ q
        order = np.argsort(-sims)
        out = []
        for p in order:
            i = int(self.ids[p])
            if allowed is not None and i not in allowed:
                continue
            if exclude and i in exclude:
                continue
            out.append((i, float(sims[p])))
            if len(out) >= k:
                break
        return out


# --- small helpers ---------------------------------------------------------------

def node(conn, node_id: int):
    return conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()


def node_dict(row) -> dict:
    d = dict(row)
    d["topics"] = json.loads(d.pop("topics_json") or "[]")
    return d


def ids_of_kind(conn, *kinds) -> set:
    q = f"SELECT id FROM nodes WHERE kind IN ({','.join('?' * len(kinds))})"
    return {r["id"] for r in conn.execute(q, kinds)}
