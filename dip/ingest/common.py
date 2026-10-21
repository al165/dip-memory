"""Shared ingest helpers."""
import sys
import time
from pathlib import Path

import numpy as np

from dip import config, db, llm


# One abstraction scale for every node kind (chunks, sections, documents, clusters, themes), so
# scores are comparable across kinds. Used verbatim in all prompts that rate abstraction.
ABSTRACTION_SCALE = """abstraction: 1-5, the scope of what THIS text itself asserts, not how big its topic is.
  1 = a single fact, date, name, number or definition (a definition of a big idea like AGI is still 1)
  2 = one specific case, project, person, event or story
  3 = an interpretation or point about one particular topic
  4 = a general argument that applies across many cases
  5 = a sweeping thesis about technology, society or the world"""


def abstraction_score(out: dict, logprobs: list) -> float:
    """Continuous 1-5 score from the model's digit probabilities; falls back to the sampled value."""
    score = llm.expected_score(logprobs, "abstraction")
    return round(score if score is not None else float(out["abstraction"]), 2)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def embed_text_for(row) -> str:
    """What we embed for a node: chunks carry their context prefix (contextual retrieval), other
    nodes their title."""
    if row["kind"] == "chunk":
        return f"{row['context']}\n\n{row['text']}" if row["context"] else row["text"]
    title = row["title"] or ""
    return f"{title}\n{row['text']}" if title else row["text"]


def embed_missing_nodes(conn) -> int:
    rows = conn.execute(
        "SELECT n.* FROM nodes n LEFT JOIN vectors v ON v.owner='node' AND v.owner_id=n.id "
        "WHERE v.owner_id IS NULL"
    ).fetchall()
    if not rows:
        return 0
    vecs = llm.embed([embed_text_for(r) for r in rows])
    if config.EMBED_MULTIMODAL:
        vecs = blend_image_vectors(conn, rows, vecs)
    with db.tx(conn):
        db.put_vectors(conn, "node", [r["id"] for r in rows], vecs)
    return len(rows)


def blend_image_vectors(conn, rows, vecs):
    """Nodes of image items: average the description's vector with the image's own, so search
    also matches what the picture shows, not only what the VLM wrote about it."""
    paths = dict(conn.execute("SELECT id, path FROM items WHERE type='image'").fetchall())
    hits = [(i, paths[r["item_id"]]) for i, r in enumerate(rows)
            if r["item_id"] in paths and Path(paths[r["item_id"]]).exists()]
    if not hits:
        return vecs
    uniq = sorted({p for _, p in hits})
    by_path = dict(zip(uniq, llm.embed_images(uniq)))
    vecs = [list(v) for v in vecs]
    for i, p in hits:
        v = np.asarray(vecs[i]) + np.asarray(by_path[p])
        vecs[i] = (v / max(np.linalg.norm(v), 1e-9)).tolist()
    return vecs


def embed_missing_claims(conn) -> int:
    rows = conn.execute(
        "SELECT c.* FROM claims c LEFT JOIN vectors v ON v.owner='claim' AND v.owner_id=c.id "
        "WHERE v.owner_id IS NULL"
    ).fetchall()
    if not rows:
        return 0
    vecs = llm.embed([r["text"] for r in rows])
    with db.tx(conn):
        db.put_vectors(conn, "claim", [r["id"] for r in rows], vecs)
    return len(rows)


def progress(i: int, n: int, label: str, every: int = 10):
    if i == n or i % every == 0:
        log(f"  {label}: {i}/{n}")


def rebuild_fts(conn):
    with db.tx(conn):
        conn.execute("DELETE FROM fts_nodes")
        conn.execute(
            "INSERT INTO fts_nodes(rowid, title, text) SELECT id, coalesce(title,''), "
            "CASE WHEN context <> '' THEN context || char(10) || text ELSE text END FROM nodes"
        )
