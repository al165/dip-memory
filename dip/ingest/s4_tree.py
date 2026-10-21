"""Stage 4 — semantic tree (RAPTOR-style): cluster chunks across *all* documents, have the
LLM summarise each cluster, then cluster the clusters into themes.

    theme (level 2)  <- parent_id -  cluster (level 1)  <- cluster_id -  chunk (level 0)

Unlike the structural tree (one document's sections), this groups related passages from
different sources, so zoom-out can lift a chunk to the corpus-wide idea it belongs to.
Global stage: rebuilt from scratch on every run.
"""
import math

import numpy as np
from sklearn.cluster import KMeans

from dip import config, db, llm
from dip.ingest.common import (ABSTRACTION_SCALE, abstraction_score, embed_missing_nodes, log, progress,
                               rebuild_fts)

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "abstraction": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["title", "summary", "abstraction"],
}

PROMPT = """These {n} {what} were grouped together because they discuss related ideas.

{body}

Return JSON with:
- title: a 2-6 word name for the shared idea (not generic like "AI discussion").
- summary: 2-4 sentences stating the shared argument or theme and any disagreement within it.
- """ + ABSTRACTION_SCALE + """
"""


def _kmeans(mat: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, len(mat)))
    if k == 1:
        return np.zeros(len(mat), dtype=int)
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(mat)


def _summarise(texts: list[str], what: str) -> dict:
    per = max(200, 6500 // max(1, len(texts)))
    body = "\n\n".join(f"- {t[:per]}" for t in texts)
    out, lp = llm.generate(PROMPT.format(n=len(texts), what=what, body=body), schema=SCHEMA,
                           system="You organise a research corpus into themes. Be specific.", logprobs=True)
    out["abstraction"] = abstraction_score(out, lp)
    return out


def run(conn):
    with db.tx(conn):
        conn.execute("UPDATE nodes SET cluster_id=NULL WHERE kind='chunk'")
        conn.execute("DELETE FROM nodes WHERE kind IN ('cluster','theme')")
        db.delete_orphan_vectors(conn)

    idx = db.VectorIndex("node").refresh(conn)
    chunk_ids = sorted(db.ids_of_kind(conn, "chunk"))
    chunk_ids = [i for i in chunk_ids if idx.get(i) is not None]
    if len(chunk_ids) < 4:
        log("stage 4: too few chunks to cluster, skipping")
        return
    mat = np.stack([idx.get(i) for i in chunk_ids])
    k = config.CLUSTER_COUNT or round(math.sqrt(len(chunk_ids)))
    labels = _kmeans(mat, k)
    text_of = {r["id"]: r["text"] for r in conn.execute("SELECT id, text FROM nodes WHERE kind='chunk'")}

    cluster_ids = []
    groups = [[chunk_ids[j] for j in np.where(labels == c)[0]] for c in sorted(set(labels))]
    for gi, members in enumerate(groups, 1):
        out = _summarise([text_of[m] for m in members], "passages")
        with db.tx(conn):
            cid = conn.execute(
                "INSERT INTO nodes(kind, level, title, text, abstraction) VALUES ('cluster',1,?,?,?)",
                (out["title"].strip(), out["summary"].strip(), float(out["abstraction"])),
            ).lastrowid
            conn.executemany("UPDATE nodes SET cluster_id=? WHERE id=?", [(cid, m) for m in members])
        cluster_ids.append(cid)
        progress(gi, len(groups), "clusters summarised", every=5)
    embed_missing_nodes(conn)

    idx.refresh(conn)
    cmat = np.stack([idx.get(c) for c in cluster_ids])
    k2 = config.THEME_COUNT or max(2, round(math.sqrt(len(cluster_ids))))
    labels2 = _kmeans(cmat, k2)
    rows = {r["id"]: r for r in conn.execute("SELECT id, title, text FROM nodes WHERE kind='cluster'")}
    for t in sorted(set(labels2)):
        members = [cluster_ids[j] for j in np.where(labels2 == t)[0]]
        out = _summarise([f"{rows[m]['title']}: {rows[m]['text']}" for m in members], "sub-themes")
        with db.tx(conn):
            tid = conn.execute(
                "INSERT INTO nodes(kind, level, title, text, abstraction) VALUES ('theme',2,?,?,?)",
                (out["title"].strip(), out["summary"].strip(), float(out["abstraction"])),
            ).lastrowid
            conn.executemany("UPDATE nodes SET parent_id=? WHERE id=?", [(tid, m) for m in members])
    embed_missing_nodes(conn)
    rebuild_fts(conn)
    log(f"stage 4: {len(cluster_ids)} clusters, {len(set(labels2))} themes")
