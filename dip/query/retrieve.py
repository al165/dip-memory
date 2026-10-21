"""L1 retrieval primitives: dense search, BM25 keyword search, reciprocal-rank fusion, MMR."""
import re

import numpy as np

from dip import config, db

# One cached matrix per vector type, shared across requests; reloaded when the DB changes.
INDEXES = {name: db.VectorIndex(name)
           for name in ("node", "claim", "wiki", "image_desc", "image_visual", "image_metaphor")}


def index(conn, name: str) -> db.VectorIndex:
    return INDEXES[name].refresh(conn)


def _fts_query(text: str, max_terms: int = 32) -> str:
    terms = []
    for t in re.findall(r"\w{3,}", text.lower()):
        if t not in terms:
            terms.append(t)
    return " OR ".join(f'"{t}"' for t in terms[:max_terms])


def bm25(conn, text: str, k: int = 30, allowed: set | None = None) -> list[tuple[int, float]]:
    q = _fts_query(text)
    if not q:
        return []
    rows = conn.execute(
        "SELECT rowid, bm25(fts_nodes) AS s FROM fts_nodes WHERE fts_nodes MATCH ? ORDER BY s LIMIT ?",
        (q, k * 4),
    ).fetchall()
    out = [(r["rowid"], -r["s"]) for r in rows if allowed is None or r["rowid"] in allowed]
    return out[:k]


def rrf(*ranked_lists, k: int = None) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank). Robust way to merge rankings whose
    raw scores are on different scales (cosine vs BM25)."""
    k = k or config.RRF_K
    scores: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, (i, _) in enumerate(lst):
            scores[i] = scores.get(i, 0.0) + 1.0 / (k + rank + 1)
    return scores


def hybrid(conn, text: str, vec, k: int = 20, allowed: set | None = None,
           exclude: set | None = None) -> list[tuple[int, float]]:
    exclude = exclude or set()
    dense = index(conn, "node").search(vec, k=k * 2, allowed=allowed, exclude=exclude)
    kw = [(i, s) for i, s in bm25(conn, text, k=k * 2, allowed=allowed) if i not in exclude]
    fused = rrf(dense, kw)
    return sorted(fused.items(), key=lambda x: -x[1])[:k]


def mmr(conn, query_vec, candidate_ids: list[int], k: int, lam: float = None) -> list[tuple[int, float]]:
    """Maximal Marginal Relevance: pick items that are relevant to the query but not redundant
    with what's already picked. Returns [(id, relevance)]."""
    lam = config.MMR_LAMBDA if lam is None else lam
    idx = index(conn, "node")
    cands = [i for i in candidate_ids if idx.get(i) is not None]
    if not cands:
        return []
    q = np.asarray(query_vec, dtype=np.float32)
    q = q / max(np.linalg.norm(q), 1e-9)
    M = np.stack([idx.get(i) for i in cands])
    rel = M @ q
    picked: list[int] = []
    while len(picked) < min(k, len(cands)):
        if picked:
            red = (M @ M[picked].T).max(axis=1)
        else:
            red = np.zeros(len(cands))
        score = lam * rel - (1 - lam) * red
        score[picked] = -np.inf
        picked.append(int(np.argmax(score)))
    return [(cands[p], float(rel[p])) for p in picked]


def cosine(conn, a: int, b: int, owner: str = "node") -> float:
    idx = index(conn, owner)
    va, vb = idx.get(a), idx.get(b)
    return float(va @ vb) if va is not None and vb is not None else 0.0
