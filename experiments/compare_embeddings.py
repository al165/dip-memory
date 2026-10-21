"""Gold-set retrieval quality for the build in DIP_DATA_DIR, with finer metrics than eval_gold:
recall@1/@5 and MRR, for hybrid search (dense+BM25, what the UI uses) and dense-only (the
embedding model alone). Run once per build and compare:

    python experiments/compare_embeddings.py
    DIP_EMBED_BACKEND=st DIP_DATA_DIR=data-embeddinggemma2 python experiments/compare_embeddings.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dip import config, db, llm  # noqa: E402
from dip.query import actions, retrieve  # noqa: E402
from tests.eval_gold import GOLD, _norm  # noqa: E402


def main():
    conn = db.connect()
    chunks = {r["id"]: _norm(r["text"]) for r in conn.execute("SELECT id, text FROM nodes WHERE kind='chunk'")}
    chunk_ids = set(chunks)
    stats = {m: {"r1": 0, "r5": 0, "rr": 0.0} for m in ("hybrid", "dense")}
    total, t_embed = 0, 0.0
    for prov in json.load(open(GOLD)):
        keys = [_norm(p["text"])[:80] for p in prov["passages"] if p.get("text")]
        gold = {cid for cid, t in chunks.items() if any(key and key in t for key in keys)}
        if not gold:
            continue
        total += 1
        t0 = time.time()
        vec = llm.embed_one(prov["text"], query=True)
        t_embed += time.time() - t0
        ranked = {
            "hybrid": [r["id"] for r in actions.search(conn, prov["text"], k=20)],
            "dense": [i for i, _ in retrieve.index(conn, "node").search(vec, k=20, allowed=chunk_ids)],
        }
        for m, ids in ranked.items():
            rank = next((p for p, i in enumerate(ids) if i in gold), None)
            stats[m]["r1"] += rank == 0
            stats[m]["r5"] += rank is not None and rank < 5
            stats[m]["rr"] += 1 / (rank + 1) if rank is not None else 0
    print(f"{config.EMBED_BACKEND}:{config.EMBED_MODEL} ({config.EMBED_DIM}d) on {config.DATA_DIR.name}, "
          f"{total} provocations, {t_embed / total * 1000:.0f} ms/query embed")
    for m, s in stats.items():
        print(f"  {m:6}  recall@1 {s['r1'] / total:4.0%}  recall@5 {s['r5'] / total:4.0%}  MRR {s['rr'] / total:.3f}")


if __name__ == "__main__":
    main()
