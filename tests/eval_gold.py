"""Sanity checks + a small gold-set evaluation against the *live* database.

    python -m tests.eval_gold                 # fast: graph-only (no LLM fallbacks)
    python -m tests.eval_gold --fallback      # also exercise HyDE fallbacks (slow, uses the LLM)

1. Retrieval recall: 00_Data/provocations.json maps 40 provocations to the passages they were
   drawn from. For each provocation we search the DB and check whether a chunk containing a
   source passage appears in the top 5 (recall@5) — a check on L1.
2. Relation sanity, over a sample of chunks:
   - counter results should be *closer* to the anchor than random chunks (same topic, opposite
     stance) — the property the old "farthest vector" approach lacked
   - zoom-out results should be at least as abstract as the anchor; zoom-in at most as abstract
   - edge coverage: how many chunks have any counter / evidence / zoom-in edge at all
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from dip import db
from dip.query import actions, retrieve

GOLD = Path(__file__).resolve().parents[3] / "00_Data" / "provocations.json"


def _norm(t: str) -> str:
    return " ".join(t.lower().split())


def recall_at_k(conn, k=5):
    if not GOLD.exists():
        return None
    chunks = {r["id"]: _norm(r["text"]) for r in conn.execute("SELECT id, text FROM nodes WHERE kind='chunk'")}
    hits = total = 0
    for prov in json.load(open(GOLD)):
        # a passage "matches" a chunk if its first 80 normalised chars occur in the chunk
        keys = [_norm(p["text"])[:80] for p in prov["passages"] if p.get("text")]
        gold = {cid for cid, t in chunks.items() if any(key and key in t for key in keys)}
        if not gold:
            continue
        found = {r["id"] for r in actions.search(conn, prov["text"], k=k)}
        hits += bool(found & gold)
        total += 1
    return hits / total if total else None, total


def relation_checks(conn, n=25, fallback=False, seed=0):
    rng = random.Random(seed)
    chunk_ids = sorted(db.ids_of_kind(conn, "chunk"))
    sample = rng.sample(chunk_ids, min(n, len(chunk_ids)))
    stats = {"counter_sim": [], "random_sim": [], "zoom_out_ok": [], "zoom_in_ok": [],
             "has_counter": 0, "has_evidence": 0, "has_zoom_in": 0,
             "fallback_errors": 0, "from_fallback": 0}
    for cid in sample:
        a = db.node(conn, cid)
        co = actions.counter(conn, node_id=cid, k=3, fallback=fallback)
        ev = actions.evidence(conn, node_id=cid, k=3, fallback=fallback)
        c, e = co["results"], ev["results"]
        stats["fallback_errors"] += sum(bool(o["meta"].get("fallback_error")) for o in (co, ev))
        stats["from_fallback"] += sum(r["why"].startswith(("counter:", "evidence:")) for r in c + e)
        zi = actions.zoom_in(conn, node_id=cid, k=3)["results"]
        zo = actions.zoom_out(conn, node_id=cid, k=5)["results"]
        stats["has_counter"] += bool(c)
        stats["has_evidence"] += bool(e)
        stats["has_zoom_in"] += bool(zi)
        stats["counter_sim"] += [retrieve.cosine(conn, cid, r["id"]) for r in c]
        stats["random_sim"] += [retrieve.cosine(conn, cid, r) for r in rng.sample(chunk_ids, 3) if r != cid]
        if a["abstraction"] is not None:
            stats["zoom_out_ok"] += [r["abstraction"] >= a["abstraction"]
                                     for r in zo if r["type"] == "node" and r["abstraction"] is not None]
            stats["zoom_in_ok"] += [r["abstraction"] <= a["abstraction"]
                                    for r in zi if r["abstraction"] is not None]
    return stats, len(sample)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fallback", action="store_true")
    ap.add_argument("-n", type=int, default=25)
    args = ap.parse_args()
    conn = db.connect()

    r = recall_at_k(conn)
    if r and r[0] is not None:
        print(f"L1 search recall@5 on provocations gold set: {r[0]:.0%} ({r[1]} provocations)")

    s, n = relation_checks(conn, args.n, args.fallback)
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")  # noqa: E731
    print(f"\nRelation checks over {n} random chunks (fallbacks {'on' if args.fallback else 'off'}):")
    print(f"  chunks with ≥1 counter:   {s['has_counter']}/{n}")
    print(f"  chunks with ≥1 evidence:  {s['has_evidence']}/{n}")
    print(f"  chunks with ≥1 zoom-in:   {s['has_zoom_in']}/{n}")
    print(f"  counter cos to anchor:    {mean(s['counter_sim']):.3f}  vs random chunk: {mean(s['random_sim']):.3f}"
          "  (counter should be higher: same topic)")
    print(f"  zoom-out at least as abstract: {mean(s['zoom_out_ok']):.0%}")
    print(f"  zoom-in at most as abstract:   {mean(s['zoom_in_ok']):.0%}")
    if args.fallback:
        print(f"  results from HyDE fallback:    {s['from_fallback']}")
        print(f"  fallback LLM errors:           {s['fallback_errors']}")


if __name__ == "__main__":
    main()
