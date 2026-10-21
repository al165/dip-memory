"""Build a method-neutral, corpus-representative test set of passage pairs for Counter.

Pool: every passage paired with its K nearest passages by embedding (the neighbourhood any
Counter method realistically searches), excluding navigation/link passages. Nothing about any
judge's verdicts is used, so no method is favoured.

Sample: N_CROSS cross-source + N_WITHIN within-source pairs, drawn at random under two caps so no
text dominates: each passage appears at most once, and each source supplies at most MAX_SHARE of
all pair ends. Strata sizes are recorded so eval scripts can re-weight to the whole pool.

    python build.py            -> gold.json (labels empty) + todo.txt (labelling sheet, no verdicts)
"""
import json, random, sqlite3, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from dip import db as dipdb

K, N_CROSS, N_WITHIN, MAX_SHARE, SEED = 10, 40, 40, 0.15, 2026

conn = sqlite3.connect("file:" + str(HERE.parent.parent / "data" / "dip.sqlite") + "?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
chunks = {r["id"]: r for r in conn.execute(
    "SELECT n.id, n.item_id, n.role FROM nodes n WHERE n.kind='chunk' AND coalesce(n.role,'') != 'link'")}
idx = dipdb.VectorIndex("node").refresh(conn)
allowed = set(chunks)

pool = {}
for nid in chunks:
    vec = idx.get(nid)
    if vec is None:
        continue
    for other, sim in idx.search(vec, k=K, allowed=allowed, exclude={nid}):
        key = tuple(sorted((nid, other)))
        pool[key] = max(pool.get(key, 0.0), sim)
strata = {"cross": [k for k in pool if chunks[k[0]]["item_id"] != chunks[k[1]]["item_id"]],
          "within": [k for k in pool if chunks[k[0]]["item_id"] == chunks[k[1]]["item_id"]]}
print(f"{len(chunks)} passages, pool {len(pool)} pairs: {len(strata['cross'])} cross-source, {len(strata['within'])} within")

rng = random.Random(SEED)
total_ends = 2 * (N_CROSS + N_WITHIN)
used, per_source, sample = set(), Counter(), []
for name, n in (("cross", N_CROSS), ("within", N_WITHIN)):
    cands = strata[name][:]
    rng.shuffle(cands)
    got = 0
    for a, b in cands:
        if got == n:
            break
        ia, ib = chunks[a]["item_id"], chunks[b]["item_id"]
        if a in used or b in used:
            continue
        extra = Counter([ia, ib])
        if any(per_source[i] + c > MAX_SHARE * total_ends for i, c in extra.items()):
            continue
        used |= {a, b}
        per_source.update(extra)
        sample.append({"pair": [a, b], "stratum": name, "similarity": round(pool[(a, b)], 3)})
        got += 1
rng.shuffle(sample)

titles = {r["id"]: r["title"] for r in conn.execute("SELECT id, title FROM items")}
gold = {
    "rule": "Counter-worthy if one passage's main content is a position (argued, reported, or described in "
            "order to reject it) that the other passage argues against. Two passages that both reject the "
            "same view are compatible.",
    "labels": {"O": "opposed", "Q": "qualifies", "C": "compatible", "D": "different_question", "S": "supports"},
    "counter_worthy": ["O", "Q"],
    "pool": {"k_nearest": K, "excluded_roles": ["link"]},
    "strata": {s: {"n_total": len(strata[s]), "n_sampled": sum(x["stratum"] == s for x in sample)} for s in strata},
    "items": [{**x, "claude": None, "user": None} for x in sample],
}
(HERE / "gold.json").write_text(json.dumps(gold, indent=1, ensure_ascii=False))

def show(nid):
    r = conn.execute("""SELECT n.text, s.title sec, i.title doc FROM nodes n JOIN nodes s ON s.id=n.parent_id
                        JOIN items i ON i.id=n.item_id WHERE n.id=?""", (nid,)).fetchone()
    return f"[{r['doc'][:50]} › {r['sec']}]\n{r['text'].strip()}"
with open(HERE / "todo.txt", "w") as f:
    for i, x in enumerate(sample):
        a, b = x["pair"]
        f.write(f"### {i:02d}  {a}-{b}\nA {show(a)}\n\nB {show(b)}\n\n")

print("sampled:", Counter(x["stratum"] for x in sample))
print("pair ends per source:", {titles[i][:30]: c for i, c in per_source.most_common()})
