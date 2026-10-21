"""Compare current CONTRADICTS links with issue/stance counter pairs on the same six sources."""
import json, random, sqlite3
from pathlib import Path
HERE = Path(__file__).parent
ITEMS = (2, 3, 4, 6, 7, 8)
c = sqlite3.connect("file:" + str(HERE.parent.parent / "data" / "dip.sqlite") + "?mode=ro", uri=True); c.row_factory = sqlite3.Row
q = ",".join(map(str, ITEMS))
old = c.execute(f"""SELECT e.src_node a, e.dst_node b, na.item_id ia, nb.item_id ib, ca.text claim_a, cb.text claim_b, e.weight
  FROM edges e JOIN nodes na ON na.id=e.src_node JOIN nodes nb ON nb.id=e.dst_node
  LEFT JOIN claims ca ON ca.id=e.src_claim LEFT JOIN claims cb ON cb.id=e.dst_claim
  WHERE e.type='CONTRADICTS' AND e.method!='wiki_lint' AND e.src_node<e.dst_node
  AND na.item_id IN ({q}) AND nb.item_id IN ({q})""").fetchall()
new = json.loads((HERE / "pairs.json").read_text())
newp = {}
for p in new:
    k = tuple(sorted((p["yes"]["node"], p["no"]["node"])))
    newp.setdefault(k, p)
oldp = {(r["a"], r["b"]): r for r in old}
def stats(name, d, cross):
    n = len(d); x = sum(1 for v in d.values() if cross(v))
    print(f"{name:8} passage pairs {n:4}   cross-source {x:4} ({x/max(n,1):.0%})")
stats("current", oldp, lambda r: r["ia"] != r["ib"])
stats("stance", newp, lambda p: p["cross_source"])
print("overlap (same passage pair in both):", len(set(oldp) & set(newp)))

title = {r["id"]: r["title"] for r in c.execute("SELECT id, title FROM items")}
print("\n## Anchors")
for a in (839, 830, 361, 358, 501, 621, 589):
    o = sorted({y for x, y in oldp if x == a} | {x for x, y in oldp if y == a})
    n = sorted({y for x, y in newp if x == a} | {x for x, y in newp if y == a})
    print(f"{a}: current {o}\n     stance  {n}")
    for k, p in newp.items():
        if a in k:
            me, other = (p["yes"], p["no"]) if p["yes"]["node"] == a else (p["no"], p["yes"])
            print(f"       vs {other['node']} [{title[other['item']][:20]}] Q: {p['question']}\n          {me['canon']}: {me['claim']}\n          {other['canon']}: {other['claim']}")

# blind sample for judging: 25 of each, method hidden
random.seed(7)
so = random.sample(list(oldp.values()), min(25, len(oldp)))
sn = random.sample(list(newp.values()), min(25, len(newp)))
rows = [("current", r["claim_a"], r["claim_b"], r["ia"] != r["ib"]) for r in so] + \
       [("stance", p["yes"]["claim"], p["no"]["claim"], p["cross_source"]) for p in sn]
random.shuffle(rows)
(HERE / "sample.txt").write_text("\n".join(f"{i:02d}. A: {a}\n    B: {b}" for i, (_, a, b, _) in enumerate(rows)))
(HERE / "sample_key.json").write_text(json.dumps([m for m, *_ in rows]))
