"""Step D: strict pairwise check of candidate counter pairs, applied to BOTH candidate sets
(current CONTRADICTS links and stance pairs) so the verifier is tested independently of the candidates."""
import json, sqlite3, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dip import llm

HERE = Path(__file__).parent
SCHEMA = {"type": "object", "properties": {"relation": {"type": "string",
          "enum": ["opposed", "compatible", "different_question"]}}, "required": ["relation"]}
PROMPT = """{q}Statement A: {a}
Statement B: {b}

Would a reader see A and B as two sides of a disagreement?
- opposed: they give conflicting answers to the same question; accepting one means pushing back on the other
- compatible: both could be held by the same person at once (including when one describes a problem and the other criticises or responds to it)
- different_question: they are about different things, even if the topic is related"""

c = sqlite3.connect("file:" + str(HERE.parent.parent / "data" / "dip.sqlite") + "?mode=ro", uri=True)
c.row_factory = sqlite3.Row
ITEMS = "2,3,4,6,7,8"
cur = {}
for r in c.execute(f"""SELECT e.src_node a, e.dst_node b, na.item_id ia, nb.item_id ib, ca.text ta, cb.text tb FROM edges e
  JOIN nodes na ON na.id=e.src_node JOIN nodes nb ON nb.id=e.dst_node
  JOIN claims ca ON ca.id=e.src_claim JOIN claims cb ON cb.id=e.dst_claim
  WHERE e.type='CONTRADICTS' AND e.method!='wiki_lint' AND e.src_node<e.dst_node
  AND na.item_id IN ({ITEMS}) AND nb.item_id IN ({ITEMS})"""):
    cur[(r["a"], r["b"])] = {"method": "current", "a": r["ta"], "b": r["tb"], "q": None, "cross": r["ia"] != r["ib"]}
st = {}
for p in json.loads((HERE / "pairs.json").read_text()):
    k = tuple(sorted((p["yes"]["node"], p["no"]["node"])))
    st.setdefault(k, {"method": "stance", "a": p["yes"]["claim"], "b": p["no"]["claim"], "q": p["question"], "cross": p["cross_source"]})
jobs = [(k, v) for k, v in cur.items()] + [(k, v) for k, v in st.items()]
def ask(job):
    k, v = job
    q = f"Question: {v['q']}\n" if v["q"] else ""
    return k, v, llm.generate(PROMPT.format(q=q, a=v["a"], b=v["b"]), schema=SCHEMA,
                              system="You judge argumentative relations between statements. Be strict.")["relation"]
t0 = time.time()
with ThreadPoolExecutor(4) as pool:
    res = [{"pair": list(k), **v, "relation": rel} for k, v, rel in pool.map(ask, jobs)]
print(f"{len(jobs)} checks in {time.time()-t0:.0f}s")
(HERE / "verified.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
for m in ("current", "stance"):
    rs = [r for r in res if r["method"] == m]
    kept = [r for r in rs if r["relation"] == "opposed"]
    print(f"{m:8} {len(rs)} pairs -> {len(kept)} kept as opposed ({sum(r['cross'] for r in kept)} cross-source)")
