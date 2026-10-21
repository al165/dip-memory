"""Step E: the same strict check as verify.py, but shown the two full passages (with source and
section) instead of one-sentence claims. Same candidate pairs, so only the evidence given changes."""
import json, sqlite3, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dip import llm

HERE = Path(__file__).parent
SCHEMA = {"type": "object", "properties": {"relation": {"type": "string",
          "enum": ["opposed", "compatible", "different_question"]}, "question": {"type": "string"}},
          "required": ["relation", "question"]}
PROMPT = """Passage A ({da}, "{sa}"):
{a}

Passage B ({db}, "{sb}"):
{b}

Would a reader see A and B as two sides of a disagreement?
- opposed: they argue for conflicting answers to the same question; accepting one means pushing back on the other
- compatible: the same person could hold both (including when one describes a problem and the other criticises or responds to it)
- different_question: they are about different things, even if the topic is related
Also give `question`: the question they disagree on, or "" if not opposed."""
c = sqlite3.connect("file:" + str(HERE.parent.parent / "data" / "dip.sqlite") + "?mode=ro", uri=True)
c.row_factory = sqlite3.Row
def info(nid):
    r = c.execute("""SELECT n.text, s.title sec, i.title doc FROM nodes n JOIN nodes s ON s.id=n.parent_id
                     JOIN items i ON i.id=n.item_id WHERE n.id=?""", (nid,)).fetchone()
    return r["text"], r["sec"], r["doc"][:60]
pairs = {}
for r in json.loads((HERE / "verified.json").read_text()):
    pairs.setdefault(tuple(r["pair"]), r)
TEXT = {n: info(n) for k in pairs for n in k}  # read on the main thread; workers only call the LLM
def ask(k):
    (ta, sa, da), (tb, sb, db) = TEXT[k[0]], TEXT[k[1]]
    o = llm.generate(PROMPT.format(a=ta, sa=sa, da=da, b=tb, sb=sb, db=db), schema=SCHEMA,
                     system="You judge argumentative relations between passages. Be strict.")
    return k, o
t0 = time.time()
with ThreadPoolExecutor(4) as pool:
    res = [{**pairs[k], "passage_relation": o["relation"], "passage_question": o["question"]}
           for k, o in pool.map(ask, list(pairs))]
print(f"{len(res)} passage checks in {time.time()-t0:.0f}s")
(HERE / "verified_passages.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
for m in ("current", "stance", None):
    rs = [r for r in res if m is None or r["method"] == m]
    kept = [r for r in rs if r["passage_relation"] == "opposed"]
    print(f"{m or 'union':8} {len(rs)} pairs -> {len(kept)} opposed ({sum(r['cross'] for r in kept)} cross-source)")
