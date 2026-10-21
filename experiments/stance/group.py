"""Prototype steps B+C: group positions by issue similarity, then have the LLM write one shared
question per group and re-label every member's stance against it. Writes groups.json + pairs.json."""
import json, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from sklearn.cluster import AgglomerativeClustering
sys.path.insert(0, "/home/arran/Projects/AIxDESIGN/DIP/01_Initial_Tests/dip-memory")
from dip import llm

HERE = Path(__file__).parent
DIST = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25   # cosine distance threshold
MAXG = 16

pos = []
for nid, v in json.loads((HERE / "positions.json").read_text()).items():
    for p in v["positions"]:
        if len(p["claim"]) > 10 and len(p["issue"]) > 10:
            pos.append({"node": int(nid), "item": v["item"], **p})
emb_path = HERE / "issue_emb.npy"
if emb_path.exists() and len(np.load(emb_path)) == len(pos):
    X = np.load(emb_path)
else:
    X = np.array(llm.embed([p["issue"] for p in pos]), dtype=np.float32); np.save(emb_path, X)
X /= np.linalg.norm(X, axis=1, keepdims=True)
lab = AgglomerativeClustering(n_clusters=None, distance_threshold=DIST, metric="cosine",
                              linkage="average").fit_predict(X)
groups = {}
for i, g in enumerate(lab):
    groups.setdefault(int(g), []).append(i)
# a group can only yield a counter if it spans >=2 passages; split oversized groups
cand = []
for g, idx in groups.items():
    if len({pos[i]["node"] for i in idx}) < 2:
        continue
    for s in range(0, len(idx), MAXG):
        cand.append(idx[s:s + MAXG])
sizes = Counter(len(g) for g in cand)
print(f"{len(pos)} positions, {len(groups)} issue groups, {len(cand)} span >=2 passages; sizes {sorted(sizes.items())}", flush=True)

PROMPT = """These statements come from different texts. They were grouped because they seem to address a similar question.

{items}

1. question: write the single neutral yes/no question that most of these statements answer, so that some could answer yes and others no.
2. answers: for each statement, in order, give "yes" or "no" for how it answers that question, or "off_topic" if it does not really answer it."""
def schema(n):
    return {"type": "object", "properties": {"question": {"type": "string"},
            "answers": {"type": "array", "minItems": n, "maxItems": n,
                        "items": {"type": "string", "enum": ["yes", "no", "off_topic"]}}},
            "required": ["question", "answers"]}
def canon(idx):
    items = "\n".join(f"{k+1}. {pos[i]['claim']}" for k, i in enumerate(idx))
    o = llm.generate(PROMPT.format(items=items), schema=schema(len(idx)),
                     system="You map the arguments in a research corpus. Be precise.")
    return idx, o
t0 = time.time()
out_groups, pairs = [], []
with ThreadPoolExecutor(4) as pool:
    for idx, o in pool.map(canon, cand):
        if len(o["answers"]) != len(idx):
            continue
        members = [{**pos[i], "canon": a} for i, a in zip(idx, o["answers"])]
        out_groups.append({"question": o["question"], "members": members})
        yes = [m for m in members if m["canon"] == "yes"]
        no = [m for m in members if m["canon"] == "no"]
        for a in yes:
            for b in no:
                if a["node"] != b["node"]:
                    pairs.append({"question": o["question"], "yes": a, "no": b,
                                  "cross_source": a["item"] != b["item"]})
print(f"LLM grouping: {len(cand)} calls in {time.time()-t0:.0f}s", flush=True)
(HERE / "groups.json").write_text(json.dumps(out_groups, indent=1, ensure_ascii=False))
(HERE / "pairs.json").write_text(json.dumps(pairs, indent=1, ensure_ascii=False))
nodepairs = {tuple(sorted((p["yes"]["node"], p["no"]["node"]))) for p in pairs}
print(f"opposing claim pairs {len(pairs)}, passage pairs {len(nodepairs)}, "
      f"cross-source {sum(1 for p in pairs if p['cross_source'])}")
