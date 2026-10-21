"""Hybrid: positions *find* candidate passage pairs, the whole-passage graded judge *decides*.

Candidates = passage pairs behind any nearest-neighbour position pair scoring >= THR in
judged__<model>.json, except pairs where both positions are rejected views. Each candidate passage
pair is judged by ../stance/judges.py's graded prompt (both orders, digit probabilities), reusing
../stance/results/graded__<model>.json where the pair was already judged.

    python hybrid.py [model] [THR]        defaults: qwen3:14b 0.8  -> hybrid__<model>.json
"""
import json, sqlite3, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent / "stance"))
from dip import llm
import judges  # ../stance/judges.py

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3:14b"
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
tag = MODEL.replace(":", "-")

pos_pairs = [v for v in json.loads((HERE / f"judged__{tag}.json").read_text()).values() if v["source"] == "knn"]
cands = {}
for v in pos_pairs:
    voices = sorted((v["a"]["voice"], v["b"]["voice"]))
    if v["score"] < THR or voices == ["rejected", "rejected"]:
        continue
    k = tuple(sorted((v["a"]["node"], v["b"]["node"])))
    c = cands.setdefault(k, {"position_pairs": []})
    a, b = (v["a"], v["b"]) if v["a"]["node"] == k[0] else (v["b"], v["a"])
    c["position_pairs"].append({"score": v["score"], "a": {x: a[x] for x in ("statement", "voice", "holder")},
                                "b": {x: b[x] for x in ("statement", "voice", "holder")}})

db = sqlite3.connect("file:" + str(HERE.parent.parent / "data" / "dip.sqlite") + "?mode=ro", uri=True)
db.row_factory = sqlite3.Row
text = {}
for k in cands:
    for n in k:
        if n not in text:
            r = db.execute("""SELECT n.text, n.item_id, s.title sec, i.title doc FROM nodes n JOIN nodes s ON s.id=n.parent_id
                              JOIN items i ON i.id=n.item_id WHERE n.id=?""", (n,)).fetchone()
            text[n] = {"a": r["text"], "sa": r["sec"], "da": r["doc"][:60], "item": r["item_id"]}

cached = {}
old_path = HERE.parent / "stance" / "results" / f"graded__{tag}.json"
if old_path.exists():
    cached = json.loads(old_path.read_text())
out_path = HERE / f"hybrid__{tag}.json"
done = json.loads(out_path.read_text()) if out_path.exists() else {}
todo = []
for k, c in cands.items():
    key = f"{k[0]}-{k[1]}"
    if key in done:
        continue
    if key in cached:
        done[key] = {**c, "judge": cached[key], "from_cache": True}
    else:
        todo.append(k)
print(f"{len(cands)} candidate passage pairs (THR {THR}); {len(cands) - len(todo)} cached, {len(todo)} to judge", flush=True)

t0 = time.time()
with ThreadPoolExecutor(4) as pool:
    futs = {pool.submit(judges.judge, "graded", MODEL, k, text): k for k in todo}
    for i, f in enumerate(as_completed(futs), 1):
        k = futs[f]
        try:
            done[f"{k[0]}-{k[1]}"] = {**cands[k], "judge": f.result(), "from_cache": False}
        except llm.LLMError as e:
            print("skip", k, e, flush=True)
        if i % 50 == 0 or i == len(todo):
            out_path.write_text(json.dumps(done, indent=1, ensure_ascii=False))
            print(f"  {i}/{len(todo)}  {(time.time() - t0) / i:.2f}s/pair", flush=True)
for key, v in done.items():
    a, b = map(int, key.split("-"))
    v["cross_source"] = text[a]["item"] != text[b]["item"]
    v["score"] = (v["judge"]["ab"]["score"] + v["judge"]["ba"]["score"]) / 2
out_path.write_text(json.dumps(done, indent=1, ensure_ascii=False))
print(f"done in {time.time() - t0:.0f}s -> {out_path.name}")
