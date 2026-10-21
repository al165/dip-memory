"""Steps 2-3: candidate position pairs, then the graded 1-5 judge on position statements.

Candidates: each position's K nearest positions (cosine on nomic embeddings) from other passages,
skipping pairs where the same author argues both (a text rarely opposes itself). Every position
pair between the two passages of each gold pair is added too, so eval.py can score all 60.

The judge sees only the two statements (who holds them is deliberately left out: the question is
whether the positions conflict). Both orders, digit probabilities from logprobs, as in
../stance/judges.py; score = mean over orders of P(1 opposed) + P(2 qualifies).

    python judge.py [model] [K]        defaults: qwen3:14b 6  -> judged.json
"""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stance"))
from dip import llm
from judges import digit_probs

HERE = Path(__file__).parent
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3:14b"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 6
SCHEMA = {"type": "object", "properties": {"relation": {"type": "integer", "minimum": 1, "maximum": 5}},
          "required": ["relation"]}
PROMPT = """Position A: {a}
Position B: {b}

Judge the two positions as stated, whoever holds them. How does B relate to A? Answer with one digit:
1 = opposed: B conflicts with A; holding one means rejecting the other
2 = qualifies: B partly pushes back on A, adding a limit, exception or complication
3 = different question: B is about something else, even if the topic is related
4 = compatible: the same person could hold both
5 = supports: B agrees with A or backs it up"""

data = json.loads((HERE / "positions.json").read_text())
pos = [{"id": i, "node": int(n), "item": v["item"], **p}
       for i, (n, v, p) in enumerate((n, v, p) for n, v in data.items() for p in v["positions"])]
emb_path = HERE / "statement_emb.npy"
if emb_path.exists() and len(np.load(emb_path)) == len(pos):
    X = np.load(emb_path)
else:
    X = np.array(llm.embed([p["statement"] for p in pos]), dtype=np.float32)
    np.save(emb_path, X)
X /= np.linalg.norm(X, axis=1, keepdims=True)
S = X @ X.T


def same_author_both_argued(a, b):
    return a["item"] == b["item"] and a["voice"] == b["voice"] == "argued" and a["holder"] == b["holder"]


pairs = {}
for a in pos:
    order = np.argsort(-S[a["id"]])
    n = 0
    for j in order:
        b = pos[j]
        if b["node"] == a["node"] or same_author_both_argued(a, b):
            continue
        pairs.setdefault(tuple(sorted((a["id"], b["id"]))), "knn")
        n += 1
        if n == K:
            break
by_node = {}
for p in pos:
    by_node.setdefault(p["node"], []).append(p["id"])
gold = json.loads((HERE / "gold.json").read_text())
for it in gold["items"]:
    x, y = it["pair"]
    for i in by_node.get(x, []):
        for j in by_node.get(y, []):
            k = tuple(sorted((i, j)))
            pairs[k] = pairs.get(k, "gold") if pairs.get(k) == "knn" else "gold"
print(f"{len(pos)} positions, {len(pairs)} position pairs "
      f"({sum(v == 'knn' for v in pairs.values())} from {K}-NN, rest gold-only)", flush=True)

out_path = HERE / f"judged__{MODEL.replace(':', '-')}.json"
done = json.loads(out_path.read_text()) if out_path.exists() else {}
todo = [k for k in pairs if f"{k[0]}-{k[1]}" not in done]


def judge(k):
    a, b = pos[k[0]], pos[k[1]]
    out = {}
    for name, (x, y) in (("ab", (a, b)), ("ba", (b, a))):
        res, lp = llm.generate(PROMPT.format(a=x["statement"], b=y["statement"]), schema=SCHEMA,
                               system="You judge argumentative relations between positions. Be strict.",
                               model=MODEL, logprobs=True)
        p = digit_probs(lp) or {res["relation"]: 1.0}
        out[name] = {"digit": res["relation"], "probs": p, "score": p.get(1, 0.0) + p.get(2, 0.0)}
    return k, out


t0 = time.time()
print(f"{len(todo)} to judge with {MODEL}", flush=True)
with ThreadPoolExecutor(4) as pool:
    futs = [pool.submit(judge, k) for k in todo]
    for i, f in enumerate(as_completed(futs), 1):
        try:
            k, o = f.result()
        except llm.LLMError as e:
            print("skip", e, flush=True)
            continue
        done[f"{k[0]}-{k[1]}"] = {"a": pos[k[0]], "b": pos[k[1]], "source": pairs[k], **o,
                                  "score": (o["ab"]["score"] + o["ba"]["score"]) / 2}
        if i % 100 == 0 or i == len(todo):
            out_path.write_text(json.dumps(done, indent=1, ensure_ascii=False))
            print(f"  {i}/{len(todo)}  {(time.time() - t0) / i:.2f}s/pair", flush=True)
out_path.write_text(json.dumps(done, indent=1, ensure_ascii=False))
print(f"done in {time.time() - t0:.0f}s -> {out_path.name}")
