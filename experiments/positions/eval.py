"""Score the position-based approach and the earlier passage-based judges against gold.json
(the 60 stance pairs relabelled under the position rule; a user label wins over Claude's).

A passage pair's position score = the highest score over all position pairs between its two
passages (0 if either passage has no positions). Metrics are re-weighted by stratum as in
../stance/eval.py.

    python eval.py [--top N]     also list the N highest-scoring K-NN position pairs for review
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stance"))
from eval import decisions, metrics  # noqa: E402  (../stance/eval.py)

HERE = Path(__file__).parent
STANCE = HERE.parent / "stance"


def load_gold():
    g = json.loads((HERE / "gold.json").read_text())
    w = {s: v["n_total"] / v["n_sampled"] for s, v in g["strata"].items()}
    return [{"pair": tuple(it["pair"]), "key": f"{it['pair'][0]}-{it['pair'][1]}",
             "pos": (it["user"] or it["claude"]) in g["counter_worthy"], "w": w[it["stratum"]]} for it in g["items"]]


def main():
    gold = load_gold()
    print(f"gold: {len(gold)} pairs, {sum(g['pos'] for g in gold)} counter-worthy under the position rule\n")
    print(f"{'approach':44} {'prec':>5} {'rec':>5} {'F1':>5} {'AUC':>5} {'flag':>4}")
    for path in sorted(HERE.glob("judged__*.json")):
        res = json.loads(path.read_text())
        best = {}
        for v in res.values():
            k = tuple(sorted((v["a"]["node"], v["b"]["node"])))
            best[k] = max(best.get(k, 0.0), v["score"])
        rows = [(best.get(g["pair"], 0.0), g["pos"], g["w"]) for g in gold]
        p, r, f, auc, fl = metrics(rows)
        print(f"{'positions, ' + path.stem.split('__')[1]:44} {p:5.2f} {r:5.2f} {f:5.2f} {auc:5.2f} {fl:4}")
    for path in sorted((STANCE / "results").glob("*.json")):
        variant = path.stem.split("__")[0]
        res = json.loads(path.read_text())
        rule = "opposed" if variant == "strict3" else "mean of orders"
        rows = [(decisions(variant, res[g["key"]])[rule][0], g["pos"], g["w"]) for g in gold]
        p, r, f, auc, fl = metrics(rows)
        print(f"{'passages, ' + path.stem:44} {p:5.2f} {r:5.2f} {f:5.2f} {auc:5.2f} {fl:4}")

    if "--top" in sys.argv:
        n = int(sys.argv[sys.argv.index("--top") + 1])
        for path in sorted(HERE.glob("judged__*.json")):
            res = [v for v in json.loads(path.read_text()).values() if v["source"] == "knn"]
            res.sort(key=lambda v: -v["score"])
            print(f"\n== top {n} K-NN position pairs, {path.stem}")
            for i, v in enumerate(res[:n]):
                a, b = v["a"], v["b"]
                print(f"{i:02d} {v['score']:.2f}  {a['node']}-{b['node']}{' cross' if a['item'] != b['item'] else ''}\n"
                      f"   A [{a['voice']}, {a['holder']}] {a['statement']}\n"
                      f"   B [{b['voice']}, {b['holder']}] {b['statement']}")


if __name__ == "__main__":
    main()
