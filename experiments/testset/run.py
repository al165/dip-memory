"""Run judges on the test-set pairs, then score everything against gold.json.

    python run.py                 judge with the default line-up, then score
    python run.py --score-only    just score what is in results/

Line-up: current stage-5 CONTRADICTS links (read from the DB, no LLM), and ../stance/judges.py's
graded and strict3 judges on whole passages with qwen3:8b and qwen3:14b. Results are cached per
pair in results/<variant>__<model>.json. Metrics are re-weighted by stratum (cross / within) to
estimate performance over the whole nearest-neighbour pool.
"""
import json, sqlite3, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent / "stance"))
import judges  # ../stance/judges.py
from eval import metrics  # ../stance/eval.py

RUNS = [("graded", "qwen3:14b"), ("graded", "qwen3:8b"), ("strict3", "qwen3:8b")]

gold = json.loads((HERE / "gold.json").read_text())
items = gold["items"]
db = sqlite3.connect("file:" + str(HERE.parent.parent / "data" / "dip.sqlite") + "?mode=ro", uri=True)
db.row_factory = sqlite3.Row


def text_of(nid):
    r = db.execute("""SELECT n.text, s.title sec, i.title doc FROM nodes n JOIN nodes s ON s.id=n.parent_id
                      JOIN items i ON i.id=n.item_id WHERE n.id=?""", (nid,)).fetchone()
    return {"a": r["text"], "sa": r["sec"], "da": r["doc"][:60]}


def judge_all():
    text = {n: text_of(n) for it in items for n in it["pair"]}
    (HERE / "results").mkdir(exist_ok=True)
    for variant, model in RUNS:
        path = HERE / "results" / f"{variant}__{model.replace(':', '-')}.json"
        done = json.loads(path.read_text()) if path.exists() else {}
        todo = [tuple(it["pair"]) for it in items if f"{it['pair'][0]}-{it['pair'][1]}" not in done]
        t0 = time.time()
        with ThreadPoolExecutor(4) as pool:
            for k, r in zip(todo, pool.map(lambda k: judges.judge(variant, model, k, text), todo)):
                done[f"{k[0]}-{k[1]}"] = r
        path.write_text(json.dumps(done, indent=1))
        print(f"{variant} {model}: {len(todo)} judged in {time.time() - t0:.0f}s", flush=True)


def score_of(variant, r):
    if variant == "strict3":
        return r["score"]
    return (r["ab"]["score"] + r["ba"]["score"]) / 2


def main():
    if "--score-only" not in sys.argv:
        judge_all()
    w = {s: v["n_total"] / v["n_sampled"] for s, v in gold["strata"].items()}
    rows_for = {}
    # whatever stage 5 last stored: 1 if a CONTRADICTS edge joins the two passages
    edges = {(r[0], r[1]) for r in db.execute("SELECT src_node, dst_node FROM edges WHERE type='CONTRADICTS'")}
    rows_for["stored CONTRADICTS links (live DB)"] = [
        (1.0 if tuple(it["pair"]) in edges or tuple(it["pair"][::-1]) in edges else 0.0, it) for it in items]
    for path in sorted((HERE / "results").glob("*.json")):
        variant = path.stem.split("__")[0]
        res = json.loads(path.read_text())
        rows_for[f"{variant}, {path.stem.split('__')[1]}"] = [
            (score_of(variant, res[f"{it['pair'][0]}-{it['pair'][1]}"]), it) for it in items]
    npos = sum((it["user"] or it["claude"]) in gold["counter_worthy"] for it in items)
    print(f"\ngold: {len(items)} pairs, {npos} counter-worthy; prec/rec/F1/AUC re-weighted to the pool\n")
    print(f"{'judge':36} {'prec':>5} {'rec':>5} {'F1':>5} {'AUC':>5} {'flag':>4}  {'flags on junk':>13}")
    for name, rows in rows_for.items():
        triples = [(s, (it["user"] or it["claude"]) in gold["counter_worthy"], w[it["stratum"]]) for s, it in rows]
        p, r, f, auc, fl = metrics(triples)
        junk = sum(1 for s, it in rows if s >= 0.5 and it.get("junk_side"))
        print(f"{name:36} {p:5.2f} {r:5.2f} {f:5.2f} {auc:5.2f} {fl:4}  {junk:13}")


if __name__ == "__main__":
    main()
