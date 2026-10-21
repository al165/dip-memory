"""Score every results/*.json judge run against gold.json.

Gold label per pair = the user's label if given, else Claude's. "Counter-worthy" = opposed or
qualifies. The 60 gold pairs were drawn 30 + 30 from two strata (pairs qwen3:8b's earlier passage
check called opposed, and the rest), so precision/recall are re-weighted by stratum size to
estimate performance over all 619 candidate pairs.

    python eval.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
DIGIT = {1: "O", 2: "Q", 3: "D", 4: "C", 5: "S"}


def load_gold():
    g = json.loads((HERE / "gold.json").read_text())
    w = {s: v["n_total"] / v["n_sampled"] for s, v in g["strata"].items()}
    items = []
    for it in g["items"]:
        lab = it["user"] or it["claude"]
        items.append({"key": f"{it['pair'][0]}-{it['pair'][1]}", "label": lab, "pos": lab in g["counter_worthy"],
                      "w": w[it["stratum"]], "by": "user" if it["user"] else "claude"})
    return items


def decisions(variant, r):
    """{decision rule name: (score in [0,1], predicted 5-way label or None)}"""
    if variant == "strict3":
        return {"opposed": (r["score"], None)}
    ab, ba = ({**o, "probs": {int(d): p for d, p in o["probs"].items()}} for o in (r["ab"], r["ba"]))
    mean = (ab["score"] + ba["score"]) / 2
    # 5-way label from the averaged digit distribution
    probs = {d: (ab["probs"].get(d, 0) + ba["probs"].get(d, 0)) / 2 for d in range(1, 6)}
    top = DIGIT[max(probs, key=probs.get)]
    return {"mean of orders": (mean, top), "both orders": (min(ab["score"], ba["score"]), top),
            "A→B only": (ab["score"], DIGIT[max(ab["probs"], key=ab["probs"].get)])}


def metrics(rows, thr=0.5):
    tp = sum(w for s, pos, w in rows if s >= thr and pos)
    fp = sum(w for s, pos, w in rows if s >= thr and not pos)
    fn = sum(w for s, pos, w in rows if s < thr and pos)
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p + r else float("nan")
    # weighted AUC: P(score of a random positive > score of a random negative)
    pos = [(s, w) for s, y, w in rows if y]; neg = [(s, w) for s, y, w in rows if not y]
    num = sum(wp * wn * (1.0 if sp > sn else 0.5 if sp == sn else 0.0) for sp, wp in pos for sn, wn in neg)
    den = sum(wp for _, wp in pos) * sum(wn for _, wn in neg)
    flagged = sum(1 for s, _, _ in rows if s >= thr)
    return p, r, f, (num / den if den else float("nan")), flagged


def main():
    gold = load_gold()
    npos = sum(g["pos"] for g in gold)
    print(f"gold: {len(gold)} pairs, {npos} counter-worthy "
          f"({sum(g['by'] == 'user' for g in gold)} labelled by user, rest by Claude)\n")
    print(f"{'run':30} {'rule':16} {'prec':>5} {'rec':>5} {'F1':>5} {'AUC':>5} {'flag':>4} {'5-way':>6}")
    for path in sorted((HERE / "results").glob("*.json")):
        variant, model = path.stem.split("__")
        res = json.loads(path.read_text())
        missing = [g["key"] for g in gold if g["key"] not in res]
        if missing:
            print(f"{path.stem:30} incomplete ({len(missing)} gold pairs not judged yet)")
            continue
        rules = {}
        for g in gold:
            for rule, (s, top) in decisions(variant, res[g["key"]]).items():
                rules.setdefault(rule, []).append((s, g["pos"], g["w"], top, g["label"]))
        for rule, rows in rules.items():
            p, r, f, auc, flagged = metrics([(s, y, w) for s, y, w, _, _ in rows])
            acc = (sum(1 for *_, top, lab in rows if top == lab) / len(rows)) if rows[0][3] else None
            print(f"{path.stem:30} {rule:16} {p:5.2f} {r:5.2f} {f:5.2f} {auc:5.2f} {flagged:4} "
                  f"{'' if acc is None else f'{acc:6.2f}'}")
    print("\nprec/rec/F1/AUC are re-weighted to all 619 pairs; 'flag' = gold pairs scored >= 0.5;"
          "\n'5-way' = share of gold pairs where the most likely digit matches the gold label (unweighted).")


if __name__ == "__main__":
    main()
