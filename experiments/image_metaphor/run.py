"""Image retrieval test: does the Images action find pictures by subject AND by metaphor?

gold.json: passages (mostly the provocations) with the images that show what they talk about
(`subject`) and the images that could stand for their ideas (`metaphor`). Labels are Claude's.

Each signal is scored on its own, then the combined action, for subject and metaphor golds:
  visual            passage vector vs the picture itself (gemma only)
  description       passage vector vs the image description
  subject           description + picture, rank-fused (the action's subject route)
  meta~passage      passage vector vs the metaphor lines
  meta~concepts     LLM concepts of the passage vs the metaphor lines (the action's metaphor route)
  ... (raw)         the same without hubness centring (see actions._sims)
  meta~both         concepts and the passage itself vs the metaphor lines
  action            the Images action's interleaved list

    python experiments/image_metaphor/run.py                                       # .env: the gemma build
    DIP_DATA_DIR=data DIP_EMBED_BACKEND=ollama python experiments/image_metaphor/run.py  # the nomic build
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dip import config, db, llm  # noqa: E402
from dip.query import actions  # noqa: E402

HERE = Path(__file__).parent
K = 3  # the library has ~20 images, so recall@3 is a meaningful bar


def score(ranked: list[int], gold: set[int]) -> tuple[float, float]:
    """(recall@K, reciprocal rank of the first gold image)."""
    if not gold:
        return None
    rank = next((p for p, i in enumerate(ranked) if i in gold), None)
    return len(gold & set(ranked[:K])) / len(gold), (1 / (rank + 1) if rank is not None else 0.0)


def main():
    conn = db.connect()
    slug_id = {r["slug"]: r["id"] for r in conn.execute("SELECT id, slug FROM images")}
    slug_of = {v: k for k, v in slug_id.items()}
    has_visual = conn.execute("SELECT count(*) FROM vectors WHERE owner='image_visual'").fetchone()[0] > 0
    gold = json.load(open(HERE / "gold.json"))
    methods = (["visual"] if has_visual else []) + [
        "description (raw)", "description", "subject (raw)", "subject", "meta~passage",
        "meta~concepts (raw)", "meta~concepts", "meta~both", "action"]
    totals = {m: {"subject": [], "metaphor": []} for m in methods}
    examples = []
    for g in gold:
        vec = llm.embed_one(g["text"], query=True)
        anchor = actions.resolve(conn, text=g["text"])
        concepts, how = actions.passage_concepts(conn, anchor, g["text"])
        cvecs = [llm.embed_one(c, query=True) for c in concepts]
        ranked = {
            "visual": lambda: [i for i, *_ in actions.subject_ranking(conn, vec, desc=False)],
            "description (raw)": lambda: [i for i, *_ in actions.subject_ranking(conn, vec, visual=False,
                                                                                 centre=False)],
            "description": lambda: [i for i, *_ in actions.subject_ranking(conn, vec, visual=False)],
            "subject (raw)": lambda: [i for i, *_ in actions.subject_ranking(conn, vec, centre=False)],
            "subject": lambda: [i for i, *_ in actions.subject_ranking(conn, vec)],
            "meta~passage": lambda: [i for i, *_ in actions.metaphor_ranking(conn, [g["text"]], qvecs=[vec])],
            "meta~concepts (raw)": lambda: [i for i, *_ in actions.metaphor_ranking(conn, concepts,
                                                                                    centre=False)],
            "meta~concepts": lambda: [i for i, *_ in actions.metaphor_ranking(conn, concepts, qvecs=cvecs)],
            "meta~both": lambda: [i for i, *_ in actions.metaphor_ranking(conn, concepts + ["(the passage)"],
                                                                          qvecs=cvecs + [vec])],
            "action": lambda: [r["id"] for r in actions.images(conn, text=g["text"], k=len(slug_id))["results"]],
        }
        for m in methods:
            ids = ranked[m]()
            for kind in ("subject", "metaphor"):
                s = score(ids, {slug_id[x] for x in g[kind] if x in slug_id})
                if s is not None:
                    totals[m][kind].append(s)
            if m == "action":
                examples.append((g, concepts, how, [slug_of[i] for i in ids[:4]]))

    name = f"{config.EMBED_BACKEND}:{config.EMBED_MODEL}"
    lines = [f"# Image retrieval: {name} on {config.DATA_DIR.name}", "",
             f"{len(gold)} passages, {len(slug_id)} images, recall@{K} and MRR. Gold labels are Claude's.", "",
             "| signal | subject R@3 | subject MRR | metaphor R@3 | metaphor MRR |", "|---|---|---|---|---|"]
    for m in methods:
        cells = []
        for kind in ("subject", "metaphor"):
            xs = totals[m][kind]
            cells += [f"{sum(r for r, _ in xs) / len(xs):.0%}", f"{sum(rr for _, rr in xs) / len(xs):.2f}"]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "## Top 4 from the action", ""]
    for g, concepts, how, top in examples:
        want = ", ".join(g["subject"] + [f"*{x}*" for x in g["metaphor"]])
        lines += [f"- **{g['text'][:110]}**", f"  - concepts ({how}): " + "; ".join(concepts),
                  f"  - got: {', '.join(top)} · wanted: {want}"]
    out = "\n".join(lines) + "\n"
    print(out)
    (HERE / f"results-{config.EMBED_BACKEND}.md").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
