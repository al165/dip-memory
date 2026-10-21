"""Run one judge variant with one Ollama model over the 619 candidate passage pairs.

    python judges.py strict3 qwen3:8b
    python judges.py graded qwen3:14b

Variants
  strict3  one call per pair: opposed / compatible / different_question (verify_passages.py's prompt).
           score = 1 if opposed else 0.
  graded   one call per pair *per order* (A,B) and (B,A): a digit 1-5 for how B relates to A.
           The digit's probability distribution is read from the logprobs (as for abstraction),
           so score = P(1 opposed) + P(2 qualifies), per order. Stored per order; eval.py combines.

Results are cached per pair in results/<variant>__<model>.json, so an interrupted run resumes.
"""
import json, math, re, sqlite3, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dip import llm

HERE = Path(__file__).parent
SYSTEM = "You judge argumentative relations between passages. Be strict."

STRICT_SCHEMA = {"type": "object", "properties": {"relation": {"type": "string",
                 "enum": ["opposed", "compatible", "different_question"]}}, "required": ["relation"]}
STRICT_PROMPT = """Passage A ({da}, "{sa}"):
{a}

Passage B ({db}, "{sb}"):
{b}

Would a reader see A and B as two sides of a disagreement?
- opposed: they argue for conflicting answers to the same question; accepting one means pushing back on the other
- compatible: the same person could hold both (including when one describes a problem and the other criticises or responds to it)
- different_question: they are about different things, even if the topic is related"""

GRADED_SCHEMA = {"type": "object", "properties": {"relation": {"type": "integer", "minimum": 1, "maximum": 5}},
                 "required": ["relation"]}
GRADED_PROMPT = """Passage A ({da}, "{sa}"):
{a}

Passage B ({db}, "{sb}"):
{b}

How does B relate to the position A takes? Answer with one digit:
1 = opposed: B argues against A's position on the same question
2 = qualifies: B partly pushes back on A, adding a limit, exception or complication
3 = different question: B is about something else, even if the topic is related
4 = compatible: the same person could hold both, including when one describes a problem and the other criticises or responds to it
5 = supports: B agrees with A or backs it up"""


def digit_probs(token_logprobs, key="relation", lo=1, hi=5):
    """{digit: probability} at the position where the JSON value of `key` is written."""
    text = ""
    for tok in token_logprobs:
        if tok["token"].strip().isdigit() and re.search(rf'"{key}"\s*:\s*$', text):
            probs = {}
            for alt in tok.get("top_logprobs") or [tok]:
                d = alt["token"].strip()
                if d.isdigit() and lo <= int(d) <= hi:
                    probs[int(d)] = probs.get(int(d), 0.0) + math.exp(alt["logprob"])
            total = sum(probs.values())
            return {k: v / total for k, v in probs.items()} if total else None
        text += tok["token"]
    return None


def load_pairs():
    c = sqlite3.connect("file:" + str(HERE.parent.parent / "data" / "dip.sqlite") + "?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    pairs = [tuple(r["pair"]) for r in json.loads((HERE / "verified_passages.json").read_text())]
    text = {}
    for k in pairs:
        for n in k:
            if n not in text:
                r = c.execute("""SELECT n.text, s.title sec, i.title doc FROM nodes n
                    JOIN nodes s ON s.id=n.parent_id JOIN items i ON i.id=n.item_id WHERE n.id=?""", (n,)).fetchone()
                text[n] = {"a": r["text"], "sa": r["sec"], "da": r["doc"][:60]}
    return pairs, text


def fill(prompt, A, B):
    return prompt.format(a=A["a"], sa=A["sa"], da=A["da"], b=B["a"], sb=B["sa"], db=B["da"])


def judge(variant, model, k, text):
    A, B = text[k[0]], text[k[1]]
    if variant == "strict3":
        rel = llm.generate(fill(STRICT_PROMPT, A, B), schema=STRICT_SCHEMA, system=SYSTEM, model=model)["relation"]
        return {"relation": rel, "score": 1.0 if rel == "opposed" else 0.0}
    out = {}
    for name, (x, y) in (("ab", (A, B)), ("ba", (B, A))):
        res, lp = llm.generate(fill(GRADED_PROMPT, x, y), schema=GRADED_SCHEMA, system=SYSTEM,
                               model=model, logprobs=True)
        p = digit_probs(lp) or {res["relation"]: 1.0}
        out[name] = {"digit": res["relation"], "probs": p, "score": p.get(1, 0.0) + p.get(2, 0.0)}
    return out


def main():
    variant, model = sys.argv[1], sys.argv[2]
    assert variant in ("strict3", "graded"), variant
    names = {m["name"] for m in llm._client.get("/api/tags", timeout=5).json().get("models", [])}
    if model not in names:
        sys.exit(f"{model} is not pulled on the Ollama host yet (has: {', '.join(sorted(names))})")
    pairs, text = load_pairs()
    out_path = HERE / "results" / f"{variant}__{model.replace(':', '-')}.json"
    out_path.parent.mkdir(exist_ok=True)
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [k for k in pairs if f"{k[0]}-{k[1]}" not in done]
    print(f"{variant} with {model}: {len(pairs)} pairs, {len(todo)} to judge", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(4) as pool:
        futs = {pool.submit(judge, variant, model, k, text): k for k in todo}
        for i, f in enumerate(as_completed(futs), 1):
            k = futs[f]
            try:
                done[f"{k[0]}-{k[1]}"] = f.result()
            except llm.LLMError as e:
                print("skip", k, e, flush=True)
            if i % 50 == 0 or i == len(todo):
                out_path.write_text(json.dumps(done, indent=1))
                print(f"  {i}/{len(todo)}  {(time.time() - t0) / i:.2f}s/pair", flush=True)
    out_path.write_text(json.dumps(done, indent=1))
    print(f"done in {time.time() - t0:.0f}s -> {out_path.name}")


if __name__ == "__main__":
    main()
