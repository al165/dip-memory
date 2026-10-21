"""Stage 5 — L3 claim graph: typed edges between chunks, derived from their claims.

1. Candidates: for each claim, its EDGE_CANDIDATES nearest claims from *other* chunks.
   (Checking only near pairs keeps cost O(n·k) instead of O(n²), and stance only matters
   between claims that are about the same thing.)
2. NLI filter: a cross-encoder scores each pair in both directions and proposes
     contradiction > NLI_THRESHOLD           -> candidate CONTRADICTS
     entail(a => b) > NLI_THRESHOLD          -> candidate a SUPPORTS b
   plus "unsure but very similar" pairs (cos >= LLM_ESCALATE_SIM) and pairs whose abstraction
   differs by >= 2 (candidates for ELABORATES / EXAMPLE_OF).
3. LLM verification: the LLM judges every candidate pair (strongest first, up to
   LLM_VERIFY_MAX) and its verdict decides the edge — NLI alone is not trusted, because small
   NLI models label "related but not implied" pairs as contradictions.
Both NLI scores (nli_cache) and LLM verdicts (pair_cache) are cached by claim text, so after
adding sources a re-run only pays for pairs involving new claims.

Counter links (CONTRADICTS) come from a separate check when COUNTER_JUDGE = "graded" (default):
each passage and its COUNTER_NEIGHBOURS nearest passages are rated by COUNTER_MODEL on a 1-5
scale (opposed / qualifies / different question / compatible / supports), in both orders, with
the digit probabilities read from the logprobs. A pair whose mean P(opposed) + P(qualifies) is at
least COUNTER_THRESHOLD gets a CONTRADICTS link both ways, weighted by that score. Judging whole
passages beat claim-level NLI + LLM by a wide margin on a hand-labelled test set (AUC 0.93 vs
0.51, experiments/testset). Judgements are cached by passage text + model (counter_cache).
With COUNTER_JUDGE = "legacy", "challenges" verdicts from step 3 make the CONTRADICTS links.
Global stage: nli/llm/graded edges are rebuilt from scratch each run (wiki_lint edges are kept).
"""
import collections
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dip import config, db, llm, nli
from dip.ingest.common import log, progress

RELATIONS = ["supports", "challenges", "elaborates", "example", "unrelated"]
SCHEMA = {
    "type": "object",
    "properties": {"relation": {"type": "string", "enum": RELATIONS}},
    "required": ["relation"],
}
PROMPT = """Statement A: {a}
Statement B: {b}

How does B relate to A?
- supports: B gives evidence or reasons for A
- challenges: B disagrees with, contradicts or offers a counter-argument to A
- elaborates: B is a more detailed or specific treatment of A
- example: B is a concrete example or case of A
- unrelated: none of the above (including simply restating A)
"""
LLM_WEIGHT = 0.7


def _ask_llm(a: str, b: str) -> str:
    return llm.generate(PROMPT.format(a=a, b=b), schema=SCHEMA,
                        system="You judge argumentative relations between statements.")["relation"]


def _verdicts(conn, jobs: list[tuple[str, str]]) -> dict:
    """{(a_text, b_text): relation} for every job, from the cache or the LLM (in parallel).
    Only the main thread touches SQLite."""
    out, todo = {}, []
    for a, b in dict.fromkeys(jobs):
        hit = conn.execute("SELECT verdict FROM pair_cache WHERE a_text=? AND b_text=?", (a, b)).fetchone()
        if hit:
            out[(a, b)] = hit["verdict"]
        else:
            todo.append((a, b))
    log(f"  LLM verification: {len(out)} cached, {len(todo)} to ask "
        f"({config.LLM_CONCURRENCY} at a time)")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=config.LLM_CONCURRENCY) as pool:
        futures = {pool.submit(_ask_llm, a, b): (a, b) for a, b in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            a, b = futures[fut]
            try:
                out[(a, b)] = fut.result()
            except llm.LLMError as e:
                log(f"  pair skipped: {e}")
                continue
            conn.execute("INSERT OR REPLACE INTO pair_cache VALUES (?,?,?)", (a, b, out[(a, b)]))
            if i % 50 == 0 or i == len(todo):
                conn.commit()
                rate = i / max(time.time() - t0, 1e-6)
                log(f"  LLM verification: {i}/{len(todo)} ({rate:.1f}/s, ~{(len(todo) - i) / rate / 60:.1f} min left)")
    conn.commit()
    return out


def _nli_cached(conn, pairs, label, block=512):
    """NLI scores for (premise, hypothesis) pairs. Cached scores are reused; only new pairs are
    run through the (CPU-bound) cross-encoder, with progress, and then cached."""
    model = config.NLI_MODEL
    out = [None] * len(pairs)
    todo = []
    for i, (p, h) in enumerate(pairs):
        row = conn.execute("SELECT entailment, contradiction, neutral FROM nli_cache "
                           "WHERE premise=? AND hypothesis=? AND model=?", (p, h, model)).fetchone()
        if row:
            out[i] = dict(row)
        else:
            todo.append(i)
    log(f"  NLI {label}: {len(pairs) - len(todo)} cached, {len(todo)} to score")
    t0 = time.time()
    for start in range(0, len(todo), block):
        idxs = todo[start:start + block]
        scores = nli.score([pairs[i] for i in idxs])
        conn.executemany("INSERT OR REPLACE INTO nli_cache VALUES (?,?,?,?,?,?)",
                         [(*pairs[i], model, sc.get("entailment", 0.0), sc.get("contradiction", 0.0),
                           sc.get("neutral", 0.0)) for i, sc in zip(idxs, scores)])
        conn.commit()
        for i, sc in zip(idxs, scores):
            out[i] = sc
        done = start + len(idxs)
        rate = done / max(time.time() - t0, 1e-6)
        log(f"  NLI {label}: {done}/{len(todo)} scored ({rate:.0f}/s, ~{(len(todo) - done) / rate / 60:.1f} min left)")
    return out


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


def _ask_graded(a: dict, b: dict) -> dict[int, float]:
    """1-5 digit probabilities for how passage b relates to passage a."""
    res, lp = llm.generate(
        GRADED_PROMPT.format(a=a["text"], sa=a["section"], da=a["doc"], b=b["text"], sb=b["section"], db=b["doc"]),
        schema=GRADED_SCHEMA, system="You judge argumentative relations between passages. Be strict.",
        model=config.COUNTER_MODEL, logprobs=True)
    return llm.digit_probs(lp, "relation") or {int(res["relation"]): 1.0}


def _counter_edges(conn, edges: dict):
    """Add graded CONTRADICTS links between each passage and its nearest passages (see module doc)."""
    rows = {r["id"]: r for r in conn.execute(
        "SELECT n.id, n.text, s.title section, d.title doc FROM nodes n "
        "JOIN nodes s ON s.id = n.parent_id JOIN nodes d ON d.id = s.parent_id "
        "WHERE n.kind='chunk' AND coalesce(n.role, '') != 'link'")}
    idx = db.VectorIndex("node").refresh(conn)
    allowed = set(rows)
    pairs = set()
    for nid in rows:
        vec = idx.get(nid)
        if vec is None:
            continue
        for other, _ in idx.search(vec, k=config.COUNTER_NEIGHBOURS, allowed=allowed, exclude={nid}):
            pairs.add(tuple(sorted((nid, other))))

    model = config.COUNTER_MODEL
    probs, todo = {}, []
    for x, y in pairs:
        for a, b in ((x, y), (y, x)):
            hit = conn.execute("SELECT probs_json FROM counter_cache WHERE a_text=? AND b_text=? AND model=?",
                               (rows[a]["text"], rows[b]["text"], model)).fetchone()
            if hit:
                probs[(a, b)] = {int(d): p for d, p in json.loads(hit["probs_json"]).items()}
            else:
                todo.append((a, b))
    log(f"stage 5: counter check with {model}: {len(pairs)} passage pairs, "
        f"{len(probs)} judgements cached, {len(todo)} to ask ({config.LLM_CONCURRENCY} at a time)")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=config.LLM_CONCURRENCY) as pool:
        futures = {pool.submit(_ask_graded, rows[a], rows[b]): (a, b) for a, b in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            a, b = futures[fut]
            try:
                probs[(a, b)] = fut.result()
            except llm.LLMError as e:
                log(f"  counter judgement skipped (retried next run): {e}")
                continue
            conn.execute("INSERT OR REPLACE INTO counter_cache VALUES (?,?,?,?)",
                         (rows[a]["text"], rows[b]["text"], model, json.dumps(probs[(a, b)])))
            if i % 100 == 0 or i == len(todo):
                conn.commit()
                rate = i / max(time.time() - t0, 1e-6)
                log(f"  counter check: {i}/{len(todo)} ({rate:.1f}/s, ~{(len(todo) - i) / rate / 60:.0f} min left)")
    conn.commit()

    n = 0
    for x, y in pairs:
        if (x, y) not in probs or (y, x) not in probs:
            continue  # an order failed; the pair is retried next run
        score = sum(probs[o].get(1, 0.0) + probs[o].get(2, 0.0) for o in ((x, y), (y, x))) / 2
        if score >= config.COUNTER_THRESHOLD:
            _add(edges, x, y, "CONTRADICTS", score, "graded")
            _add(edges, y, x, "CONTRADICTS", score, "graded")
            n += 1
    log(f"stage 5: counter check linked {n} of {len(pairs)} passage pairs (threshold {config.COUNTER_THRESHOLD})")


def _add(edges: dict, src, dst, typ, weight, method, sc=None, dc=None):
    key = (src, dst, typ)
    if key not in edges or edges[key][2] < weight:
        edges[key] = (sc, dc, weight, method)


def run(conn):
    claims = {r["id"]: r for r in conn.execute(
        "SELECT c.id, c.node_id, c.text, n.abstraction FROM claims c JOIN nodes n ON n.id=c.node_id")}
    if not claims:
        log("stage 5: no claims (run stage 3 first)")
        return
    idx = db.VectorIndex("claim").refresh(conn)

    # 1. candidate pairs (unordered, across different chunks)
    by_node = collections.defaultdict(set)
    for cid, c in claims.items():
        by_node[c["node_id"]].add(cid)
    pairs = {}
    for cid, c in claims.items():
        vec = idx.get(cid)
        if vec is None:
            continue
        for other, sim in idx.search(vec, k=config.EDGE_CANDIDATES, exclude=by_node[c["node_id"]]):
            key = tuple(sorted((cid, other)))
            pairs[key] = max(pairs.get(key, 0.0), sim)
    pair_list = list(pairs.items())
    log(f"stage 5: {len(claims)} claims, {len(pair_list)} candidate pairs")

    # 2. NLI both directions (CPU-bound: ~20-100 pairs/s depending on the machine)
    fwd = _nli_cached(conn, [(claims[a]["text"], claims[b]["text"]) for (a, b), _ in pair_list], "forward")
    bwd = _nli_cached(conn, [(claims[b]["text"], claims[a]["text"]) for (a, b), _ in pair_list], "backward")

    # Candidate pairs for the LLM, oriented as (A, B) = "how does B relate to A?", with a priority.
    T = config.NLI_THRESHOLD
    cands = {}  # (a_claim, b_claim) -> (priority, nli_label, nli_prob)

    def propose(a, b, priority, label, prob):
        if (a, b) not in cands or cands[(a, b)][0] < priority:
            cands[(a, b)] = (priority, label, prob)

    def general_first(a, b):
        return (b, a) if (claims[a]["abstraction"] or 3) < (claims[b]["abstraction"] or 3) else (a, b)

    for ((a, b), sim), f, r in zip(pair_list, fwd, bwd):
        contra = max(f["contradiction"], r["contradiction"])
        flagged = False
        if contra > T:
            propose(*general_first(a, b), 3 + contra * sim, "challenges", contra)
            flagged = True
        if f["entailment"] > T:   # a entails b: does a support b?  A=b, B=a
            propose(b, a, 2 + f["entailment"] * sim, "supports", f["entailment"])
            flagged = True
        if r["entailment"] > T:
            propose(a, b, 2 + r["entailment"] * sim, "supports", r["entailment"])
            flagged = True
        gap = abs((claims[a]["abstraction"] or 3) - (claims[b]["abstraction"] or 3))
        if not flagged and (sim >= config.LLM_ESCALATE_SIM or gap >= 2):
            propose(*general_first(a, b), 1 + sim, None, None)

    ranked = sorted(cands.items(), key=lambda x: -x[1][0])
    escalations = [x for x in ranked if x[1][1] is None][: config.LLM_ESCALATE_MAX]
    checked = sorted([x for x in ranked if x[1][1] is not None] + escalations, key=lambda x: -x[1][0])
    dropped = len(checked) - config.LLM_VERIFY_MAX
    checked = checked[: config.LLM_VERIFY_MAX]
    nli_flags = collections.Counter(x[1][1] for x in checked if x[1][1])
    log(f"stage 5: NLI flagged {dict(nli_flags)}, +{len(escalations)} similar/abstraction pairs"
        + (f"; {dropped} lowest-priority pairs over LLM_VERIFY_MAX skipped" if dropped > 0 else ""))

    verdicts = _verdicts(conn, [(claims[a]["text"], claims[b]["text"]) for (a, b), _ in checked])

    # 3. edges from LLM verdicts. When the LLM agrees with NLI, keep the NLI probability as weight.
    edges, agree = {}, collections.Counter()
    for (a, b), (_, label, prob) in checked:
        rel = verdicts.get((claims[a]["text"], claims[b]["text"]))
        if rel is None or rel == "unrelated":
            if label:
                agree[f"{label}: rejected"] += 1
            continue
        if label:
            agree[f"{label}: {'confirmed' if rel == label else 'relabelled ' + rel}"] += 1
        w, method = (prob, "nli+llm") if rel == label else (LLM_WEIGHT, "llm")
        A, B = claims[a]["node_id"], claims[b]["node_id"]
        if rel == "supports":
            _add(edges, B, A, "SUPPORTS", w, method, b, a)
        elif rel == "challenges" and config.COUNTER_JUDGE == "legacy":
            _add(edges, A, B, "CONTRADICTS", w, method, a, b)
            _add(edges, B, A, "CONTRADICTS", w, method, b, a)
        elif rel == "elaborates":
            _add(edges, B, A, "ELABORATES", w, method, b, a)
        elif rel == "example":
            _add(edges, B, A, "EXAMPLE_OF", w, method, b, a)
    log(f"stage 5: NLI vs LLM {dict(agree)}")

    if config.COUNTER_JUDGE == "graded":
        _counter_edges(conn, edges)

    with db.tx(conn):
        conn.execute("DELETE FROM edges WHERE method IN ('nli','llm','nli+llm','graded')")
        conn.executemany(
            "INSERT INTO edges(src_node, dst_node, src_claim, dst_claim, type, weight, method) "
            "VALUES (?,?,?,?,?,?,?) ON CONFLICT(src_node, dst_node, type) DO UPDATE SET "
            "weight=max(weight, excluded.weight)",
            [(s, d, sc, dc, t, w, m) for (s, d, t), (sc, dc, w, m) in edges.items()],
        )
    counts = collections.Counter(t for (_, _, t) in edges)
    log(f"stage 5: edges {dict(counts)} from {len(checked)} LLM-checked pairs")
