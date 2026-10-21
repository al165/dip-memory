"""The six query actions. Each is a recipe over the four index layers:

  L1 vectors + BM25      neighbours, candidate generation
  L2 hierarchy           zoom-in / zoom-out (structural + semantic tree, abstraction score)
  L3 claim graph         counter / evidence (typed edges), zoom-in (ELABORATES / EXAMPLE_OF)
  L4 wiki                zoom-out (pages covering a chunk), question
  image library          images (by subject and by metaphor; see dip/ingest/s7_images.py)

Every result carries a `why` string naming the layer/edge that produced it.
"""
import json
import re

import numpy as np

from dip import config, db, llm, nli
from dip.query import retrieve

ROLE_BOOST = {"evidence": 1.3, "example": 1.2, "caption": 1.1, "definition": 1.0}


class NotFound(LookupError):
    pass


# --- result formatting -----------------------------------------------------------------

def _context(conn, row) -> dict:
    """Section/document titles for a chunk, for display."""
    sec = db.node(conn, row["parent_id"]) if row["parent_id"] and row["kind"] == "chunk" else None
    doc_id = sec["parent_id"] if sec else (row["parent_id"] if row["kind"] == "section" else None)
    doc = db.node(conn, doc_id) if doc_id else None
    item = conn.execute("SELECT path, type FROM items WHERE id=?", (row["item_id"],)).fetchone() \
        if row["item_id"] else None
    return {"section": sec["title"] if sec else None, "document": doc["title"] if doc else None,
            "source": item["path"] if item else None, "source_type": item["type"] if item else None}


def node_result(conn, row, score: float, why: str, **extra) -> dict:
    return {
        "type": "node", "id": row["id"], "kind": row["kind"], "title": row["title"],
        "text": row["text"], "role": row["role"], "abstraction": row["abstraction"],
        **_context(conn, row), "score": round(float(score), 4), "why": why, **extra,
    }


def wiki_result(page, score: float, why: str) -> dict:
    return {"type": "wiki", "id": page["id"], "kind": "wiki", "slug": page["slug"],
            "title": page["title"], "text": page["summary"], "score": round(float(score), 4), "why": why}


def image_result(row, score: float, why: str, **extra) -> dict:
    return {"type": "image", "id": row["id"], "kind": "image", "slug": row["slug"], "title": row["title"],
            "text": row["description"], "subjects": json.loads(row["subjects_json"] or "[]"),
            "url": f"/image-file/{row['slug']}", "page": f"/image-page/{row['slug']}",
            "credit": row["credit"], "source": row["source"], "creator": row["creator"], "license": row["license"],
            "score": round(float(score), 4), "why": why, **extra}


class Results:
    """Collects results in priority order, de-duplicated, up to k."""

    def __init__(self, conn, k: int, exclude: set = ()):
        self.conn, self.k, self.items, self.seen = conn, k, [], set(("node", i) for i in exclude)

    def full(self):
        return len(self.items) >= self.k

    def add_node(self, node_id, score, why, **extra):
        if self.full() or ("node", node_id) in self.seen:
            return
        row = db.node(self.conn, node_id)
        if row is None:
            return
        self.seen.add(("node", node_id))
        self.items.append(node_result(self.conn, row, score, why, **extra))

    def add_wiki(self, page, score, why):
        if self.full() or ("wiki", page["id"]) in self.seen:
            return
        self.seen.add(("wiki", page["id"]))
        self.items.append(wiki_result(page, score, why))

    def add_image(self, row, score, why, **extra):
        if self.full() or ("image", row["id"]) in self.seen:
            return
        self.seen.add(("image", row["id"]))
        self.items.append(image_result(row, score, why, **extra))


# --- anchor resolution -------------------------------------------------------------------

def resolve(conn, node_id: int | None = None, text: str | None = None):
    """Actions operate on a node. Free text is mapped to its closest chunk (the 'anchor')."""
    if node_id is not None:
        row = db.node(conn, node_id)
        if row is None:
            raise NotFound(f"node {node_id} not found")
        return row
    if not text:
        raise ValueError("give node_id or text")
    vec = llm.embed_one(text, query=True)
    hits = retrieve.hybrid(conn, text, vec, k=1, allowed=db.ids_of_kind(conn, "chunk"))
    if not hits:
        raise NotFound("database is empty")
    return db.node(conn, hits[0][0])


def _vec(conn, row):
    v = retrieve.index(conn, "node").get(row["id"])
    return v if v is not None else llm.embed_one(row["text"])


def _envelope(action, anchor, results: Results, **meta):
    return {"action": action,
            "anchor": {"id": anchor["id"], "kind": anchor["kind"], "title": anchor["title"],
                       "text": anchor["text"], "abstraction": anchor["abstraction"]},
            "results": results.items, "meta": meta}


# --- neighbours ------------------------------------------------------------------------

def neighbours(conn, node_id=None, text=None, k=None):
    k = k or config.DEFAULT_K
    if text and node_id is None:
        anchor = resolve(conn, text=text)
        vec, qtext, exclude = llm.embed_one(text, query=True), text, set()
    else:
        anchor = resolve(conn, node_id)
        vec, qtext, exclude = _vec(conn, anchor), anchor["text"], {anchor["id"]}
    chunks = db.ids_of_kind(conn, "chunk")
    cands = retrieve.hybrid(conn, qtext, vec, k=k * 4, allowed=chunks, exclude=exclude)
    picked = retrieve.mmr(conn, vec, [i for i, _ in cands], k)
    res = Results(conn, k, exclude)
    for i, sim in picked:
        res.add_node(i, sim, f"similar: hybrid dense+bm25, mmr (cos={sim:.2f})")
    return _envelope("neighbours", anchor, res)


# --- counter / evidence (claim graph) ---------------------------------------------------

def _claims_of(conn, node_id):
    return conn.execute("SELECT id, text FROM claims WHERE node_id=?", (node_id,)).fetchall()


def _claim_text(conn, cid):
    r = conn.execute("SELECT text FROM claims WHERE id=?", (cid,)).fetchone() if cid else None
    return r["text"] if r else None


STATEMENT_SCHEMA = {"type": "object", "properties": {"statement": {"type": "string"}}, "required": ["statement"]}
HYDE = {
    "counter": ("Claim: {c}\n\nWrite `statement`: one or two sentences arguing the OPPOSITE "
                "position — the strongest counter-argument a critic would make. Do not repeat the claim."),
    "evidence": ("Claim: {c}\n\nWrite `statement`: one or two sentences giving a specific fact, "
                 "case, project or research finding that would back up this claim (names, places, "
                 "numbers if plausible). Do not repeat or paraphrase the claim itself."),
}


def _hyde_fallback(conn, anchor, mode: str, res: Results, label: str):
    """HyDE-style fallback: have the LLM write a hypothetical counter/supporting statement,
    search claims near *that*, then keep only candidates NLI agrees with."""
    claims = _claims_of(conn, anchor["id"])
    targets = [c["text"] for c in claims][:2] or [anchor["text"][:500]]
    own = {c["id"] for c in claims}
    cidx = retrieve.index(conn, "claim")
    for target in targets:
        if res.full():
            return
        hypo = llm.generate(HYDE[mode].format(c=target), schema=STATEMENT_SCHEMA, temperature=0.4)["statement"]
        hits = cidx.search(llm.embed_one(hypo), k=20, exclude=own)
        if not hits:
            continue
        rows = {r["id"]: r for r in conn.execute(
            f"SELECT id, node_id, text FROM claims WHERE id IN ({','.join('?' * len(hits))})",
            [h for h, _ in hits])}
        cands = [rows[h] for h, _ in hits if h in rows and rows[h]["node_id"] != anchor["id"]]
        # premise = candidate, hypothesis = anchor claim
        scores = nli.score([(c["text"], target) for c in cands])
        key = "contradiction" if mode == "counter" else "entailment"
        ranked = sorted(zip(cands, scores), key=lambda x: -x[1][key])
        for c, s in ranked:
            if s[key] < config.FALLBACK_NLI_THRESHOLD:
                break
            res.add_node(c["node_id"], s[key], f"{label}: hyde+nli {key}={s[key]:.2f}",
                         claim=c["text"], against=target)


def counter(conn, node_id=None, text=None, k=None, fallback=True):
    k = k or config.DEFAULT_K
    anchor = resolve(conn, node_id, text)
    res = Results(conn, k, {anchor["id"]})
    for e in conn.execute("SELECT * FROM edges WHERE src_node=? AND type='CONTRADICTS' ORDER BY weight DESC",
                          (anchor["id"],)):
        res.add_node(e["dst_node"], e["weight"], f"edge:CONTRADICTS {e['method']}={e['weight']:.2f}",
                     claim=_claim_text(conn, e["dst_claim"]), against=_claim_text(conn, e["src_claim"]))
    used_fallback = False
    if fallback and not res.full() and anchor["kind"] == "chunk":
        used_fallback = True
        try:
            _hyde_fallback(conn, anchor, "counter", res, "counter")
        except llm.LLMError as e:  # the graph results are still useful without the fallback
            return _envelope("counter", anchor, res, fallback_used=True, fallback_error=str(e))
    return _envelope("counter", anchor, res, fallback_used=used_fallback)


def evidence(conn, node_id=None, text=None, k=None, fallback=True):
    k = k or config.DEFAULT_K
    anchor = resolve(conn, node_id, text)
    res = Results(conn, k, {anchor["id"]})
    edges = conn.execute(
        "SELECT e.*, n.role FROM edges e JOIN nodes n ON n.id=e.src_node "
        "WHERE e.dst_node=? AND e.type IN ('SUPPORTS','EXAMPLE_OF')", (anchor["id"],)).fetchall()
    ranked = sorted(edges, key=lambda e: -e["weight"] * ROLE_BOOST.get(e["role"], 1.0))
    for e in ranked:
        res.add_node(e["src_node"], e["weight"] * ROLE_BOOST.get(e["role"], 1.0),
                     f"edge:{e['type']} {e['method']}={e['weight']:.2f} role={e['role']}",
                     claim=_claim_text(conn, e["src_claim"]), supports=_claim_text(conn, e["dst_claim"]))
    used_fallback = False
    if fallback and not res.full() and anchor["kind"] == "chunk":
        used_fallback = True
        try:
            _hyde_fallback(conn, anchor, "evidence", res, "evidence")
        except llm.LLMError as e:  # the graph results are still useful without the fallback
            return _envelope("evidence", anchor, res, fallback_used=True, fallback_error=str(e))
    return _envelope("evidence", anchor, res, fallback_used=used_fallback)


# --- zoom (hierarchy + abstraction) ---------------------------------------------------------

def _same_cluster_by_abstraction(conn, anchor, res: Results, direction: str):
    """Chunks in the anchor's cluster that are more general (out) / more specific (in)."""
    if not anchor["cluster_id"] or anchor["abstraction"] is None:
        return
    op = ">" if direction == "out" else "<"
    rows = conn.execute(
        f"SELECT id, abstraction FROM nodes WHERE cluster_id=? AND id!=? AND abstraction {op} ?",
        (anchor["cluster_id"], anchor["id"], anchor["abstraction"])).fetchall()
    ranked = sorted(rows, key=lambda r: -retrieve.cosine(conn, anchor["id"], r["id"]))
    for r in ranked:
        sim = retrieve.cosine(conn, anchor["id"], r["id"])
        res.add_node(r["id"], sim, f"same cluster, abstraction {r['abstraction']:.0f} "
                                   f"{op} {anchor['abstraction']:.0f} (cos={sim:.2f})")


def _children_by_similarity(conn, parent, rows, res: Results, why: str):
    for r in sorted(rows, key=lambda r: -retrieve.cosine(conn, parent["id"], r["id"])):
        res.add_node(r["id"], retrieve.cosine(conn, parent["id"], r["id"]), why)


def zoom_in(conn, node_id=None, text=None, k=None):
    k = k or config.DEFAULT_K
    anchor = resolve(conn, node_id, text)
    res = Results(conn, k, {anchor["id"]})
    kind = anchor["kind"]
    if kind == "theme":
        rows = conn.execute("SELECT id FROM nodes WHERE kind='cluster' AND parent_id=?", (anchor["id"],)).fetchall()
        _children_by_similarity(conn, anchor, rows, res, "tree: theme → cluster")
    elif kind == "cluster":
        rows = conn.execute("SELECT id FROM nodes WHERE cluster_id=?", (anchor["id"],)).fetchall()
        _children_by_similarity(conn, anchor, rows, res, "tree: cluster → chunk")
    elif kind in ("document", "section"):
        rows = conn.execute("SELECT id FROM nodes WHERE parent_id=?", (anchor["id"],)).fetchall()
        _children_by_similarity(conn, anchor, rows, res, f"tree: {kind} → child")
    else:  # chunk: graph edges first, then more-specific chunks on the same topic
        for e in conn.execute(
                "SELECT * FROM edges WHERE dst_node=? AND type IN ('ELABORATES','EXAMPLE_OF') "
                "ORDER BY weight DESC", (anchor["id"],)):
            res.add_node(e["src_node"], e["weight"], f"edge:{e['type']} {e['method']}",
                         claim=_claim_text(conn, e["src_claim"]))
        _same_cluster_by_abstraction(conn, anchor, res, "in")
        if not res.full() and anchor["abstraction"] is not None:
            # widen: nearest chunks anywhere that are more specific
            vec = _vec(conn, anchor)
            for i, sim in retrieve.index(conn, "node").search(vec, k=40, allowed=db.ids_of_kind(conn, "chunk"),
                                                               exclude={anchor["id"]}):
                r = db.node(conn, i)
                if r["abstraction"] is not None and r["abstraction"] < anchor["abstraction"]:
                    res.add_node(i, sim, f"nearby, more specific (abstraction {r['abstraction']:.0f}, cos={sim:.2f})")
    return _envelope("zoom_in", anchor, res)


def zoom_out(conn, node_id=None, text=None, k=None):
    k = k or config.DEFAULT_K
    anchor = resolve(conn, node_id, text)
    res = Results(conn, k, {anchor["id"]})
    kind = anchor["kind"]
    if kind == "chunk":
        # semantic tree first (cross-document idea), then structural context, then wiki
        if anchor["cluster_id"]:
            cl = db.node(conn, anchor["cluster_id"])
            res.add_node(cl["id"], retrieve.cosine(conn, anchor["id"], cl["id"]), "tree: chunk → cluster")
            if cl["parent_id"]:
                res.add_node(cl["parent_id"], retrieve.cosine(conn, anchor["id"], cl["parent_id"]),
                             "tree: cluster → theme")
        for e in conn.execute("SELECT * FROM edges WHERE src_node=? AND type IN ('ELABORATES','EXAMPLE_OF') "
                              "ORDER BY weight DESC", (anchor["id"],)):
            res.add_node(e["dst_node"], e["weight"], f"edge:{e['type']} (this is the specific case)",
                         claim=_claim_text(conn, e["dst_claim"]))
        res.add_node(anchor["parent_id"], retrieve.cosine(conn, anchor["id"], anchor["parent_id"]),
                     "tree: chunk → section")
        for p in conn.execute("SELECT * FROM wiki_pages"):
            if anchor["id"] in json.loads(p["node_ids_json"]):
                res.add_wiki(p, 1.0, "wiki page covering this chunk")
        _same_cluster_by_abstraction(conn, anchor, res, "out")
    elif kind == "section":
        res.add_node(anchor["parent_id"], 1.0, "tree: section → document")
        # the clusters most of this section's chunks fall into
        for r in conn.execute("SELECT cluster_id, count(*) n FROM nodes WHERE parent_id=? AND cluster_id "
                              "IS NOT NULL GROUP BY cluster_id ORDER BY n DESC", (anchor["id"],)):
            res.add_node(r["cluster_id"], r["n"], f"tree: {r['n']} of this section's chunks are in this cluster")
    elif kind == "cluster":
        if anchor["parent_id"]:
            res.add_node(anchor["parent_id"], 1.0, "tree: cluster → theme")
        for e in conn.execute("SELECT * FROM edges WHERE src_node=? AND type='CONTRADICTS'", (anchor["id"],)):
            res.add_node(e["dst_node"], e["weight"], "wiki lint: a conflicting cluster")
    elif kind == "theme":
        # nothing above a theme: offer the other themes as the wider landscape
        for r in conn.execute("SELECT id FROM nodes WHERE kind='theme' AND id!=?", (anchor["id"],)):
            res.add_node(r["id"], retrieve.cosine(conn, anchor["id"], r["id"]), "sibling theme (top of tree)")
    return _envelope("zoom_out", anchor, res)


# --- images (subject + metaphor) -----------------------------------------------------------

CONCEPT_PROMPT = """Passage:
{text}

What is this passage about at the level of IDEAS, not objects? Give `concepts`: {n} short
phrases (4-10 words each) naming the abstract ideas, relationships, processes or tensions it
deals with, e.g. "many small parts acting together without a centre" or "knowledge handed down
between generations". Do not name the concrete things in the passage (technologies,
institutions, places, people); say what kind of situation they are in."""
CONCEPT_SCHEMA = {"type": "object", "properties": {"concepts": {"type": "array", "items": {"type": "string"}}},
                  "required": ["concepts"]}
NEAR_DUPLICATE = 0.95  # images whose descriptions are this similar count as the same picture


def llm_concepts(text: str) -> list[str]:
    out = llm.generate(CONCEPT_PROMPT.format(text=text[:3000], n=config.IMAGE_CONCEPTS),
                       schema=CONCEPT_SCHEMA, temperature=0.3)
    return [c.strip() for c in out["concepts"] if c.strip()][:config.IMAGE_CONCEPTS]


def cache_concepts(conn, text: str, concepts: list[str]):
    conn.execute("INSERT OR REPLACE INTO concept_cache(text, model, concepts_json) VALUES (?,?,?)",
                 (text, config.LLM_MODEL, json.dumps(concepts)))
    conn.commit()


def _tree_concepts(conn, anchor) -> list[str]:
    """Without the LLM: the anchor's cluster and theme titles (already abstract), then its claims."""
    out = []
    if anchor["kind"] in ("cluster", "theme"):
        out.append(anchor["title"])
    cl = db.node(conn, anchor["cluster_id"]) if anchor["cluster_id"] else None
    if cl is not None:
        out.append(cl["title"])
        th = db.node(conn, cl["parent_id"]) if cl["parent_id"] else None
        if th is not None:
            out.append(th["title"])
    out += [c["text"] for c in _claims_of(conn, anchor["id"])]
    return [c for c in out if c][:config.IMAGE_CONCEPTS] or [anchor["text"][:300]]


def passage_concepts(conn, anchor, text: str) -> tuple[list[str], str]:
    """Abstract concepts of a passage (cached), and where they came from: 'llm' or 'tree'."""
    row = conn.execute("SELECT concepts_json FROM concept_cache WHERE text=? AND model=?",
                       (text, config.LLM_MODEL)).fetchone()
    if row:
        return json.loads(row["concepts_json"]), "llm"
    try:
        concepts = llm_concepts(text)
    except llm.LLMError:
        return _tree_concepts(conn, anchor), "tree"
    if not concepts:
        return _tree_concepts(conn, anchor), "tree"
    cache_concepts(conn, text, concepts)
    return concepts, "llm"


def _sims(conn, owner: str, qvecs, centre: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Similarity of every `owner` vector to each query vector: (ids, matrix [vectors x queries]).

    With `centre`, each vector's mean cosine to the corpus passages is subtracted (hubness
    correction). Without it, an image whose text is vaguely like everything (e.g. "city lights
    of the world": networks, scale, control) tops every list. The mean cosine to all passages
    equals v · c, with c the mean of the passage vectors, so this is one dot product per image."""
    idx = retrieve.index(conn, owner)
    if len(idx.ids) == 0 or not len(qvecs):
        return idx.ids, np.zeros((len(idx.ids), len(qvecs)), dtype=np.float32)
    Q = np.asarray(qvecs, dtype=np.float32).reshape(len(qvecs), -1)
    Q = Q / np.clip(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9, None)
    S = idx.mat @ Q.T
    if centre:
        S = S - (idx.mat @ _passage_centroid(conn))[:, None]
    return idx.ids, S


_centroid = {"version": None, "vec": None}


def _passage_centroid(conn) -> np.ndarray:
    nidx = retrieve.index(conn, "node")
    if _centroid["version"] != nidx.version:
        chunks = db.ids_of_kind(conn, "chunk")
        rows = [nidx.get(i) for i in chunks if nidx.get(i) is not None]
        _centroid["vec"] = np.mean(rows, axis=0) if rows else np.zeros(config.EMBED_DIM, dtype=np.float32)
        _centroid["version"] = nidx.version
    return _centroid["vec"]


def subject_ranking(conn, vec, desc=True, visual=True, centre=True) -> list[tuple[int, float, str]]:
    """Images by what they show: the passage vector against each image's description and (gemma
    only) the picture itself. The two lists are fused by rank, since image-text and text-text
    cosines sit on different scales. Returns [(image_id, score, why)] best first."""
    def cosines(owner):
        ids, S = _sims(conn, owner, [vec], centre)
        return dict(zip(ids.tolist(), S[:, 0].tolist()))

    d = cosines("image_desc") if desc else {}
    v = cosines("image_visual") if visual else {}
    lists = [sorted(x.items(), key=lambda kv: -kv[1]) for x in (d, v) if x]
    fused = retrieve.rrf(*lists) if lists else {}
    out = []
    for i in sorted(fused, key=lambda i: -fused[i]):
        parts = ([f"description {d[i]:+.2f}"] if i in d else []) + ([f"picture {v[i]:+.2f}"] if i in v else [])
        out.append((int(i), d.get(i, v.get(i, 0.0)), "subject: " + ", ".join(parts)
                    + (" (cosine minus corpus mean)" if centre else " (cosine)")))
    return out


def metaphor_ranking(conn, concepts: list[str], qvecs=None, centre=True) -> list[tuple[int, float, str, dict]]:
    """Images by what they can stand for: each passage concept against each metaphor line on the
    image pages; an image scores its best (concept, metaphor) pair. Returns
    [(image_id, score, why, extra)] best first."""
    if qvecs is None:
        qvecs = [llm.embed_one(c, query=True) for c in concepts]
    ids, S = _sims(conn, "image_metaphor", qvecs, centre)
    if not len(ids):
        return []
    meta = {r["id"]: r for r in conn.execute("SELECT * FROM image_metaphors")}
    best: dict[int, tuple] = {}
    for p, mid in enumerate(ids.tolist()):
        m = meta.get(mid)
        if m is None:
            continue
        q = int(np.argmax(S[p]))
        s = float(S[p, q])
        if m["image_id"] not in best or s > best[m["image_id"]][0]:
            best[m["image_id"]] = (s, m, concepts[q])
    out = []
    for iid, (s, m, c) in sorted(best.items(), key=lambda kv: -kv[1][0]):
        out.append((iid, s, f"metaphor: “{m['keyword']}” ↔ concept “{c}” ({s:+.2f}{' over corpus mean' if centre else ' cosine'})",
                    {"metaphor": {"keyword": m["keyword"], "gloss": m["gloss"]}, "concept": c}))
    return out


def images(conn, node_id=None, text=None, k=None):
    """Images for a passage: by subject (what they show) and by metaphor (what they can stand for)."""
    k = k or config.DEFAULT_K
    if text and node_id is None:
        anchor = resolve(conn, text=text)
        vec, src = llm.embed_one(text, query=True), text
    else:
        anchor = resolve(conn, node_id)
        vec, src = _vec(conn, anchor), anchor["text"]
    rows = {r["id"]: r for r in conn.execute("SELECT * FROM images")}
    res = Results(conn, k)
    if not rows:
        return _envelope("images", anchor, res,
                         note="the image library is empty: add files to IMAGE_DIR and run stage 7")
    concepts, concepts_from = passage_concepts(conn, anchor, src)
    routes = {"subject": iter((i, s, w, {}) for i, s, w in subject_ranking(conn, vec)),
              "metaphor": iter(metaphor_ranking(conn, concepts))}
    didx, picked = retrieve.index(conn, "image_desc"), []
    while not res.full() and routes:
        for match in list(routes):  # alternate subject, metaphor, subject, ...
            for iid, score, why, extra in routes[match]:
                if ("image", iid) in res.seen or iid not in rows:
                    continue
                v = didx.get(iid)
                if v is not None and any(float(v @ p) > NEAR_DUPLICATE for p in picked):
                    continue
                res.add_image(rows[iid], score, why, match=match, **extra)
                if v is not None:
                    picked.append(v)
                break
            else:
                del routes[match]
            if res.full():
                break
    return _envelope("images", anchor, res, concepts=concepts, concepts_from=concepts_from)


# Reverse search (image → passages): how much of each passage's mean cosine to the image library
# to subtract. Without it, short junk passages (a citation, "(This is more like it!)") top the
# lists for many images. On the 24-passage test set, full correction cost the metaphor route
# accuracy (expected passage in the top 10: 12% raw, 5% full) while 0.25 kept 12% and still
# capped any passage at 4 of the 26 images' top 5s. Subject accuracy was the same either way.
REVERSE_CENTRE = {"subject": 1.0, "metaphor": 0.25}


def _centred_hits(conn, owner: str, qvecs, query_owner: str, ids=None,
                  strength: float = 1.0) -> dict[int, tuple[float, int]]:
    """Reverse search: rows of `owner` (passages or claims) against query vectors from the image
    library. Hub correction from the other side: each row's score has `strength` × its mean cosine
    to all `query_owner` vectors subtracted. Returns {row id: (best score, index of best query)}."""
    idx, qidx = retrieve.index(conn, owner), retrieve.index(conn, query_owner)
    if len(idx.ids) == 0 or not len(qvecs) or len(qidx.ids) == 0:
        return {}
    Q = np.asarray(qvecs, dtype=np.float32).reshape(len(qvecs), -1)
    Q = Q / np.clip(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9, None)
    S = idx.mat @ Q.T - strength * (idx.mat @ qidx.mat.mean(axis=0))[:, None]
    best_q = S.argmax(axis=1)
    keep = None if ids is None else np.isin(idx.ids, list(ids))
    return {int(i): (float(S[p, best_q[p]]), int(best_q[p])) for p, i in enumerate(idx.ids.tolist())
            if keep is None or keep[p]}


def image_passages(conn, slug: str, k=None, metaphor_via: str = "claims"):
    """The reverse of Images: passages a picture fits, by subject (its description and pixels vs
    each passage) and by metaphor (its metaphor lines vs each passage's claims, or vs the passages
    themselves with metaphor_via='passages'). Interleaved like Images, subject first."""
    k = k or config.DEFAULT_K
    img = conn.execute("SELECT * FROM images WHERE slug=?", (slug,)).fetchone()
    if img is None:
        raise NotFound(f"no image {slug!r}")
    chunks = db.ids_of_kind(conn, "chunk")
    # subject: description and picture, fused by rank
    lists, cos = [], {}
    for owner in ("image_desc", "image_visual"):
        v = retrieve.index(conn, owner).get(img["id"])
        if v is None:
            continue
        hits = _centred_hits(conn, "node", [v], owner, ids=chunks, strength=REVERSE_CENTRE["subject"])
        lists.append(sorted(((i, s) for i, (s, _) in hits.items()), key=lambda x: -x[1])[:200])
        cos[owner] = {i: s for i, (s, _) in hits.items()}
    fused = retrieve.rrf(*lists) if lists else {}
    subject = []
    for i in sorted(fused, key=lambda i: -fused[i]):
        parts = [f"{n} {cos[o][i]:+.2f}" for o, n in (("image_desc", "description"), ("image_visual", "picture"))
                 if o in cos and i in cos[o]]
        subject.append((i, cos.get("image_desc", cos.get("image_visual", {})).get(i, 0.0),
                        "subject: " + ", ".join(parts) + " (cosine minus the passage's mean to all images)", {}))
    # metaphor: each metaphor line vs claims (mapped to their passage) or vs passages
    metas = conn.execute("SELECT * FROM image_metaphors WHERE image_id=? ORDER BY id", (img["id"],)).fetchall()
    midx = retrieve.index(conn, "image_metaphor")
    metas = [m for m in metas if midx.get(m["id"]) is not None]
    best: dict[int, tuple] = {}
    if metas:
        qv = [midx.get(m["id"]) for m in metas]
        if metaphor_via == "claims":
            claim_node = dict(conn.execute("SELECT id, node_id FROM claims").fetchall())
            for cid, (s, q) in _centred_hits(conn, "claim", qv, "image_metaphor",
                                             strength=REVERSE_CENTRE["metaphor"]).items():
                nid = claim_node.get(cid)
                if nid in chunks and (nid not in best or s > best[nid][0]):
                    best[nid] = (s, metas[q], cid)
            claim_text = {r["id"]: r["text"] for r in conn.execute(
                f"SELECT id, text FROM claims WHERE id IN ({','.join('?' * len(best))})",
                [c for *_, c in best.values()])} if best else {}
        else:
            for nid, (s, q) in _centred_hits(conn, "node", qv, "image_metaphor", ids=chunks,
                                             strength=REVERSE_CENTRE["metaphor"]).items():
                best[nid] = (s, metas[q], None)
    metaphor = []
    for nid, (s, m, cid) in sorted(best.items(), key=lambda kv: -kv[1][0]):
        extra = {"metaphor": {"keyword": m["keyword"], "gloss": m["gloss"]}}
        via = ""
        if cid is not None:
            extra["claim"] = claim_text.get(cid)
            via = " via its claim"
        metaphor.append((nid, s, f"metaphor: “{m['keyword']}”{via} ({s:+.2f}, partly corrected for the passage's mean)",
                         extra))
    res = Results(conn, k)
    routes = {"subject": iter(subject), "metaphor": iter(metaphor)}
    while not res.full() and routes:
        for match in list(routes):
            for nid, score, why, extra in routes[match]:
                if ("node", nid) in res.seen:
                    continue
                res.add_node(nid, score, why, match=match, **extra)
                break
            else:
                del routes[match]
            if res.full():
                break
    return {"action": "image-passages",
            "anchor": image_result(img, 1.0, "the image you started from"),
            "results": res.items, "meta": {"metaphor_via": metaphor_via}}


# --- question (wiki-first RAG) ------------------------------------------------------------

QUESTION_PROMPT = """Answer the question using ONLY the material below. Cite passages inline by
their id in square brackets, e.g. [42]. If the material does not contain the answer, say so.
{focus}
WIKI SUMMARIES:
{wiki}

PASSAGES:
{passages}

QUESTION: {question}

Reply as JSON with `answer` (a few sentences in your own words, citing ids inline like [42])
and `found` (false if the material does not answer the question).
"""
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "found": {"type": "boolean"}},
    "required": ["answer", "found"],
}


def question(conn, question: str, node_id: int | None = None, k: int = 8):
    qvec = llm.embed_one(question, query=True)
    focus = db.node(conn, node_id) if node_id is not None else None
    if node_id is not None and focus is None:
        raise NotFound(f"node {node_id} not found")

    pages = retrieve.index(conn, "wiki").search(qvec, k=3)
    page_rows = {r["id"]: r for r in conn.execute("SELECT * FROM wiki_pages")}
    chunks = db.ids_of_kind(conn, "chunk")
    from_pages = set()
    for pid, _ in pages:
        from_pages |= set(json.loads(page_rows[pid]["node_ids_json"]))
    nidx = retrieve.index(conn, "node")
    cand = dict(retrieve.hybrid(conn, question, qvec, k=k, allowed=chunks))
    for i, sim in nidx.search(qvec, k=k, allowed=from_pages & chunks):
        cand[i] = cand.get(i, 0) + sim / 10
    passage_ids = [i for i, _ in sorted(cand.items(), key=lambda x: -x[1])][:k]
    if focus is not None and focus["kind"] == "chunk" and focus["id"] not in passage_ids:
        passage_ids = [focus["id"]] + passage_ids[: k - 1]

    rows = {i: db.node(conn, i) for i in passage_ids}
    prompt = QUESTION_PROMPT.format(
        focus=(f"\nThe reader is looking at passage [{focus['id']}]: {focus['text']}\n" if focus is not None else ""),
        wiki="\n".join(f"- {page_rows[p]['title']}: {page_rows[p]['summary']}" for p, _ in pages) or "(none)",
        passages="\n\n".join(f"[{i}] {rows[i]['text']}" for i in passage_ids),
        question=question,
    )
    out = llm.generate(prompt, schema=ANSWER_SCHEMA, temperature=0.2)
    answer = out["answer"].strip()
    cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer) if int(x) in rows})
    return {
        "action": "question", "question": question, "answer": answer, "found": out["found"], "cited": cited,
        "passages": [node_result(conn, rows[i], cand.get(i, 0), "retrieved: wiki-routed + hybrid")
                     for i in passage_ids],
        "wiki": [wiki_result(page_rows[p], s, "wiki page matched the question") for p, s in pages],
    }


def search(conn, q: str, k: int = 10, kinds=("chunk",)):
    vec = llm.embed_one(q, query=True)
    hits = retrieve.hybrid(conn, q, vec, k=k, allowed=db.ids_of_kind(conn, *kinds))
    return [node_result(conn, db.node(conn, i), s, "hybrid dense+bm25 (rrf)") for i, s in hits]


ACTIONS = {
    "neighbours": neighbours,
    "counter": counter,
    "evidence": evidence,
    "zoom-in": zoom_in,
    "zoom-out": zoom_out,
    "images": images,
}
