"""Stage 3 — per-chunk annotation with the local LLM, plus section/document summaries.

For every chunk without one, first a context prefix (contextual retrieval): 1-2 sentences
situating the chunk in its document, prepended to the chunk for embedding and keyword search only
(the stored and displayed text is unchanged). Then, for every chunk not yet annotated:
  claims       1-3 atomic, self-contained propositions (the units the claim graph links)
  role         rhetorical function (claim / evidence / example / definition / ...)
  abstraction  1 (concrete detail) .. 5 (broad thesis) — drives zoom-in / zoom-out
  topics       2-4 tags (reused across chunks so the wiki can group them)

Sections and documents get an LLM summary + abstraction score so they can be returned as
"zoom-out" results. Only un-annotated nodes are processed, so this stage is resumable.
"""
import collections
import json

from dip import config, db, llm
from dip.ingest.common import (ABSTRACTION_SCALE, abstraction_score, embed_missing_claims,
                               embed_missing_nodes, log, progress, rebuild_fts)

ROLES = ["claim", "evidence", "example", "definition", "provocation", "question", "caption", "link"]
SYSTEM = "You index passages of a research corpus for a retrieval system. Be precise and terse."

CHUNK_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "role": {"type": "string", "enum": ROLES},
        "abstraction": {"type": "integer", "minimum": 1, "maximum": 5},
        "topics": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4},
    },
    "required": ["claims", "role", "abstraction", "topics"],
}

CHUNK_PROMPT = """Analyse this passage from "{doc}", section "{section}".

PASSAGE:
{text}

Return JSON with:
- claims: 1-3 short statements of what the passage asserts. Each must make sense on its own
  (replace pronouns with names, no "the author says"). If it asserts nothing (e.g. a list of
  links), return [].
- role: the passage's main function. claim = an argument or position; evidence = facts, data or
  research findings; example = a concrete case, project or story; definition; provocation = a
  challenging question or quote; question; caption = describes an image; link = navigation/links.
- """ + ABSTRACTION_SCALE + """
- topics: 2-4 short lowercase tags naming what this passage is specifically about (e.g.
  "data centres", "indigenous knowledge", "community governance"). Never use generic tags that
  fit the whole corpus, such as "ai" or "technology". If one of these existing tags names the
  same concept, reuse its exact wording: {topics}
"""

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "abstraction": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["summary", "abstraction"],
}

SUMMARY_PROMPT = """Summarise this {kind} titled "{title}".

{body}

Return JSON with:
- summary: 2-4 sentences stating its main argument rather than listing contents.
- """ + ABSTRACTION_SCALE + """
"""


CONTEXT_PROMPT = """<document title="{doc}">
{body}
</document>

Here is a passage from this document:
<passage>
{text}
</passage>

Give a short, succinct context (1-2 sentences, under 50 words) to situate this passage within the overall
document, for the purposes of improving search retrieval of the passage: what the document is
and which part of it the passage comes from, and who or what the passage refers to if it doesn't
say. Answer only with the context, not a summary of the passage."""


def _context_body(chunks: list, target, limit: int) -> str:
    """The document as the LLM sees it for one chunk: all of it if it fits in `limit` chars
    (the same prompt prefix for every chunk, which Ollama's prompt cache reuses), otherwise an
    outline of its sections plus the chunk's own section, cut to a window around the chunk."""
    sections = {}  # section id -> its chunks, in document order
    for c in chunks:
        sections.setdefault(c["parent_id"], []).append(c)
    whole = "\n\n".join(f"## {cs[0]['title']}\n\n" + "\n\n".join(c["text"] for c in cs)
                         for cs in sections.values())
    if len(whole) <= limit:
        return whole
    outline = "Sections: " + "; ".join(cs[0]["title"] for cs in sections.values())
    sec = sections[target["parent_id"]]
    i = next(n for n, c in enumerate(sec) if c["id"] == target["id"])
    lo, hi, size = i, i + 1, len(outline) + len(target["text"])
    while lo > 0 or hi < len(sec):  # grow the window both ways until the budget is spent
        grown = False
        for side in (-1, 1):
            j = lo - 1 if side < 0 else hi
            if 0 <= j < len(sec) and size + len(sec[j]["text"]) <= limit:
                size += len(sec[j]["text"])
                lo, hi = (j, hi) if side < 0 else (lo, j + 1)
                grown = True
        if not grown:
            break
    text = "\n\n".join(c["text"] for c in sec[lo:hi])
    return f"{outline}\n\n## {target['title']}\n\n" + ("[…]\n\n" if lo else "") + text[:limit]


def contextualise_chunks(conn) -> int:
    """Write the context prefix of every chunk that has none yet, and drop its vector so it is
    re-embedded with the prefix. Documents with a single chunk (e.g. images) get an empty one."""
    todo = conn.execute("SELECT DISTINCT item_id FROM nodes WHERE kind='chunk' AND context IS NULL").fetchall()
    total = conn.execute("SELECT count(*) FROM nodes WHERE kind='chunk' AND context IS NULL").fetchone()[0]
    done = 0
    for item in todo:
        doc = conn.execute("SELECT title FROM nodes WHERE item_id=? AND kind='document'",
                           (item["item_id"],)).fetchone()
        chunks = conn.execute(
            "SELECT n.* FROM nodes n JOIN nodes s ON s.id = n.parent_id "
            "WHERE n.item_id=? AND n.kind='chunk' ORDER BY s.ord, n.ord", (item["item_id"],)).fetchall()
        for c in chunks:
            if c["context"] is not None:
                continue
            if len(chunks) == 1:
                context = ""
            else:
                body = _context_body(chunks, c, config.CONTEXT_DOC_CHARS)
                try:
                    context = llm.generate(CONTEXT_PROMPT.format(doc=doc["title"] if doc else "", body=body,
                                                                 text=c["text"]), system=SYSTEM)
                except llm.LLMError as e:
                    log(f"  chunk {c['id']}: {e}; no context (will retry next run)")
                    continue
            with db.tx(conn):
                conn.execute("UPDATE nodes SET context=? WHERE id=?", (" ".join(context.split()), c["id"]))
                conn.execute("DELETE FROM vectors WHERE owner='node' AND owner_id=?", (c["id"],))
            done += 1
            progress(done, total, "contextualised chunks")
    return done


def _top_topics(conn, n=40) -> list[str]:
    counts = collections.Counter()
    for r in conn.execute("SELECT topics_json FROM nodes WHERE topics_json IS NOT NULL"):
        counts.update(json.loads(r["topics_json"]))
    return [t for t, _ in counts.most_common(n)]


def annotate_chunks(conn):
    rows = conn.execute(
        "SELECT n.*, d.title AS doc_title FROM nodes n "
        "JOIN nodes s ON s.id = n.parent_id JOIN nodes d ON d.id = s.parent_id "
        "WHERE n.kind='chunk' AND n.abstraction IS NULL ORDER BY n.item_id, n.ord"
    ).fetchall()
    topics = _top_topics(conn)
    for i, r in enumerate(rows, 1):
        try:
            out, lp = llm.generate(
                CHUNK_PROMPT.format(doc=r["doc_title"], section=r["title"], text=r["text"],
                                    topics=", ".join(topics) or "(none yet)"),
                schema=CHUNK_SCHEMA, system=SYSTEM, logprobs=True,
            )
        except llm.LLMError as e:
            log(f"  chunk {r['id']}: {e}; skipped (will retry next run)")
            continue
        role = r["role"] or out["role"]  # structural pre-tags (definition/caption) win
        tags = sorted({t.strip().lower() for t in out["topics"] if t.strip()})
        claims = [c.strip() for c in out["claims"] if len(c.strip()) > 10]
        if role == "link":
            claims = []
        with db.tx(conn):
            conn.execute(
                "UPDATE nodes SET role=?, abstraction=?, topics_json=? WHERE id=?",
                (role, abstraction_score(out, lp), json.dumps(tags), r["id"]),
            )
            conn.execute("DELETE FROM claims WHERE node_id=?", (r["id"],))
            conn.executemany("INSERT INTO claims(node_id, text) VALUES (?,?)",
                             [(r["id"], c) for c in claims])
        for t in tags:
            if t not in topics:
                topics.append(t)
        progress(i, len(rows), "annotated chunks")
    return len(rows)


def summarise(conn, kind: str, child_kind: str):
    """Summarise sections from their chunks, documents from their section summaries."""
    rows = conn.execute(
        "SELECT * FROM nodes WHERE kind=? AND abstraction IS NULL", (kind,)
    ).fetchall()
    for i, r in enumerate(rows, 1):
        children = conn.execute(
            "SELECT title, text FROM nodes WHERE parent_id=? AND kind=? ORDER BY ord",
            (r["id"], child_kind),
        ).fetchall()
        body = "\n\n".join(
            (f"## {c['title']}\n{c['text']}" if kind == "document" else c["text"]) for c in children
        )[:7000]
        try:
            out, lp = llm.generate(SUMMARY_PROMPT.format(kind=kind, title=r["title"], body=body),
                                   schema=SUMMARY_SCHEMA, system=SYSTEM, logprobs=True)
        except llm.LLMError as e:
            log(f"  {kind} {r['id']}: {e}; skipped")
            continue
        with db.tx(conn):
            conn.execute("UPDATE nodes SET text=?, abstraction=? WHERE id=?",
                         (out["summary"].strip(), abstraction_score(out, lp), r["id"]))
            conn.execute("DELETE FROM vectors WHERE owner='node' AND owner_id=?", (r["id"],))
        progress(i, len(rows), f"summarised {kind}s", every=5)


def run(conn):
    if config.CHUNK_CONTEXT:
        contextualise_chunks(conn)
    n = annotate_chunks(conn)
    summarise(conn, "section", "chunk")
    summarise(conn, "document", "section")
    c = embed_missing_claims(conn)
    embed_missing_nodes(conn)
    rebuild_fts(conn)  # section/document text changed to summaries
    log(f"stage 3: annotated {n} chunks, embedded {c} claims")
