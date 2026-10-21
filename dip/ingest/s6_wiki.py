"""Stage 6 — L4 LLM-Wiki (after Karpathy's "LLM Wiki" pattern).

The LLM writes a markdown page per cluster and per theme into data/wiki/, citing the chunks it
draws on as [[chunk:ID]]. The wiki is a human-readable, editable layer over the index:
  - zoom-out returns the pages that cite a chunk
  - question answers search pages first, then pull their cited chunks
  - index.md lists every page, log.md records each build

Deterministic parts (Tensions, Sources, Related) are generated from the DB, not the LLM, so
they are always accurate. A page whose front matter contains `locked: true` is never
overwritten — use that for hand-edited pages.

`lint()` reports near-duplicate pages, pages with no citations and cross-page tensions, and
asks the LLM whether very similar pages contradict each other (-> wiki_lint edges).
"""
import datetime as dt
import json
import re

import numpy as np

from dip import config, db, llm
from dip.ingest.common import log, progress

PAGE_PROMPT = """Theme: "{title}"
Theme summary: {summary}

PASSAGES (each starts with its id in square brackets):
{passages}

Using ONLY these passages, write the content of a short wiki page about this theme for a
research team, as JSON:
- overview: one paragraph (3-5 sentences) explaining the theme in your own words.
- key_points: 3-6 points; each has `point` (one sentence) and `ids` (the ids of the passages it
  comes from, as numbers).
- open_questions: 1-3 questions the passages raise but don't answer.
"""

PAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "overview": {"type": "string"},
        "key_points": {"type": "array", "minItems": 1, "maxItems": 6, "items": {
            "type": "object",
            "properties": {"point": {"type": "string"},
                           "ids": {"type": "array", "items": {"type": "integer"}}},
            "required": ["point", "ids"]}},
        "open_questions": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
    },
    "required": ["overview", "key_points", "open_questions"],
}

LINT_SCHEMA = {
    "type": "object",
    "properties": {"conflict": {"type": "boolean"}, "explanation": {"type": "string"}},
    "required": ["conflict", "explanation"],
}


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "page"


def _is_locked(path) -> bool:
    if not path.exists():
        return False
    head = path.read_text(encoding="utf-8")[:500]
    return bool(re.search(r"^locked:\s*true\s*$", head, flags=re.M))


def _snippet(t: str, n=160) -> str:
    t = " ".join(t.split())
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "…"


def _page(conn, node, members, related, slug_of) -> tuple[str, set]:
    by_id = {m["id"]: m for m in members}
    passages = "\n\n".join(f"[{m['id']}] {m['text'][:700]}" for m in members)[:9000]
    out = llm.generate(PAGE_PROMPT.format(title=node["title"], summary=node["text"], passages=passages),
                       schema=PAGE_SCHEMA, temperature=0.3)
    # render markdown ourselves; keep only citations to real member chunks
    cited = set()
    lines = ["## Overview", out["overview"].strip(), "", "## Key points"]
    for kp in out["key_points"]:
        ids = [i for i in kp["ids"] if i in by_id]
        cited.update(ids)
        refs = " ".join(f"[[chunk:{i}]]" for i in ids)
        lines.append(f"- {kp['point'].strip()} {refs}".rstrip())
    if out["open_questions"]:
        lines += ["", "## Open questions"] + [f"- {q.strip()}" for q in out["open_questions"]]
    body = "\n".join(lines)

    ids = list(by_id)
    tensions = conn.execute(
        f"SELECT DISTINCT src_node, dst_node, weight, method FROM edges WHERE type='CONTRADICTS' "
        f"AND src_node < dst_node AND src_node IN ({','.join('?'*len(ids))}) "
        f"AND dst_node IN ({','.join('?'*len(ids))}) ORDER BY weight DESC LIMIT 5",
        ids + ids,
    ).fetchall()
    lines = [f"---\ntitle: {node['title']}\nkind: {node['kind']}\nnode: {node['id']}\nlocked: false\n---",
             f"# {node['title']}", f"> {node['text']}", "", body, ""]
    if tensions:
        lines.append("## Tensions")
        for t in tensions:
            a, b = by_id[t["src_node"]], by_id[t["dst_node"]]
            lines.append(f"- [[chunk:{a['id']}]] “{_snippet(a['text'])}” **vs** "
                         f"[[chunk:{b['id']}]] “{_snippet(b['text'])}” ({t['method']} {t['weight']:.2f})")
        lines.append("")
    if related:
        lines.append("## Related")
        lines += [f"- [[{slug_of[r['id']]}]] {r['title']}" for r in related if r["id"] in slug_of]
        lines.append("")
    lines.append("## Sources")
    for m in members:
        lines.append(f"- [[chunk:{m['id']}]] *{m['doc']} › {m['title']}*: {_snippet(m['text'], 100)}")
    return "\n".join(lines) + "\n", cited


def run(conn):
    wiki = config.WIKI_DIR
    wiki.mkdir(parents=True, exist_ok=True)
    nodes = conn.execute("SELECT * FROM nodes WHERE kind IN ('theme','cluster') ORDER BY level DESC, id").fetchall()
    if not nodes:
        log("stage 6: no clusters/themes (run stage 4 first)")
        return

    slug_of, used = {}, set()
    for n in nodes:
        s = slugify(n["title"])
        while s in used:
            s += "-x"
        used.add(s)
        slug_of[n["id"]] = s

    with db.tx(conn):
        conn.execute("DELETE FROM wiki_pages")
        db.delete_orphan_vectors(conn)

    written = 0
    for i, n in enumerate(nodes, 1):
        if n["kind"] == "cluster":
            members = conn.execute(
                "SELECT c.id, c.text, c.title, d.title AS doc FROM nodes c "
                "JOIN nodes s ON s.id=c.parent_id JOIN nodes d ON d.id=s.parent_id "
                "WHERE c.cluster_id=? ORDER BY c.item_id, c.ord", (n["id"],)).fetchall()
            related = conn.execute(
                "SELECT id, title FROM nodes WHERE id=? OR (kind='cluster' AND parent_id=? AND id!=?)",
                (n["parent_id"], n["parent_id"], n["id"])).fetchall()
        else:  # theme: its member chunks are the chunks of its clusters
            members = conn.execute(
                "SELECT c.id, c.text, c.title, d.title AS doc FROM nodes c "
                "JOIN nodes k ON k.id=c.cluster_id "
                "JOIN nodes s ON s.id=c.parent_id JOIN nodes d ON d.id=s.parent_id "
                "WHERE k.parent_id=? ORDER BY c.item_id, c.ord", (n["id"],)).fetchall()
            related = conn.execute("SELECT id, title FROM nodes WHERE kind='cluster' AND parent_id=?",
                                   (n["id"],)).fetchall()
        path = wiki / f"{slug_of[n['id']]}.md"
        if _is_locked(path):
            text = path.read_text(encoding="utf-8")
            cited = {int(x) for x in re.findall(r"\[\[chunk:(\d+)\]\]", text)}
        else:
            try:
                text, cited = _page(conn, n, members, related, slug_of)
            except llm.LLMError as e:
                log(f"  page {n['title']}: {e}")
                continue
            path.write_text(text, encoding="utf-8")
            written += 1
        # A page "covers" all its member chunks (used by zoom-out), cited or not.
        node_ids = sorted({m["id"] for m in members} | cited)
        with db.tx(conn):
            conn.execute(
                "INSERT INTO wiki_pages(slug, title, summary, path, source_node, node_ids_json) "
                "VALUES (?,?,?,?,?,?)",
                (slug_of[n["id"]], n["title"], n["text"], str(path), n["id"], json.dumps(node_ids)),
            )
        progress(i, len(nodes), "wiki pages", every=5)

    pages = conn.execute("SELECT id, title, summary FROM wiki_pages").fetchall()
    vecs = llm.embed([f"{p['title']}\n{p['summary']}" for p in pages])
    with db.tx(conn):
        db.put_vectors(conn, "wiki", [p["id"] for p in pages], vecs)

    _write_index(conn, wiki)
    with open(wiki / "log.md", "a", encoding="utf-8") as f:
        f.write(f"- {dt.datetime.now():%Y-%m-%d %H:%M} build: {len(pages)} pages "
                f"({written} regenerated), model {config.LLM_MODEL}\n")
    log(f"stage 6: {len(pages)} wiki pages ({written} written) in {wiki}")


def _write_index(conn, wiki):
    themes = conn.execute("SELECT n.id, n.title, n.text, w.slug FROM nodes n JOIN wiki_pages w "
                          "ON w.source_node=n.id WHERE n.kind='theme' ORDER BY n.id").fetchall()
    lines = ["# Wiki index", "", "_Generated by `dip.ingest.run`. Edit pages freely; set "
             "`locked: true` in a page's front matter to stop it being regenerated._", ""]
    for t in themes:
        lines += [f"## [[{t['slug']}]] {t['title']}", t["text"], ""]
        for c in conn.execute("SELECT n.title, w.slug FROM nodes n JOIN wiki_pages w ON "
                              "w.source_node=n.id WHERE n.parent_id=? ORDER BY n.id", (t["id"],)):
            lines.append(f"- [[{c['slug']}]] {c['title']}")
        lines.append("")
    n_images = conn.execute("SELECT count(*) FROM images").fetchone()[0]
    if n_images:
        lines += ["## Images", f"[[images/index]]: {n_images} images with descriptions, metaphors and credit", ""]
    (wiki / "index.md").write_text("\n".join(lines), encoding="utf-8")


def lint(conn, dup_threshold: float = 0.9, conflict_threshold: float = 0.8):
    """Health check for the wiki. Writes data/wiki/lint.md and adds wiki_lint CONTRADICTS
    edges between cluster/theme nodes whose pages the LLM judges to conflict."""
    pages = conn.execute("SELECT * FROM wiki_pages").fetchall()
    idx = db.VectorIndex("wiki").refresh(conn)
    report = [f"# Wiki lint — {dt.datetime.now():%Y-%m-%d %H:%M}", ""]

    ids = [p["id"] for p in pages if idx.get(p["id"]) is not None]
    by_id = {p["id"]: p for p in pages}
    sims = []
    if len(ids) > 1:
        mat = np.stack([idx.get(i) for i in ids])
        S = mat @ mat.T
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                sims.append((float(S[a, b]), ids[a], ids[b]))
    sims.sort(reverse=True)

    report.append("## Possible duplicates")
    dups = [(s, a, b) for s, a, b in sims if s >= dup_threshold]
    report += [f"- [[{by_id[a]['slug']}]] ≈ [[{by_id[b]['slug']}]] ({s:.2f})" for s, a, b in dups] or ["- none"]

    report += ["", "## Pages without citations"]
    empty = []
    for p in pages:
        path = config.WIKI_DIR / f"{p['slug']}.md"
        if path.exists() and "[[chunk:" not in path.read_text(encoding="utf-8").split("## Sources")[0]:
            empty.append(f"- [[{p['slug']}]]")
    report += empty or ["- none"]

    report += ["", "## Conflicts between similar pages (LLM-checked)"]
    found = []
    for s, a, b in [x for x in sims if conflict_threshold <= x[0] < dup_threshold][:20]:
        pa, pb = by_id[a], by_id[b]
        out = llm.generate(
            f"Page A ({pa['title']}): {pa['summary']}\n\nPage B ({pb['title']}): {pb['summary']}\n\n"
            "Do these two summaries make conflicting claims? Explain in one sentence.",
            schema=LINT_SCHEMA)
        if out["conflict"]:
            found.append((pa, pb, out["explanation"]))
            report.append(f"- [[{pa['slug']}]] vs [[{pb['slug']}]]: {out['explanation']}")
    if not found:
        report.append("- none")
    with db.tx(conn):
        conn.execute("DELETE FROM edges WHERE method='wiki_lint'")
        for pa, pb, _ in found:
            for s, d in ((pa["source_node"], pb["source_node"]), (pb["source_node"], pa["source_node"])):
                conn.execute("INSERT OR IGNORE INTO edges(src_node, dst_node, type, weight, method) "
                             "VALUES (?,?,'CONTRADICTS',0.6,'wiki_lint')", (s, d))

    (config.WIKI_DIR / "lint.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    log(f"lint: {len(dups)} duplicates, {len(empty)} uncited pages, {len(found)} conflicts "
        f"-> {config.WIKI_DIR / 'lint.md'}")
