"""Stage 1 — parse a source (file or web URL) and build its structural tree:
document -> section -> chunk.

Idempotent: a source whose content hash hasn't changed is skipped; a changed one is deleted
(with all its nodes, claims, edges and vectors) and rebuilt. Files are hashed by their bytes;
web pages are re-fetched and hashed by their extracted article text, so layout or ad changes
on the page don't trigger a re-ingest.
"""
import hashlib
from pathlib import Path

from dip import chunk, db, parse
from dip.ingest.common import log


def _placeholder(texts: list[str], limit: int = 1200) -> str:
    """Section/document text before stage 3 writes a real summary."""
    return " ".join(texts)[:limit]


def _delete_item(conn, item_id: int):
    """Remove an item with its nodes (and, by cascade, claims and edges) and vectors."""
    conn.execute("DELETE FROM nodes WHERE item_id=?", (item_id,))
    conn.execute("DELETE FROM items WHERE id=?", (item_id,))
    db.delete_orphan_vectors(conn)


def _source_exists(path: str) -> bool:
    return parse.is_url(path) or Path(path).exists()


def find_duplicates(conn) -> list[tuple[dict, list[dict]]]:
    """Groups of items with identical content: [(item to keep, [items to remove])].
    Keep the copy whose file still exists, then the most annotated one, then the oldest."""
    rows = [dict(r) for r in conn.execute(
        "SELECT i.*, (SELECT count(*) FROM nodes n WHERE n.item_id=i.id AND n.kind='chunk' "
        "AND n.abstraction IS NOT NULL) AS annotated FROM items i "
        "WHERE sha256 IN (SELECT sha256 FROM items GROUP BY sha256 HAVING count(*) > 1) ORDER BY id")]
    groups = {}
    for r in rows:
        groups.setdefault(r["sha256"], []).append(r)
    out = []
    for items in groups.values():
        items.sort(key=lambda r: (not _source_exists(r["path"]), -r["annotated"], r["id"]))
        out.append((items[0], items[1:]))
    return out


def remove_duplicates(conn) -> int:
    removed = 0
    for keep, drop in find_duplicates(conn):
        with db.tx(conn):
            for d in drop:
                _delete_item(conn, d["id"])
                log(f"removed duplicate item {d['id']} ({d['path']}); kept item {keep['id']} ({keep['path']})")
                removed += 1
    return removed


def ingest_source(conn, source, force: bool = False) -> tuple[int | None, bool]:
    """Ingest a file path or an http(s) URL. Returns (item_id, changed)."""
    if parse.is_url(source):
        key, name = source, source
        parsed = parse.load_url(source)
        digest = hashlib.sha256(parsed.text.encode()).hexdigest()
    else:
        path = Path(source).resolve()
        key, name = str(path), path.name
        digest = parse.sha256(path)
        parsed = None
    existing = conn.execute("SELECT id, sha256 FROM items WHERE path=?", (key,)).fetchone()
    if existing and existing["sha256"] == digest and not force:
        log(f"unchanged, skipping: {name}")
        return existing["id"], False

    # Same content already ingested under another path/URL: never index it twice.
    twin = conn.execute("SELECT id, path FROM items WHERE sha256=? AND path!=?", (digest, key)).fetchone()
    if twin and not force:
        with db.tx(conn):
            if existing:  # this path's old content is gone; its new content is the twin's
                _delete_item(conn, existing["id"])
            if _source_exists(twin["path"]):
                log(f"duplicate of {twin['path']}, skipping: {name}")
            else:  # moved or renamed file: repoint the existing item instead of re-ingesting
                conn.execute("UPDATE items SET path=? WHERE id=?", (key, twin["id"]))
                log(f"moved, not re-ingested: {twin['path']} -> {key}")
        return twin["id"], bool(existing)

    parsed = parsed or parse.load(path)
    title, sections = chunk.structure(parsed.text, parsed.title)
    if title == "Untitled":
        title = Path(key).stem if parsed.type != "web" else key
    chunks = chunk.chunk(sections)
    if not chunks:
        log(f"no text found in {name}, skipping")
        return None, False

    with db.tx(conn):
        if existing:
            log(f"changed, re-ingesting: {name}")
            _delete_item(conn, existing["id"])
        item_id = conn.execute(
            "INSERT INTO items(path, type, title, sha256) VALUES (?,?,?,?)",
            (key, parsed.type, title, digest),
        ).lastrowid
        doc_id = conn.execute(
            "INSERT INTO nodes(item_id, kind, level, title, text) VALUES (?,?,?,?,?)",
            (item_id, "document", 2, title, _placeholder([c.text for c in chunks])),
        ).lastrowid
        sec_ids = {}
        for s_ord, sec in enumerate(sections):
            sec_chunks = [c.text for c in chunks if c.section_idx == s_ord]
            if not sec_chunks:
                continue
            sec_ids[s_ord] = conn.execute(
                "INSERT INTO nodes(item_id, parent_id, kind, level, title, text, ord) "
                "VALUES (?,?,?,?,?,?,?)",
                (item_id, doc_id, "section", 1, sec.title, _placeholder(sec_chunks), s_ord),
            ).lastrowid
        for c in chunks:
            conn.execute(
                "INSERT INTO nodes(item_id, parent_id, kind, level, title, text, role, ord) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (item_id, sec_ids[c.section_idx], "chunk", 0, c.section, c.text, c.role, c.ord),
            )
    log(f"{name}: {len(sec_ids)} sections, {len(chunks)} chunks")
    return item_id, True


ingest_file = ingest_source  # backwards-compatible name
