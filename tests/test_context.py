"""Contextual retrieval: stage 3 writes a context prefix per chunk, used for vectors and FTS only."""
from dip import db
from dip.ingest import common, s1_parse_chunk, s3_annotate

DOC = ("# Data trusts\n\n## Why local\n" + "Community data trusts keep power local. " * 10 +
       "\n\n## How\n" + "They pool residents' records under a shared charter. " * 10)


def _ingest(tmp_path):
    conn = db.connect(tmp_path / "t.sqlite")
    f = tmp_path / "doc.md"
    f.write_text(DOC)
    s1_parse_chunk.ingest_source(conn, str(f))
    return conn


def test_context_is_written_embedded_and_searchable(tmp_path, monkeypatch):
    conn = _ingest(tmp_path)
    prompts = []

    def fake_generate(prompt, **kw):
        prompts.append(prompt)
        return f"From the Data trusts report, on governance number {len(prompts)}."

    monkeypatch.setattr(s3_annotate.llm, "generate", fake_generate)
    monkeypatch.setattr(common.llm, "embed", lambda texts, **kw: [[1.0] * 8 for _ in texts])
    common.embed_missing_nodes(conn)
    n = s3_annotate.contextualise_chunks(conn)
    chunks = conn.execute("SELECT * FROM nodes WHERE kind='chunk' ORDER BY ord").fetchall()
    assert n == len(chunks) >= 2
    assert all(c["context"].startswith("From the Data trusts report") for c in chunks)
    assert all("Community data trusts" in p and "shared charter" in p for p in prompts)  # whole doc fits
    # the displayed text is untouched; vectors were dropped so they are rebuilt with the prefix
    assert not chunks[0]["text"].startswith("From")
    assert conn.execute("SELECT count(*) FROM vectors WHERE owner='node' AND owner_id=?",
                        (chunks[0]["id"],)).fetchone()[0] == 0
    assert common.embed_text_for(chunks[0]).startswith(chunks[0]["context"] + "\n\n")
    common.rebuild_fts(conn)
    hits = conn.execute("SELECT rowid FROM fts_nodes WHERE fts_nodes MATCH 'governance'").fetchall()
    assert {r[0] for r in hits} == {c["id"] for c in chunks}
    assert s3_annotate.contextualise_chunks(conn) == 0  # nothing left to do


def test_long_documents_get_outline_and_own_section(tmp_path):
    conn = _ingest(tmp_path)
    chunks = conn.execute("SELECT * FROM nodes WHERE kind='chunk' ORDER BY ord").fetchall()
    body = s3_annotate._context_body(chunks, chunks[-1], limit=700)
    assert body.startswith("Sections: Why local; How")
    assert "shared charter" in body and "Community data trusts" not in body


def test_old_databases_gain_the_column(tmp_path):
    import sqlite3
    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY, item_id INTEGER, parent_id INTEGER, "
                "cluster_id INTEGER, kind TEXT NOT NULL, level INTEGER NOT NULL, title TEXT, "
                "text TEXT NOT NULL, abstraction REAL, role TEXT, topics_json TEXT, ord INTEGER DEFAULT 0)")
    old.commit()
    old.close()
    conn = db.connect(path)
    assert "context" in {r["name"] for r in conn.execute("PRAGMA table_info(nodes)")}