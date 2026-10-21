"""Web sources: .urls expansion and hash-by-extracted-text idempotency (no network)."""
from dip import db, parse
from dip.ingest import run, s1_parse_chunk

ARTICLE = "# A web article\n\n## Part one\n" + "Community data trusts keep power local. " * 12


def test_expand_reads_url_lists(tmp_path):
    (tmp_path / "reading.urls").write_text(
        "# reading list\nhttps://example.org/a\n\nhttps://example.org/b  # note\nhttps://example.org/a\n")
    (tmp_path / "doc.md").write_text("# x")
    out = run.expand([str(tmp_path), "https://example.org/c"])
    # folder contents in name order (doc.md, reading.urls), duplicates removed
    assert out == [tmp_path / "doc.md", "https://example.org/a", "https://example.org/b", "https://example.org/c"]


def test_url_ingest_is_idempotent_on_article_text(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "t.sqlite")
    text = {"v": ARTICLE}
    monkeypatch.setattr(parse, "load_url", lambda url: parse.Parsed("web", "A web article", text["v"]))
    url = "https://example.org/article"

    item, changed = s1_parse_chunk.ingest_source(conn, url)
    assert changed and conn.execute("SELECT path, type FROM items").fetchone()[:] == (url, "web")
    assert s1_parse_chunk.ingest_source(conn, url) == (item, False)       # same text: skipped

    text["v"] = ARTICLE + "\n\nAn added paragraph about cooperatives and local servers."
    _, changed = s1_parse_chunk.ingest_source(conn, url)                     # edited: rebuilt
    assert changed
    assert conn.execute("SELECT count(*) FROM items").fetchone()[0] == 1
    texts = " ".join(r[0] for r in conn.execute("SELECT text FROM nodes WHERE kind='chunk'"))
    assert "cooperatives and local servers" in texts


def test_markdown_front_matter_is_stripped(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("---\ntitle: My note\nsource: https://example.org\n---\n\n# Heading\n\nBody text.\n")
    parsed = parse.load(p)
    assert parsed.title == "My note"
    assert "source:" not in parsed.text and parsed.text.startswith("# Heading")


def test_local_html_is_extracted(tmp_path):
    body = "<p>" + "Community data trusts keep power local and accountable to residents. " * 6 + "</p>"
    p = tmp_path / "page.html"
    p.write_text(f"<html><head><title>Data trusts</title></head><body><nav>Home | About | Shop</nav>"
                 f"<article><h1>Data trusts</h1><h2>Why local</h2>{body}<h2>How</h2>{body}</article>"
                 f"<footer>Copyright 2026</footer></body></html>")
    parsed = parse.load(p)
    assert parsed.type == "html" and parsed.title == "Data trusts"
    assert "## Why local" in parsed.text and "Community data trusts" in parsed.text
    assert "Shop" not in parsed.text and "Copyright" not in parsed.text
