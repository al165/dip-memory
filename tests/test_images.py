"""Image library (stage 7) and the Images action, with stub models."""
import re
import zlib

import numpy as np
import pytest

from dip import config, db, llm
from dip.ingest import s7_images
from dip.query import actions
from tests.test_actions import fx  # noqa: F401  (fixture)


def bow(text, query=False):
    """Bag-of-words stub embedding: texts sharing words are similar."""
    v = np.zeros(config.EMBED_DIM, dtype=np.float32)
    for w in re.findall(r"[a-z]{3,}", text.lower()):
        v[zlib.crc32(w.encode()) % config.EMBED_DIM] += 1
    return v


DRAFTS = {
    "village": {"title": "Village co-op", "description": "A small local village cooperative running a tiny AI model.",
                "subjects": ["village", "cooperative", "computer"],
                "metaphors": [{"keyword": "self-reliance", "gloss": "a community looking after its own tools"}]},
    "mycelium": {"title": "Mycelium", "description": "Fungal threads under a forest floor.",
                 "subjects": ["fungi", "soil", "roots"],
                 "metaphors": [{"keyword": "decentralisation", "gloss": "many parts working without centre"},
                               {"keyword": "hidden infrastructure", "gloss": "what supports the visible"}]},
}


@pytest.fixture
def lib(fx, tmp_path, monkeypatch):  # noqa: F811
    conn, I = fx
    monkeypatch.setattr(config, "IMAGE_DIR", tmp_path / "images")
    monkeypatch.setattr(config, "IMAGE_WIKI_DIR", tmp_path / "wiki" / "images")
    monkeypatch.setattr(config, "EMBED_BACKEND", "ollama")
    monkeypatch.setattr(config, "EMBED_MULTIMODAL", False)
    monkeypatch.setattr(llm, "embed", lambda texts, query=False, batch=32: [bow(t) for t in texts])
    monkeypatch.setattr(llm, "embed_one", bow)
    monkeypatch.setattr(s7_images, "draft", lambda img, rel: {**DRAFTS[img.stem], "file": rel,
                                                              **s7_images.read_source_file(img)})
    monkeypatch.setattr(actions, "llm_concepts", lambda text: ["many parts working without centre"])
    config.IMAGE_DIR.mkdir()
    for name in DRAFTS:
        (config.IMAGE_DIR / f"{name}.png").write_bytes(name.encode())
    (config.IMAGE_DIR / "mycelium.png.source.txt").write_text(
        "https://commons.wikimedia.org/wiki/File:Mycelium.jpg\nlicense: CC BY-SA 4.0\n"
        "credit: A. Fungus, Mycelium, CC BY-SA 4.0, via Wikimedia Commons\n")
    # the anchor passage is about small local AI (shares words with the village image)
    db.put_vectors(conn, "node", [I["thesis"]], [bow("Small local AI beats big AI.")])
    conn.commit()
    db.bump_version(conn)
    s7_images.run(conn)
    return conn, I


def test_page_round_trip():
    d = {**DRAFTS["mycelium"], "file": "m.jpg", "source": "https://x.org/a:b#c", "creator": "", "license": "PD",
         "credit": "Someone, via X"}
    out = s7_images.parse_page(s7_images.render_page(d))
    assert out["title"] == "Mycelium" and out["file"] == "m.jpg"
    assert out["source"] == "https://x.org/a:b#c" and out["creator"] == "TODO" and out["credit"] == "Someone, via X"
    assert out["subjects"] == ["fungi", "soil", "roots"]
    assert out["metaphors"][1] == {"keyword": "hidden infrastructure", "gloss": "what supports the visible"}
    assert out["locked"] == "false"


def test_source_file(tmp_path):
    img = tmp_path / "a.jpg"
    img.with_name("a.jpg.source.txt").write_text("https://example.org/a\nJane Doe, 2019, CC0\n")
    assert s7_images.read_source_file(img) == {"source": "https://example.org/a", "credit": "Jane Doe, 2019, CC0"}


def test_stage_builds_library(capsys, lib):
    conn, _ = lib
    rows = {r["slug"]: r for r in conn.execute("SELECT * FROM images")}
    assert set(rows) == {"village", "mycelium"}
    assert rows["mycelium"]["credit"].startswith("A. Fungus") and rows["mycelium"]["license"] == "CC BY-SA 4.0"
    assert rows["village"]["credit"] is None
    assert "credit missing for image village" in capsys.readouterr().err
    n = {r["owner"]: r["n"] for r in conn.execute("SELECT owner, count(*) n FROM vectors GROUP BY owner")}
    assert n["image_desc"] == 2 and n["image_metaphor"] == 3 and "image_visual" not in n
    assert (config.IMAGE_WIKI_DIR / "index.md").exists()


def test_locked_page_is_kept_and_edits_reload(lib):
    conn, _ = lib
    page = config.IMAGE_WIKI_DIR / "village.md"
    text = page.read_text().replace("locked: false", "locked: true").replace(
        "- self-reliance: a community looking after its own tools", "- commons: shared tools held in common")
    page.write_text(text)
    (config.IMAGE_DIR / "village.png").write_bytes(b"a new picture")  # would trigger a redraft if unlocked
    s7_images.run(conn)
    assert page.read_text() == text
    assert [r["keyword"] for r in conn.execute(
        "SELECT keyword FROM image_metaphors m JOIN images i ON i.id=m.image_id WHERE i.slug='village'")] == ["commons"]


def test_removed_image_drops_row(lib):
    conn, _ = lib
    (config.IMAGE_DIR / "village.png").unlink()
    s7_images.run(conn)
    assert [r["slug"] for r in conn.execute("SELECT slug FROM images")] == ["mycelium"]
    assert conn.execute("SELECT count(*) FROM vectors WHERE owner='image_desc'").fetchone()[0] == 1


def test_images_action(lib):
    conn, I = lib
    out = actions.images(conn, node_id=I["thesis"], k=4)
    assert out["action"] == "images" and out["meta"]["concepts_from"] == "llm"
    by_match = {r["match"]: r for r in out["results"]}
    assert by_match["subject"]["slug"] == "village"
    m = by_match["metaphor"]
    assert m["slug"] == "mycelium" and m["metaphor"]["keyword"] == "decentralisation"
    assert m["concept"] == "many parts working without centre"
    assert m["url"] == "/image-file/mycelium" and m["credit"].startswith("A. Fungus")
    assert len({r["id"] for r in out["results"]}) == len(out["results"]) == 2  # each image once
    assert conn.execute("SELECT count(*) FROM concept_cache").fetchone()[0] == 1


def test_images_action_without_llm(lib, monkeypatch):
    conn, I = lib

    def down(text):
        raise llm.LLMError("offline")
    monkeypatch.setattr(actions, "llm_concepts", down)
    out = actions.images(conn, node_id=I["case"], k=2)
    assert out["meta"]["concepts_from"] == "tree"
    assert out["meta"]["concepts"][:2] == ["Cl", "Theme"]  # cluster, then theme title
    assert len(out["results"]) == 2


def test_images_action_empty_library(fx):  # noqa: F811
    conn, I = fx
    out = actions.images(conn, node_id=I["thesis"])
    assert out["results"] == [] and "empty" in out["meta"]["note"]


def test_image_library_endpoint(lib):
    from fastapi.testclient import TestClient

    from dip import api
    conn, _ = lib
    api.app.dependency_overrides[api.get_conn] = lambda: conn
    try:
        client = TestClient(api.app)
        rows = {r["slug"]: r for r in client.get("/image-library").json()}
        assert set(rows) == {"village", "mycelium"}
        m = rows["mycelium"]
        assert m["url"] == "/image-file/mycelium" and m["license"] == "CC BY-SA 4.0"
        assert [x["keyword"] for x in m["metaphors"]] == ["decentralisation", "hidden infrastructure"]
        assert m["subjects"] == ["fungi", "soil", "roots"] and rows["village"]["credit"] is None
        assert client.get("/image-page/mycelium").text.startswith("---\ntitle: Mycelium")
    finally:
        api.app.dependency_overrides.clear()


def test_image_passages(lib):
    conn, I = lib
    out = actions.image_passages(conn, "village", k=3)
    assert out["action"] == "image-passages" and out["anchor"]["slug"] == "village"
    first = out["results"][0]
    assert first["id"] == I["thesis"] and first["match"] == "subject"  # shares "small local AI"
    assert len({r["id"] for r in out["results"]}) == len(out["results"])
    via = actions.image_passages(conn, "mycelium", k=4, metaphor_via="passages")
    m = [r for r in via["results"] if r["match"] == "metaphor"]
    assert m and m[0]["metaphor"]["keyword"] in ("decentralisation", "hidden infrastructure")
    with pytest.raises(actions.NotFound):
        actions.image_passages(conn, "nope")
