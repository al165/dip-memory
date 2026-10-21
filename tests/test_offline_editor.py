"""Offline demo mode (no LLM, no embedding model) and the wiki editor endpoints."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from dip import api, config, llm
from dip.query import actions
from tests.test_actions import fx  # noqa: F401  (fixture)
from tests.test_images import lib  # noqa: F401  (fixture)


@pytest.fixture
def offline(fx, monkeypatch):  # noqa: F811
    monkeypatch.setattr(config, "OFFLINE", True)
    # undo the fixture's fake query embedding: the real one must refuse at once
    monkeypatch.setattr(llm, "embed_one", lambda t, query=False: llm.embed([t], query=query)[0])
    return fx


def client_for(conn):
    api.app.dependency_overrides[api.get_conn] = lambda: conn
    return TestClient(api.app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    api.app.dependency_overrides.clear()


def test_llm_calls_fail_fast_offline(monkeypatch):
    monkeypatch.setattr(config, "OFFLINE", True)
    t0 = time.time()
    for call in (lambda: llm.generate("hi"), lambda: llm.embed(["hi"])):
        with pytest.raises(llm.LLMError, match="offline"):
            call()
    assert time.time() - t0 < 1
    assert llm.available() == {"ollama": False, "offline": True}


def test_actions_work_offline(offline):
    conn, I = offline
    hits = actions.search(conn, "village co-op model")
    assert hits[0]["id"] == I["data"] and "keywords only" in hits[0]["why"]
    out = actions.counter(conn, node_id=I["thesis"], k=5)
    assert [r["id"] for r in out["results"]] == [I["anti"]]
    assert "offline" in out["meta"]["fallback_error"]  # stored links, fallback skipped
    assert actions.neighbours(conn, text="village co-op", k=2)["anchor"]["id"] == I["data"]
    assert actions.zoom_out(conn, node_id=I["case"], k=3)["results"]
    with pytest.raises(llm.LLMError):
        actions.question(conn, "why small?")


def test_api_offline(offline):
    conn, I = offline
    c = client_for(conn)
    assert c.get("/health").json()["offline"] is True
    assert c.get(f"/evidence?node_id={I['thesis']}").status_code == 200
    assert c.post("/question", json={"question": "why?"}).status_code == 503


def test_images_offline_use_cached_concept_vectors(lib, monkeypatch):  # noqa: F811
    conn, I = lib
    online = actions.images(conn, node_id=I["thesis"], k=4)
    assert conn.execute("SELECT count(*) FROM concept_vectors").fetchone()[0] == 1

    def down(*a, **kw):
        raise llm.LLMError("offline")
    monkeypatch.setattr(llm, "embed", down)
    monkeypatch.setattr(llm, "embed_one", down)
    monkeypatch.setattr(actions, "llm_concepts", down)
    off = actions.images(conn, node_id=I["thesis"], k=4)
    assert off["results"] == online["results"]
    assert any(r["match"] == "metaphor" for r in off["results"])


@pytest.fixture
def wiki(lib, tmp_path, monkeypatch):  # noqa: F811
    conn, I = lib
    monkeypatch.setattr(config, "WIKI_DIR", tmp_path / "wiki")
    page = config.WIKI_DIR / "cl.md"
    page.write_text("---\ntitle: Cl\nkind: cluster\nnode: 1\nlocked: false\n---\n# Cl\n> old summary\n\n"
                    f"## Sources\n- [[chunk:{I['thesis']}]] thesis\n")
    conn.execute("INSERT INTO wiki_pages(slug, title, summary, path, source_node, node_ids_json) "
                 "VALUES ('cl','Cl','old summary',?,?,?)", (str(page), I["cluster"], json.dumps([I["thesis"]])))
    conn.commit()
    return conn, I, page


def test_editor_lists_and_saves(wiki, monkeypatch):
    conn, I, page = wiki
    c = client_for(conn)
    files = c.get("/wiki-files").json()
    assert [(f["kind"], f["name"]) for f in files][:1] == [("topic", "cl.md")]
    assert {f["name"] for f in files if f["kind"] == "image"} == {"village.md", "mycelium.md"}

    doc = c.get("/wiki-files/topic/cl.md").json()
    text = doc["text"].replace("> old summary", "> a sharper summary") + f"- [[chunk:{I['case']}]] case\n"
    r = c.put("/wiki-files/topic/cl.md", json={"text": text, "base_hash": doc["hash"], "lock": True})
    assert r.status_code == 200 and r.json()["locked"] is True
    saved = page.read_text()
    assert "locked: true" in saved and "a sharper summary" in saved
    row = conn.execute("SELECT * FROM wiki_pages WHERE slug='cl'").fetchone()
    assert row["summary"] == "a sharper summary"
    assert set(json.loads(row["node_ids_json"])) == {I["thesis"], I["case"]}  # zoom-out sees the new citation

    # a second editor still holding the old version is refused
    stale = c.put("/wiki-files/topic/cl.md", json={"text": "x", "base_hash": doc["hash"]})
    assert stale.status_code == 409 and page.read_text() == saved


def test_editor_refuses_unknown_pages_and_missing_token(wiki, monkeypatch):
    conn, I, page = wiki
    c = client_for(conn)
    assert c.get("/wiki-files/topic/index.md").status_code == 404   # not a page in the database
    assert c.get("/wiki-files/topic/..%2Fdip.sqlite").status_code == 404
    assert c.get("/wiki-files/other/cl.md").status_code == 404
    monkeypatch.setattr(config, "EDIT_TOKEN", "s3cret")
    h = c.get("/wiki-files/topic/cl.md").json()["hash"]
    assert c.put("/wiki-files/topic/cl.md", json={"text": "x", "base_hash": h}).status_code == 403
    ok = c.put("/wiki-files/topic/cl.md", json={"text": page.read_text() + "\nmore\n", "base_hash": h},
               headers={"x-edit-token": "s3cret"})
    assert ok.status_code == 200
    assert c.get("/health").json()["editing"] == "token"


def test_image_files_found_by_name_on_another_server(lib, tmp_path, monkeypatch):  # noqa: F811
    conn, _ = lib
    conn.execute("UPDATE images SET path='/elsewhere/' || slug || '.png', page='/elsewhere/' || slug || '.md'")
    conn.commit()
    c = client_for(conn)
    assert c.get("/image-file/village").status_code == 200
    assert c.get("/image-page/village").status_code == 200
