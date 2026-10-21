"""Action contracts against a tiny hand-built database (no models needed)."""
import numpy as np
import pytest

from dip import config, db, llm
from dip.query import actions


def vec(*hot):
    v = np.zeros(config.EMBED_DIM, dtype=np.float32)
    for i, w in hot:
        v[i] = w
    return v


@pytest.fixture
def fx(tmp_path, monkeypatch):
    c = db.connect(tmp_path / "t.sqlite")
    ins = lambda sql, *a: c.execute(sql, a).lastrowid  # noqa: E731
    item = ins("INSERT INTO items(path,type,title,sha256) VALUES ('x','text','Doc','h')")
    doc = ins("INSERT INTO nodes(item_id,kind,level,title,text,abstraction) VALUES (?,?,?,?,?,?)",
              item, "document", 2, "Doc", "A document about scale in AI.", 5)
    sec = ins("INSERT INTO nodes(item_id,parent_id,kind,level,title,text,abstraction) VALUES (?,?,?,?,?,?,?)",
              item, doc, "section", 1, "Scale", "Section on scale.", 4)
    theme = ins("INSERT INTO nodes(kind,level,title,text,abstraction) VALUES ('theme',2,'Theme','t',5)")
    cluster = ins("INSERT INTO nodes(parent_id,kind,level,title,text,abstraction) VALUES (?,'cluster',1,'Cl','c',4)", theme)
    chunks = {}
    for name, text, role, ab in [
        ("thesis", "Small local AI beats big AI.", "claim", 4),
        ("anti", "Bigger models are always better.", "claim", 4),
        ("data", "A village co-op ran a tiny model for 5 years.", "evidence", 2),
        ("case", "StuntyAI is a regional model.", "example", 1),
    ]:
        chunks[name] = ins("INSERT INTO nodes(item_id,parent_id,cluster_id,kind,level,title,text,role,abstraction) "
                           "VALUES (?,?,?,'chunk',0,'Scale',?,?,?)", item, sec, cluster, text, role, ab)
        ins("INSERT INTO claims(node_id,text) VALUES (?,?)", chunks[name], text)
    T, A, D, C = (chunks[k] for k in ("thesis", "anti", "data", "case"))
    for s, d, t, w in [(A, T, "CONTRADICTS", .9), (T, A, "CONTRADICTS", .9), (D, T, "SUPPORTS", .8),
                       (C, T, "EXAMPLE_OF", .7)]:
        c.execute("INSERT INTO edges(src_node,dst_node,type,weight,method) VALUES (?,?,?,?,'nli')", (s, d, t, w))
    vecs = {T: vec((0, 1)), A: vec((0, 1), (1, .5)), D: vec((0, 1), (2, .8)), C: vec((0, .7), (3, 1)),
            sec: vec((0, 1), (4, .3)), doc: vec((0, 1), (5, .5)), cluster: vec((0, 1), (6, .2)),
            theme: vec((0, 1), (7, .6))}
    db.put_vectors(c, "node", list(vecs), list(vecs.values()))
    c.execute("INSERT INTO fts_nodes(rowid,title,text) SELECT id, coalesce(title,''), text FROM nodes")
    c.commit()
    db.bump_version(c)
    monkeypatch.setattr(llm, "embed_one", lambda t, query=False: vec((0, 1)))
    return c, dict(chunks, sec=sec, doc=doc, cluster=cluster, theme=theme)


def ids(out):
    return [r["id"] for r in out["results"]]


def test_result_shape(fx):
    conn, I = fx
    out = actions.neighbours(conn, node_id=I["thesis"], k=3)
    assert out["anchor"]["id"] == I["thesis"]
    r = out["results"][0]
    assert {"id", "kind", "text", "score", "why", "section", "document"} <= set(r)


def test_neighbours_excludes_self_and_returns_chunks(fx):
    conn, I = fx
    out = actions.neighbours(conn, node_id=I["thesis"], k=5)
    assert I["thesis"] not in ids(out)
    assert all(r["kind"] == "chunk" for r in out["results"])


def test_counter_uses_contradicts_edges(fx):
    conn, I = fx
    out = actions.counter(conn, node_id=I["thesis"], k=3, fallback=False)
    assert ids(out) == [I["anti"]]
    assert out["results"][0]["why"].startswith("edge:CONTRADICTS")


def test_evidence_uses_support_and_example_edges(fx):
    conn, I = fx
    out = actions.evidence(conn, node_id=I["thesis"], k=3, fallback=False)
    assert set(ids(out)) == {I["data"], I["case"]}


def test_zoom_in_goes_more_specific(fx):
    conn, I = fx
    out = actions.zoom_in(conn, node_id=I["thesis"], k=5)
    assert ids(out)[0] == I["case"]  # EXAMPLE_OF edge first
    assert all(r["abstraction"] < 4 for r in out["results"])


def test_zoom_out_goes_up_the_tree(fx):
    conn, I = fx
    out = actions.zoom_out(conn, node_id=I["case"], k=6)
    kinds = [r["kind"] for r in out["results"]]
    assert kinds[:2] == ["cluster", "theme"]
    assert I["thesis"] in ids(out)  # the claim it is an example of
    assert "section" in kinds


def test_zoom_in_on_cluster_returns_members(fx):
    conn, I = fx
    out = actions.zoom_in(conn, node_id=I["cluster"], k=10)
    assert set(ids(out)) == {I[k] for k in ("thesis", "anti", "data", "case")}


def test_unknown_node(fx):
    conn, I = fx
    with pytest.raises(actions.NotFound):
        actions.counter(conn, node_id=99999)
