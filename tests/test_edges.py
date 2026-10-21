"""Stage 5: NLI only proposes; the LLM verdict decides which edges exist."""
import numpy as np

from dip import config, db
from dip.ingest import s5_edges


def test_llm_verdict_decides_edges(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "t.sqlite")
    texts = {
        "big": "Bigger AI models are always better.",
        "small": "Small local AI can be better than big AI.",
        "pines": "The Kodaikanal pines grew on colonised land.",
        "memory": "The Kodaikanal pines are a childhood memory.",
    }
    nodes, claims = {}, {}
    for k, t in texts.items():
        nodes[k] = conn.execute("INSERT INTO nodes(kind, level, text, abstraction) VALUES ('chunk',0,?,3)", (t,)).lastrowid
        claims[k] = conn.execute("INSERT INTO claims(node_id, text) VALUES (?,?)", (nodes[k], t)).lastrowid
    vec = lambda *h: np.eye(config.EMBED_DIM, dtype=np.float32)[list(h)].sum(0)  # noqa: E731
    db.put_vectors(conn, "claim", [claims[k] for k in texts],
                   [vec(0, 1), vec(0, 1, 2), vec(5, 6), vec(5, 6, 7)])
    conn.commit()
    db.bump_version(conn)

    # NLI says *both* pairs contradict; only the first really does.
    monkeypatch.setattr(s5_edges.nli, "score", lambda pairs: [
        {"contradiction": 0.95, "entailment": 0.01, "neutral": 0.04} for _ in pairs])
    real = {frozenset((texts["big"], texts["small"]))}
    monkeypatch.setattr(s5_edges, "_ask_llm",
                        lambda a, b: "challenges" if frozenset((a, b)) in real else "unrelated")
    monkeypatch.setattr(config, "LLM_CONCURRENCY", 2)
    monkeypatch.setattr(config, "COUNTER_JUDGE", "legacy")

    s5_edges.run(conn)
    edges = {(r["src_node"], r["dst_node"], r["type"], r["method"]) for r in conn.execute("SELECT * FROM edges")}
    assert edges == {
        (nodes["big"], nodes["small"], "CONTRADICTS", "nli+llm"),
        (nodes["small"], nodes["big"], "CONTRADICTS", "nli+llm"),
    }
    # verdicts are cached, so a re-run asks the LLM nothing
    monkeypatch.setattr(s5_edges, "_ask_llm", lambda a, b: (_ for _ in ()).throw(AssertionError("LLM called")))
    s5_edges.run(conn)


def test_nli_scores_are_cached(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "t.sqlite")
    calls = []

    def fake(pairs):
        calls.append(len(pairs))
        return [{"contradiction": 0.1, "entailment": 0.8, "neutral": 0.1} for _ in pairs]

    monkeypatch.setattr(s5_edges.nli, "score", fake)
    pairs = [("a", "b"), ("c", "d")]
    first = s5_edges._nli_cached(conn, pairs, "t")
    second = s5_edges._nli_cached(conn, pairs + [("e", "f")], "t")
    assert calls == [2, 1]  # only the new pair is scored the second time
    assert first == second[:2]


def test_graded_counter_links(tmp_path, monkeypatch):
    """COUNTER_JUDGE=graded: whole-passage judgements in both orders decide CONTRADICTS links."""
    conn = db.connect(tmp_path / "t.sqlite")
    doc = conn.execute("INSERT INTO nodes(kind, level, title, text) VALUES ('document',2,'Doc','d')").lastrowid
    sec = conn.execute("INSERT INTO nodes(kind, level, parent_id, title, text) VALUES ('section',1,?,'Sec','s')",
                       (doc,)).lastrowid
    texts = {"frictionless": "Digital technology should remove all friction.",
             "friction": "We should design friction back into technology.",
             "pines": "The pines grew on colonised land.",
             "garden": "A digital garden grows over time."}
    nodes = {k: conn.execute("INSERT INTO nodes(kind, level, parent_id, text) VALUES ('chunk',0,?,?)",
                             (sec, t)).lastrowid for k, t in texts.items()}
    claims = {k: conn.execute("INSERT INTO claims(node_id, text) VALUES (?,?)", (nodes[k], t)).lastrowid
              for k, t in texts.items()}
    vec = lambda *h: np.eye(config.EMBED_DIM, dtype=np.float32)[list(h)].sum(0)  # noqa: E731
    vecs = [vec(0, 1), vec(0, 1, 2), vec(5, 6), vec(5, 6, 7)]
    db.put_vectors(conn, "node", [nodes[k] for k in texts], vecs)
    db.put_vectors(conn, "claim", [claims[k] for k in texts], vecs)
    conn.commit()
    db.bump_version(conn)

    # the claim-level LLM calls everything a challenge: in graded mode that must not create links
    monkeypatch.setattr(s5_edges.nli, "score", lambda pairs: [
        {"contradiction": 0.95, "entailment": 0.01, "neutral": 0.04} for _ in pairs])
    monkeypatch.setattr(s5_edges, "_ask_llm", lambda a, b: "challenges")
    monkeypatch.setattr(config, "COUNTER_JUDGE", "graded")
    monkeypatch.setattr(config, "COUNTER_NEIGHBOURS", 1)
    monkeypatch.setattr(config, "LLM_CONCURRENCY", 2)
    opposed = {frozenset((texts["frictionless"], texts["friction"]))}
    asked = []

    def fake_graded(a, b):
        asked.append((a["id"], b["id"]))
        if frozenset((a["text"], b["text"])) in opposed:
            return {1: 0.6, 2: 0.2, 4: 0.2} if a["id"] == nodes["frictionless"] else {2: 0.5, 4: 0.5}
        return {3: 0.7, 2: 0.3}  # some pushback, but below the threshold
    monkeypatch.setattr(s5_edges, "_ask_graded", fake_graded)

    s5_edges.run(conn)
    edges = {(r["src_node"], r["dst_node"]): (r["weight"], r["method"])
             for r in conn.execute("SELECT * FROM edges WHERE type='CONTRADICTS'")}
    a, b = nodes["frictionless"], nodes["friction"]
    assert set(edges) == {(a, b), (b, a)}
    assert edges[(a, b)][1] == "graded" and abs(edges[(a, b)][0] - 0.65) < 1e-9  # mean of 0.8 and 0.5
    assert len(asked) == 4  # 2 passage pairs x 2 orders

    # judgements are cached by passage text, so a re-run asks nothing
    monkeypatch.setattr(s5_edges, "_ask_graded", lambda a, b: (_ for _ in ()).throw(AssertionError("LLM called")))
    s5_edges.run(conn)
