"""How well do embeddinggemma-2 vectors fit the nodes we already built (with nomic)?

Compares data/ (nomic-embed-text) with data-embeddinggemma2/ (the same build re-embedded with
google/embeddinggemma-2; clusters, themes and edges still come from the nomic run):

  structure   do the existing clusters/themes/sections hang together in each vector space?
              silhouette of the chunk clusters; share of chunks whose nearest cluster centroid,
              nearest cluster node and nearest section is their own; share of clusters whose
              nearest theme is their own
  neighbours  overlap of each chunk's 10 nearest neighbours between the two spaces (chunks
              matched across builds by text)
  retrieval   experiments/compare_embeddings.py on each build (provocations gold set)

    python experiments/gemma_alignment.py            # writes experiments/gemma_alignment.md
"""
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent.parent
BUILDS = {"nomic": ("data", {}),
          "gemma": ("data-embeddinggemma2", {"DIP_EMBED_BACKEND": "st"})}
K = 10


def _norm(t):
    return re.sub(r"\W+", " ", t.lower()).strip()


def load(data_dir):
    conn = sqlite3.connect(ROOT / data_dir / "dip.sqlite")
    conn.row_factory = sqlite3.Row
    vecs = {r["owner_id"]: np.frombuffer(r["vec"], dtype=np.float32)
            for r in conn.execute("SELECT owner_id, vec FROM vectors WHERE owner='node'")}
    vecs = {i: v / max(np.linalg.norm(v), 1e-9) for i, v in vecs.items()}
    nodes = {r["id"]: dict(r) for r in conn.execute("SELECT id, kind, parent_id, cluster_id, text FROM nodes")}
    return nodes, vecs


def _nearest_is_own(items, targets, vecs):
    """Share of (node, own target) pairs where the node's nearest target vector is its own."""
    tids = [t for t in targets if t in vecs]
    if not tids:
        return float("nan")
    T = np.stack([vecs[t] for t in tids])
    hits = [tids[int(np.argmax(T @ vecs[i]))] == own for i, own in items if i in vecs and own in vecs]
    return sum(hits) / len(hits)


def structure(nodes, vecs):
    chunks = [i for i, n in nodes.items() if n["kind"] == "chunk" and n["cluster_id"] and i in vecs]
    X = np.stack([vecs[i] for i in chunks])
    labels = np.array([nodes[i]["cluster_id"] for i in chunks])
    cents = {c: X[labels == c].mean(axis=0) for c in set(labels)}
    cids = list(cents)
    C = np.stack([cents[c] / np.linalg.norm(cents[c]) for c in cids])
    near_cent = np.mean([cids[int(np.argmax(C @ x))] == lab for x, lab in zip(X, labels)])
    clusters = [i for i, n in nodes.items() if n["kind"] == "cluster"]
    themes = [i for i, n in nodes.items() if n["kind"] == "theme"]
    sections = [i for i, n in nodes.items() if n["kind"] == "section"]
    return {
        "silhouette (chunks by cluster, cosine)": silhouette_score(X, labels, metric="cosine"),
        "chunk → own cluster centroid nearest": near_cent,
        "chunk → own cluster node nearest": _nearest_is_own([(i, nodes[i]["cluster_id"]) for i in chunks],
                                                           clusters, vecs),
        "cluster → own theme node nearest": _nearest_is_own([(i, nodes[i]["parent_id"]) for i in clusters],
                                                           themes, vecs),
        "chunk → own section nearest": _nearest_is_own(
            [(i, nodes[i]["parent_id"]) for i, n in nodes.items() if n["kind"] == "chunk"], sections, vecs),
    }


def neighbour_overlap(a, b):
    """Jaccard of the K nearest chunks, for chunks present in both builds (matched by text)."""
    (na, va), (nb, vb) = a, b
    by_text = {}
    for i, n in nb.items():
        if n["kind"] == "chunk" and i in vb:
            by_text.setdefault(_norm(n["text"]), i)
    pairs = [(i, by_text[_norm(n["text"])]) for i, n in na.items()
             if n["kind"] == "chunk" and i in va and _norm(n["text"]) in by_text]
    ia, ib = [p for p, _ in pairs], [q for _, q in pairs]
    A, B = np.stack([va[i] for i in ia]), np.stack([vb[i] for i in ib])
    SA, SB = A @ A.T, B @ B.T
    np.fill_diagonal(SA, -9)
    np.fill_diagonal(SB, -9)
    nnA, nnB = np.argsort(-SA, axis=1)[:, :K], np.argsort(-SB, axis=1)[:, :K]
    jac = [len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(nnA, nnB)]
    top1 = np.mean(nnA[:, 0] == nnB[:, 0])
    return len(pairs), float(np.mean(jac)), float(np.median(jac)), float(top1)


def retrieval(data_dir, env):
    out = subprocess.run([sys.executable, "experiments/compare_embeddings.py"], cwd=ROOT, capture_output=True,
                         text=True, env={**os.environ, "DIP_DATA_DIR": str(ROOT / data_dir), **env})
    return (out.stdout or out.stderr).strip().splitlines()[-3:]


def main():
    loaded = {name: load(d) for name, (d, _) in BUILDS.items()}
    lines = ["# embeddinggemma-2 vs nomic-embed-text on the existing nodes", "",
             "Clusters, themes and sections were built from nomic vectors; the gemma build is the same "
             "corpus re-embedded. Higher = the existing structure holds together better in that space.", "",
             "## Structure", "", "| measure | nomic | gemma |", "|---|---|---|"]
    st = {name: structure(*v) for name, v in loaded.items()}
    for m in st["nomic"]:
        lines.append(f"| {m} | {st['nomic'][m]:.3f} | {st['gemma'][m]:.3f} |")
    lines += ["", "nomic's centroid score is 1.000 by construction: k-means assigned each chunk to its nearest "
              "nomic centroid. The gemma figure is the meaningful one: how many chunks would stay put."]
    n, mean_j, med_j, top1 = neighbour_overlap(loaded["nomic"], loaded["gemma"])
    lines += ["", "## Neighbours", "",
              f"{n} chunks matched across builds. Jaccard of the {K} nearest chunks: mean {mean_j:.2f}, "
              f"median {med_j:.2f}; same single nearest neighbour for {top1:.0%}.", "",
              "## Retrieval (provocations gold set, experiments/compare_embeddings.py)", "", "```"]
    for name, (d, env) in BUILDS.items():
        lines += retrieval(d, env)
    lines += ["```", ""]
    probe = ROOT / "data-embeddinggemma2" / "image_probe" / "report.md"
    if probe.exists():
        lines += ["## Images (experiments/image_probe.py)", "", probe.read_text().splitlines()[2], ""]
    out = "\n".join(lines)
    print(out)
    (ROOT / "experiments" / "gemma_alignment.md").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
