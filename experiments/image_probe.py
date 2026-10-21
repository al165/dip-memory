"""Cross-modal probe for an image-capable embedding model (DIP_EMBED_BACKEND=st).

Extracts the images from the ingested PDFs, embeds them into the same space as the passages,
and reports:
  1. grounding: for each image, is its nearest passage from the PDF it came from? (8 sources, so
     chance is well under 50%; a text-only pipeline cannot do this at all)
  2. text → image: the closest images for a few gold-set provocations
Writes <DIP_DATA_DIR>/image_probe/ (extracted images + report.md).

    DIP_EMBED_BACKEND=st DIP_DATA_DIR=data-embeddinggemma2 python experiments/image_probe.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dip import config, db, llm  # noqa: E402
from dip.query import retrieve  # noqa: E402
from tests.eval_gold import GOLD  # noqa: E402

MIN_SIDE = 200  # skip icons, rules and logos


def extract(conn, out: Path):
    imgs = []
    for item in conn.execute("SELECT id, path, title FROM items WHERE type='pdf'"):
        if not Path(item["path"]).exists():
            continue
        doc, seen = pymupdf.open(item["path"]), set()
        for pno, page in enumerate(doc, 1):
            for im in page.get_images(full=True):
                xref, w, h = im[0], im[2], im[3]
                if xref in seen or min(w, h) < MIN_SIDE:
                    continue
                seen.add(xref)
                pix = pymupdf.Pixmap(doc, xref)
                if pix.n - pix.alpha >= 4:  # CMYK → RGB
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                f = out / f"item{item['id']}_p{pno}_x{xref}.png"
                pix.save(f)
                imgs.append({"path": f, "item_id": item["id"], "page": pno, "source": item["title"]})
    return imgs


def snippet(t, n=140):
    t = " ".join(t.split())
    return t[:n] + ("…" if len(t) > n else "")


def main():
    conn = db.connect()
    out = config.DATA_DIR / "image_probe"
    out.mkdir(parents=True, exist_ok=True)
    imgs = extract(conn, out)
    print(f"extracted {len(imgs)} images ≥{MIN_SIDE}px from PDFs → {out}")
    mat = np.asarray(llm.embed_images([i["path"] for i in imgs]), dtype=np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)

    chunk_item = dict(conn.execute("SELECT id, item_id FROM nodes WHERE kind='chunk'").fetchall())
    n_items = conn.execute("SELECT count(*) FROM items").fetchone()[0]
    idx = retrieve.index(conn, "node")
    lines = [f"# Image probe: {config.EMBED_MODEL}\n", "## 1. Image → nearest passage\n"]
    top1 = top5 = 0
    for im, v in zip(imgs, mat):
        hits = idx.search(v, k=5, allowed=set(chunk_item))
        same = [chunk_item[i] == im["item_id"] for i, _ in hits]
        top1 += same[0]
        top5 += any(same)
        best = db.node(conn, hits[0][0])
        lines.append(f"![]({im['path'].name}) **{im['source']} p.{im['page']}** → "
                     f"{'✅' if same[0] else '❌'} ({hits[0][1]:.2f}) [{best['id']}] {snippet(best['text'])}\n")
    n = len(imgs)
    summary = (f"grounding: nearest passage from the image's own PDF for {top1}/{n} images ({top1 / n:.0%}); "
               f"within top 5 for {top5}/{n} ({top5 / n:.0%}); {n_items} sources in the DB")
    print(summary)
    lines.insert(1, summary + "\n")

    lines.append("\n## 2. Provocation → nearest images\n")
    provs = [p["text"] for p in json.load(open(GOLD))][:8]
    qv = np.asarray(llm.embed(provs, query=True), dtype=np.float32)
    for p, q in zip(provs, qv):
        sims = mat @ (q / np.linalg.norm(q))
        best = np.argsort(-sims)[:3]
        lines.append(f"**{snippet(p, 200)}**\n")
        lines.append(" ".join(f"![]({imgs[b]['path'].name}) {sims[b]:.2f}" for b in best) + "\n")
        print(f"- {snippet(p, 70)}\n    " + "; ".join(
            f"{imgs[b]['source'][:25]} p.{imgs[b]['page']} ({sims[b]:.2f})" for b in best))
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
