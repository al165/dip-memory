"""Re-embed an existing build with the current embedding settings, without any LLM calls.

    DIP_EMBED_BACKEND=st DIP_DATA_DIR=data-embeddinggemma2 python -m dip.reembed --from data

Copies <from>/dip.sqlite (and wiki/, sources/) into DIP_DATA_DIR unless it already has a DB,
then replaces every node, claim and wiki vector. Claims, edges, clusters and wiki pages are kept
as built by the source run; rerun `--stages 4,5,6` afterwards to rebuild them on the new vectors.
"""
import argparse
import shutil
import time
from pathlib import Path

from dip import config, db, llm
from dip.ingest.common import embed_missing_claims, embed_missing_nodes, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default=str(config.ROOT / "data"),
                    help="data dir of the build to copy (default: data)")
    args = ap.parse_args()
    src = Path(args.src).resolve()
    if src == config.DATA_DIR.resolve():
        raise SystemExit("set DIP_DATA_DIR to a new directory; re-embedding in place would "
                         "overwrite the source build")
    if not config.DB_PATH.exists():
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "dip.sqlite", config.DB_PATH)
        for sub in ("wiki", "sources"):
            if (src / sub).exists():
                shutil.copytree(src / sub, config.DATA_DIR / sub, dirs_exist_ok=True)
        log(f"copied {src} → {config.DATA_DIR}")

    conn = db.connect()
    log(f"re-embedding with {config.EMBED_BACKEND}:{config.EMBED_MODEL} ({config.EMBED_DIM}d)")
    t0 = time.time()
    with db.tx(conn):
        conn.execute("DELETE FROM vectors")
    log(f"nodes: {embed_missing_nodes(conn)}")
    log(f"claims: {embed_missing_claims(conn)}")
    pages = conn.execute("SELECT id, title, summary FROM wiki_pages").fetchall()
    vecs = llm.embed([f"{p['title']}\n{p['summary']}" for p in pages])  # same text as stage 6
    with db.tx(conn):
        db.put_vectors(conn, "wiki", [p["id"] for p in pages], vecs)
    db.bump_version(conn)
    log(f"wiki pages: {len(pages)}; done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
