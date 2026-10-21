"""Ingest CLI.

    python -m dip.ingest.run FILE_OR_DIR_OR_URL [...]   # add/update sources, then rebuild global layers
    python -m dip.ingest.run reading-list.urls          # a text file with one URL per line
    python -m dip.ingest.run --stages 4,5,6             # rerun some stages over what's already stored
    python -m dip.ingest.run --reset FILES...           # wipe the database (and wiki) first
    python -m dip.ingest.run --lint                     # wiki health check
    python -m dip.ingest.run --status                   # what's in the database
    python -m dip.ingest.run --dedupe [--dry-run]       # remove sources ingested twice (same content)
    python -m dip.ingest.run --rescore-summaries --stages 3,4,5,6   # redo section/document summaries + scores
    python -m dip.ingest.run --stages 7                 # image library only (00_Data/images/)
    python -m dip.ingest.run --concepts                 # precompute passage concepts for the Images action

Stages
  1 parse + chunk        per file   structural tree (document > section > chunk)
  2 embed                per node   vectors + keyword index
  3 annotate             per chunk  context prefix (contextual retrieval), claims, role,
                                    abstraction, topics; section/doc summaries
  4 tree                 global     clusters + themes (RAPTOR)
  5 edges                global     claim graph: SUPPORTS / CONTRADICTS / ELABORATES / EXAMPLE_OF
  6 wiki                 global     markdown concept pages in data/wiki/
  7 images               per image  image pages in data/wiki/images/ (description, subjects,
                                    metaphors, credit) + vectors for the Images action
"""
import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

from dip import config, db, llm, parse
from dip.ingest import s1_parse_chunk, s2_embed, s3_annotate, s4_tree, s5_edges, s6_wiki, s7_images
from dip.ingest.common import log

STAGES = {
    2: s2_embed.run,
    3: s3_annotate.run,
    4: s4_tree.run,
    5: s5_edges.run,
    6: s6_wiki.run,
    7: s7_images.run,
}


GLOBALS_KEY = "globals_built_for"


def corpus_fingerprint(conn) -> str:
    """Identifies the set of ingested files (by content), so we know whether stages 4-6
    were last completed for exactly this corpus."""
    shas = sorted(r["sha256"] for r in conn.execute("SELECT sha256 FROM items"))
    return hashlib.sha256("\n".join(shas).encode()).hexdigest()


def _meta(conn, key):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def _set_meta(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)", (key, value))
    conn.commit()


def _read_url_list(path: Path) -> list[str]:
    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split(" #")[0].strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def expand(paths: list[str]) -> list:
    """Files, folders (recursively), http(s) URLs and `.urls` files (one URL per line)."""
    out = []
    for raw in paths:
        if parse.is_url(raw):
            out.append(raw)
            continue
        p = Path(raw)
        if p.suffix.lower() == ".urls" and p.is_file():
            out += _read_url_list(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() == ".urls":
                    out += _read_url_list(f)
                elif f.suffix.lower() in parse.LOADERS:
                    out.append(f)
        elif p.exists():
            out.append(p)
        else:
            sys.exit(f"not found: {p}")
    return list(dict.fromkeys(out))


def status(conn):
    q = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    print(f"database: {config.DB_PATH}")
    print(f"items:    {q('SELECT count(*) FROM items')}")
    for kind in ("document", "section", "chunk", "cluster", "theme"):
        print(f"  {kind:9s} {q(f'SELECT count(*) FROM nodes WHERE kind={kind!r}')}")
    annotated = q("SELECT count(*) FROM nodes WHERE kind='chunk' AND abstraction IS NOT NULL")
    print(f"annotated chunks: {annotated}")
    contexts = q("SELECT count(*) FROM nodes WHERE kind='chunk' AND context <> ''")
    print(f"chunks with a context prefix: {contexts}")
    print(f"claims:   {q('SELECT count(*) FROM claims')}")
    for r in conn.execute("SELECT type, method, count(*) n FROM edges GROUP BY type, method"):
        print(f"  edge {r['type']:12s} {r['method']:9s} {r['n']}")
    print(f"wiki pages: {q('SELECT count(*) FROM wiki_pages')}")
    print(f"images:   {q('SELECT count(*) FROM images')} "
          f"({q('SELECT count(*) FROM image_metaphors')} metaphors, "
          f"{q('SELECT count(*) FROM images WHERE credit IS NULL AND source IS NULL')} without credit)")
    print(f"passage concepts cached: {q('SELECT count(*) FROM concept_cache')}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="files, folders, URLs or .urls files to ingest")
    ap.add_argument("--stages", default="1,2,3,4,5,6,7", help="comma list of stages to run (default all)")
    ap.add_argument("--reset", action="store_true", help="delete the database and wiki first")
    ap.add_argument("--force", action="store_true", help="re-ingest files even if unchanged")
    ap.add_argument("--lint", action="store_true", help="run the wiki lint pass")
    ap.add_argument("--status", action="store_true", help="print database contents and exit")
    ap.add_argument("--dedupe", action="store_true",
                    help="remove items whose content was ingested more than once, then exit")
    ap.add_argument("--dry-run", action="store_true", help="with --dedupe: only list what would be removed")
    ap.add_argument("--rescore-summaries", action="store_true",
                    help="re-summarise and re-score all sections and documents in stage 3 "
                         "(e.g. after changing the abstraction scale); chunks are left as they are")
    ap.add_argument("--concepts", action="store_true",
                    help="draw the abstract concepts of every passage now (for the Images action), then exit")
    args = ap.parse_args(argv)

    if args.reset:
        for p in (config.DB_PATH, Path(f"{config.DB_PATH}-wal"), Path(f"{config.DB_PATH}-shm")):
            p.unlink(missing_ok=True)
        # image pages are hand-edited, not generated: keep them
        for p in config.WIKI_DIR.iterdir() if config.WIKI_DIR.exists() else ():
            if p.resolve() == config.IMAGE_WIKI_DIR.resolve():
                continue
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        log("reset: database and wiki deleted (image pages kept)")

    conn = db.connect()
    if args.dedupe:
        groups = s1_parse_chunk.find_duplicates(conn)
        if not groups:
            log("no duplicate sources")
            return
        if args.dry_run:
            for keep, drop in groups:
                log(f"keep item {keep['id']} ({keep['path']})")
                for d in drop:
                    log(f"  would remove item {d['id']} ({d['path']})")
            return
        n = s1_parse_chunk.remove_duplicates(conn)
        # clusters, edges and wiki pages still include the removed passages: force a rebuild
        _set_meta(conn, GLOBALS_KEY, "")
        log(f"removed {n} duplicate item(s). Rebuild the global layers with: "
            f"python -m dip.ingest.run --stages 4,5,6")
        return
    if args.status:
        status(conn)
        return
    if args.concepts:
        s7_images.precompute_concepts(conn)
        return

    stages = sorted({int(s) for s in args.stages.split(",") if s.strip()})
    if args.rescore_summaries:
        with db.tx(conn):
            n = conn.execute("UPDATE nodes SET abstraction=NULL WHERE kind IN ('section','document')").rowcount
        log(f"marked {n} sections/documents for re-summarising in stage 3")
        if 3 not in stages:
            log("note: --rescore-summaries only takes effect when stage 3 runs (add --stages 3,4,5,6)")
    need = []
    if (any(s >= 2 for s in stages) or args.lint) and config.EMBED_BACKEND == "remote" \
            and not llm.remote_available():
        sys.exit(f"Embedding server not reachable at {config.EMBED_URL} (or not serving {config.EMBED_MODEL}) "
                 f"— start embed_server/ on the GPU machine.")
    if (any(s >= 2 for s in stages) or args.lint) and config.EMBED_BACKEND == "ollama":
        need.append(("embed", config.EMBED_MODEL))
    if any(s >= 3 for s in stages) or args.lint:
        need.append(("llm", config.LLM_MODEL))
    if need:
        ok = llm.available()
        if not ok.get("ollama"):
            sys.exit(f"Ollama is not reachable at {config.OLLAMA_URL} — start it with `ollama serve`.")
        missing = [m for k, m in need if not ok[k]]
        if missing:
            sys.exit("Missing Ollama models: " + ", ".join(missing) +
                     "  →  " + " && ".join(f"ollama pull {m}" for m in missing))

    t0 = time.time()
    if 1 in stages:
        files = expand(args.paths)
        if not files and not args.lint:
            log("stage 1: no files given")
        changed = 0
        failed = []
        for f in files:
            try:
                _, ch = s1_parse_chunk.ingest_source(conn, f, force=args.force)
            except (ValueError, OSError, llm.LLMError) as e:  # one bad source shouldn't stop the run
                log(f"FAILED {f}: {e}")
                failed.append(f)
                continue
            changed += ch
        if failed:
            log(f"{len(failed)} source(s) failed and were skipped: " + ", ".join(map(str, failed)))
        if (files and not changed and not args.force and stages == [1, 2, 3, 4, 5, 6, 7]
                and _meta(conn, GLOBALS_KEY) == corpus_fingerprint(conn)):
            # nothing new, and the last global build (4-6) completed for exactly this corpus
            log("nothing changed since the last complete build; use --stages 4,5,6 to rebuild anyway")
            stages = [2, 3, 7]  # still finish any interrupted embedding/annotation; images are cheap
    for s in stages:
        if s in STAGES:
            if s == 4:
                _set_meta(conn, GLOBALS_KEY, "")  # global layers are now being rebuilt
            log(f"--- stage {s} ---")
            STAGES[s](conn)
    if {4, 5, 6} <= set(stages):
        _set_meta(conn, GLOBALS_KEY, corpus_fingerprint(conn))
    if args.lint:
        s6_wiki.lint(conn)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
