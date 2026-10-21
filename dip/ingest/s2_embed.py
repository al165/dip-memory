"""Stage 2 — L1 index: dense vectors for every node + FTS5 keyword index."""
from dip.ingest.common import embed_missing_nodes, log, rebuild_fts


def run(conn):
    n = embed_missing_nodes(conn)
    rebuild_fts(conn)
    log(f"stage 2: embedded {n} new nodes, FTS rebuilt")
