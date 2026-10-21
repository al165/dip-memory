"""Same content is never indexed twice: copies are skipped, moves are repointed, and
--dedupe cleans up databases built before this check existed."""
import shutil

from dip import db
from dip.ingest import run, s1_parse_chunk

DOC = "# Essay\n\n## Part\nCommunity data trusts keep power local and accountable to people.\n"


def _items(conn):
    return [(r["id"], r["path"]) for r in conn.execute("SELECT id, path FROM items ORDER BY id")]


def test_copy_is_skipped_and_move_is_repointed(tmp_path):
    conn = db.connect(tmp_path / "t.sqlite")
    a = tmp_path / "a.md"
    a.write_text(DOC)
    item, _ = s1_parse_chunk.ingest_source(conn, a)

    copy = tmp_path / "sub" / "a.md"
    copy.parent.mkdir()
    shutil.copy(a, copy)
    assert s1_parse_chunk.ingest_source(conn, copy) == (item, False)   # copy: skipped
    assert _items(conn) == [(item, str(a.resolve()))]

    a.unlink()                                                            # now it's a move
    assert s1_parse_chunk.ingest_source(conn, copy) == (item, False)
    assert _items(conn) == [(item, str(copy.resolve()))]
    assert conn.execute("SELECT count(*) FROM nodes WHERE kind='chunk'").fetchone()[0] == 1


def test_dedupe_keeps_existing_file_and_forces_global_rebuild(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "t.sqlite")
    old, new = tmp_path / "old.md", tmp_path / "new" / "old.md"
    new.parent.mkdir()
    old.write_text(DOC)
    shutil.copy(old, new)
    first, _ = s1_parse_chunk.ingest_source(conn, old)
    second, _ = s1_parse_chunk.ingest_source(conn, new, force=True)     # simulate the pre-fix duplicate
    assert len(_items(conn)) == 2
    old.unlink()                                                          # the file was moved

    run._set_meta(conn, run.GLOBALS_KEY, run.corpus_fingerprint(conn))
    real_connect = db.connect
    monkeypatch.setattr(run.db, "connect", lambda path=None: real_connect(tmp_path / "t.sqlite"))
    run.main(["--dedupe", "--dry-run"])
    assert len(_items(conn)) == 2                                         # dry run changes nothing
    run.main(["--dedupe"])
    assert _items(conn) == [(second, str(new.resolve()))]                # kept the copy that exists
    assert run._meta(conn, run.GLOBALS_KEY) == ""                         # next run rebuilds 4-6
    assert conn.execute("SELECT count(*) FROM nodes WHERE kind='chunk'").fetchone()[0] == 1
