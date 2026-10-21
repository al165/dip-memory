"""The ingest CLI must not skip global stages (4-6) after an interrupted run."""
from dip import db
from dip.ingest import run


def _calls(monkeypatch):
    calls = []
    for s in run.STAGES:
        monkeypatch.setitem(run.STAGES, s, lambda conn, s=s: calls.append(s))
    monkeypatch.setattr(run.llm, "available", lambda: {"ollama": True, "llm": True, "embed": True})
    return calls


def test_unchanged_file_after_interrupted_run_still_builds_globals(tmp_path, monkeypatch):
    monkeypatch.setattr(run.config, "DB_PATH", tmp_path / "t.sqlite")
    monkeypatch.setattr(run.config, "DATA_DIR", tmp_path)
    real_connect = db.connect
    monkeypatch.setattr(run.db, "connect", lambda path=None: real_connect(tmp_path / "t.sqlite"))
    f = tmp_path / "doc.md"
    f.write_text("# Doc\n\n## Part\nA paragraph that is long enough to be a chunk of text.\n")

    calls = _calls(monkeypatch)
    run.main(["--stages", "1,2,3", str(f)])            # simulates a run interrupted before stage 4
    calls.clear()
    run.main([str(f)])                                   # file unchanged, but globals never built
    assert calls == [2, 3, 4, 5, 6, 7]

    calls.clear()
    run.main([str(f)])                                   # now genuinely nothing to do
    assert calls == [2, 3, 7]                            # stage 7 (images) is cheap and always runs
