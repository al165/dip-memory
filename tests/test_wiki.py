"""Stage 6 housekeeping: pages of clusters that no longer exist."""
from dip.ingest import s6_wiki


def test_stale_pages_removed_locked_kept(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "mycelium.md").write_text("image page")       # other folder: untouched
    for name in ("current", "old-cluster", "old-but-edited", "index", "log", "lint"):
        (tmp_path / f"{name}.md").write_text("---\nlocked: true\n---\n" if name == "old-but-edited" else "x")
    removed, kept = s6_wiki.remove_stale_pages(tmp_path, {"current"})
    assert removed == ["old-cluster.md"] and kept == ["old-but-edited.md"]
    assert sorted(p.name for p in tmp_path.glob("*.md")) == \
        ["current.md", "index.md", "lint.md", "log.md", "old-but-edited.md"]
    assert (tmp_path / "images" / "mycelium.md").exists()


def test_nothing_removed_when_image_pages_share_the_folder(tmp_path, monkeypatch):
    from dip import config
    monkeypatch.setattr(config, "IMAGE_WIKI_DIR", tmp_path)
    (tmp_path / "mycelium.md").write_text("image page")
    assert s6_wiki.remove_stale_pages(tmp_path, {"current"}) == ([], [])
    assert (tmp_path / "mycelium.md").exists()
