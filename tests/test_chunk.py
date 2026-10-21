from pathlib import Path

import pytest

from dip.chunk import chunk, structure

_DATA = Path(__file__).resolve().parents[3] / "00_Data"
SMALL_AI = next((p for p in (_DATA / "ingest" / "Small AI – text only.md", _DATA / "Small AI – text only.md")
                 if p.exists()), _DATA / "Small AI – text only.md")


def test_markdown_headings():
    text = "# Doc\n\n## First\nOne paragraph here that is long enough.\n\n## Second\nAnother one."
    title, secs = structure(text)
    assert title == "Doc"
    assert [s.title for s in secs] == ["First", "Second"]


def test_definitions_block_pairs_terms():
    text = ("Intro\nSome opening text that is a sentence.\n\nDefinitions\n\nDataset\n"
            "A structured collection of data.\n\nFriction\nThe tension when forces meet.\n")
    _, secs = structure(text)
    defs = [p for s in secs for p in s.paras if p.role == "definition"]
    assert [d.text.split(":")[0] for d in defs] == ["Dataset", "Friction"]


def test_wrapped_lines_are_joined():
    text = "Title\nThis line was wrapped\nhalfway through a sentence."
    _, secs = structure(text)
    assert secs[0].paras[0].text == "This line was wrapped halfway through a sentence."


@pytest.mark.skipif(not SMALL_AI.exists(), reason="test corpus not present")
def test_small_ai_corpus():
    title, secs = structure(SMALL_AI.read_text())
    assert title == "Small AI"
    names = {s.title for s in secs}
    assert {"Introduction", "New Frontier", "Placefulness", "Conclusion"} <= names
    chunks = chunk(secs)
    assert len(chunks) > 100
    assert all(c.text.strip() for c in chunks)
    assert max(len(c.text) for c in chunks) < 1200
    assert any(c.role == "definition" for c in chunks)
    assert any(c.role == "caption" for c in chunks)


def test_wrap_after_comma_with_capitalised_next_line():
    text = ("Title\nNew perspectives—such as Small AI,\nAncestral AI, and Esoteric AI—as pathways "
            "to imagine other worlds.")
    _, secs = structure(text)
    assert len(secs[0].paras) == 1
    assert "Small AI, Ancestral AI" in secs[0].paras[0].text


def test_markdown_docs_only_use_explicit_headings():
    text = ("# Report\n\n## Findings\nShort line without stop\n\nA full paragraph follows here.\n\n"
            "## Next steps\nMore text here.")
    _, secs = structure(text)
    assert [s.title for s in secs] == ["Findings", "Next steps"]
    assert secs[0].paras[0].text.startswith("Short line without stop")


def test_markdown_markup_lists_and_tables():
    text = ("# Doc\n\n## Part\nSome **bold** and [a link](https://x.org) and ![alt](i.png) text\n"
            "wrapped over lines.\n\nThree reasons:\n\n- first\n- second\n  1. nested\n\n"
            "| a | b |\n|---|---|\n| 1 | 2 |\n\n```\n# not a heading\n```\n\n<div>raw html</div>\n")
    title, secs = structure(text)
    assert title == "Doc" and [s.title for s in secs] == ["Part"]
    paras = [p.text for p in secs[0].paras]
    assert paras[0] == "Some bold and a link and text wrapped over lines."
    assert paras[1] == "Three reasons:\n• first\n• second\n  1. nested"
    assert paras[2] == "a | b\n1 | 2"
    assert paras[3] == "```\n# not a heading\n```"
    assert len(paras) == 4


def test_markdown_setext_and_front_matter_free_text():
    text = "# Doc\n\nIntro text here.\n\nSecond\n------\n\nBody of second.\n\n## Third\nMore."
    _, secs = structure(text)
    assert [s.title for s in secs] == ["(intro)", "Second", "Third"]
