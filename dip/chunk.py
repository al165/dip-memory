"""Turn raw document text into a document -> section -> chunk structure.

Handles both real markdown (`#` headings, parsed with markdown-it) and "flat" text exports like the Small AI file,
where headings are just short lines with no terminal punctuation. Recognises the recurring
"Definitions" and "Captions" blocks and pre-tags those chunks with a role.
"""
import functools
import re
from dataclasses import dataclass, field

from dip import config

BULLET_RE = re.compile(r"^\s*([•◦▪▫●○■□‣⁃\-\*→]|\d+[.)])\s*")
TERMINAL = tuple('.!?:;,"”’)…')     # a heading never ends with these
SENTENCE_END = tuple('.!?:"”’)…')   # a line ending otherwise was hard-wrapped mid-sentence
SPECIAL_BLOCKS = {"definitions": "definition", "captions": "caption", "quick links": "link"}


@dataclass
class Para:
    text: str
    role: str | None = None


@dataclass
class Section:
    title: str
    paras: list = field(default_factory=list)


@dataclass
class Chunk:
    section_idx: int
    section: str
    text: str
    role: str | None
    ord: int


def _normalise(text: str) -> str:
    return text.replace("﻿", "").replace(" ", " ").replace("\xa0", " ")


def _clean_lines(text: str) -> list[str]:
    text = _normalise(text)
    lines = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.strip() and not re.search(r"\w", line):
            continue  # separators like "———" or lone bullets
        if BULLET_RE.match(line) and not BULLET_RE.sub("", line).strip():
            continue  # empty bullet
        lines.append(line)
    return lines


def _is_heading_candidate(line: str, md_only: bool = False) -> bool:
    s = line.strip()
    if md_only:  # the document marks its headings explicitly: trust only those
        return s.startswith("#")
    if not s or s.startswith("#"):
        return bool(s)
    if BULLET_RE.match(line) or s.startswith("http") or len(s) > 60:
        return False
    if s.endswith(TERMINAL):
        return False
    if len(s.split()) > 9:
        return False
    return s[0].isupper() or s[0].isdigit() or s.lower() in SPECIAL_BLOCKS


def _paragraphs(lines: list[str], md_only: bool = False) -> list[str]:
    """Group lines into paragraphs: blank lines split, wrapped lines are re-joined,
    and bullet lists stay attached to the line that introduces them."""
    paras, cur = [], []

    def flush():
        if cur:
            paras.append("\n".join(cur).strip())
            cur.clear()

    for n, line in enumerate(lines):
        s = line.strip()
        if not s:
            flush()
            continue
        nxt = lines[n + 1].strip() if n + 1 < len(lines) else ""
        continues = bool(nxt) and nxt[0].islower() and not s.startswith("#")
        if _is_heading_candidate(line, md_only) and not continues:
            flush()
            paras.append(s)
            flush()
            continue
        if cur:
            prev = cur[-1].strip()
            if BULLET_RE.match(line) or prev.endswith(":"):
                cur.append(s)  # list items stay with the line that introduces them
                continue
            if not prev.endswith(SENTENCE_END) or s[0].islower():
                cur[-1] = cur[-1] + " " + s  # hard-wrapped line: re-join
                continue
            if BULLET_RE.match(cur[-1]) or len(prev) > 60:
                # previous line was a full paragraph (one-paragraph-per-line exports)
                flush()
        cur.append(s)
    flush()
    return paras


@functools.cache
def _md_parser():
    from markdown_it import MarkdownIt
    return MarkdownIt("commonmark").enable("table")


def _inline_text(tok) -> str:
    """Plain text of an inline token: emphasis, links and images dropped, line breaks joined."""
    parts = [c.content if c.type in ("text", "code_inline") else " "
             for c in tok.children or [] if c.type in ("text", "code_inline", "softbreak", "hardbreak")]
    return " ".join("".join(parts).split())


def _md_paragraphs(text: str) -> list[str]:
    """Block-level paragraphs of a markdown document, parsed with markdown-it (CommonMark +
    tables). Headings come out as `#` lines; markup is reduced to plain text; a list becomes one
    paragraph of `•`/`1.` lines, attached to the line that introduces it when that ends with ":";
    a table becomes one paragraph of `cell | cell` rows; code stays fenced; raw HTML is dropped."""
    paras: list[str] = []
    heading = None              # level while inside a heading
    lists: list[str] = []       # open list kinds, innermost last
    items: list[str] = []       # lines of the outermost open list
    new_item = None             # marker waiting for the next item's first text
    rows: list[list[str]] = []
    for tok in _md_parser().parse(text):
        t = tok.type
        if t == "heading_open":
            heading = int(tok.tag[1])
        elif t == "heading_close":
            heading = None
        elif t in ("bullet_list_open", "ordered_list_open"):
            lists.append(t)
        elif t == "list_item_open":
            new_item = "  " * (len(lists) - 1) + (f"{tok.info}." if lists[-1] == "ordered_list_open" else "•")
        elif t in ("bullet_list_close", "ordered_list_close"):
            lists.pop()
            if not lists and items:
                block = "\n".join(items)
                if paras and paras[-1].endswith(":") and not paras[-1].startswith("#"):
                    paras[-1] += "\n" + block   # list items stay with the line that introduces them
                else:
                    paras.append(block)
                items = []
        elif t == "tr_open":
            rows.append([])
        elif t == "table_close":
            body = "\n".join(" | ".join(r) for r in rows if any(r))
            if body:
                paras.append(body)
            rows = []
        elif t in ("fence", "code_block") and tok.content.strip():
            # kept fenced, so a line like `# comment` is never read as a heading
            (items if lists else paras).append(f"```\n{tok.content.strip()}\n```")
        elif t == "inline":
            s = _inline_text(tok)
            if rows:
                rows[-1].append(s)
            elif not s:
                continue
            elif heading:
                paras.append("#" * heading + " " + s)
            elif lists:
                if new_item is not None:
                    items.append(f"{new_item} {s}")
                    new_item = None
                else:
                    items[-1] += " " + s     # a further paragraph of the same item
            else:
                paras.append(s)
    return paras


def structure(text: str, doc_title: str | None = None) -> tuple[str, list[Section]]:
    lines = _clean_lines(text)
    # Markdown-style input (incl. converted PDFs and web articles) declares its headings and is
    # parsed as markdown; only flat text exports need the "short line without a full stop" heuristic.
    md_only = sum(1 for ln in lines if re.match(r"#{1,6} \S", ln)) >= 2
    cand = functools.partial(_is_heading_candidate, md_only=md_only)
    paras = _md_paragraphs(_normalise(text)) if md_only else _paragraphs(lines)
    if not paras:
        return doc_title or "Untitled", []
    title = doc_title
    if title is None and cand(paras[0]):
        title = paras.pop(0).lstrip("# ").strip()
    title = title or "Untitled"

    sections: list[Section] = [Section(title="(intro)")]
    mode = None          # None | definition | caption | link
    current_term = None

    def is_heading(i: int) -> bool:
        p = paras[i]
        if p.startswith("#"):
            return True
        if not cand(p):
            return False
        nxt = paras[i + 1] if i + 1 < len(paras) else ""
        return bool(nxt) and not cand(nxt)

    def run_length(i: int) -> int:
        """Number of body paragraphs after heading i, before the next heading-ish line."""
        n = 0
        for p in paras[i + 1:]:
            if cand(p):
                break
            n += 1
        return n

    for i, p in enumerate(paras):
        low = p.lstrip("# ").strip().lower()
        if low in SPECIAL_BLOCKS:
            mode = SPECIAL_BLOCKS[low]
            current_term = None
            continue
        if cand(p):
            if mode == "definition" and not p.startswith("#") and run_length(i) <= 2:
                current_term = p.strip()
                continue
            if mode == "caption" and not is_heading(i):
                continue  # stray caption labels
            if is_heading(i) or p.startswith("#") or mode is None:
                mode, current_term = None, None
                sections.append(Section(title=p.lstrip("# ").strip()))
                continue
        if mode == "definition" and current_term:
            sections[-1].paras.append(Para(f"{current_term}: {p}", "definition"))
            current_term = None
        else:
            sections[-1].paras.append(Para(p, mode))

    sections = [s for s in sections if s.paras]
    return title, sections


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z“\"])", text)
    out, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > max_chars:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def chunk(sections: list[Section], min_chars=None, max_chars=None) -> list[Chunk]:
    """Pack paragraphs into chunks of roughly min..max chars, never crossing a section or
    mixing roles. Definitions are kept one-per-chunk: they are naturally atomic."""
    min_chars = min_chars or config.CHUNK_MIN_CHARS
    max_chars = max_chars or config.CHUNK_MAX_CHARS
    out: list[Chunk] = []
    for si, sec in enumerate(sections):
        buf, buf_role = [], None

        def emit():
            if buf:
                out.append(Chunk(si, sec.title, "\n\n".join(buf), buf_role, len(out)))
                buf.clear()

        for para in sec.paras:
            pieces = _split_long(para.text, max_chars)
            for piece in pieces:
                size = sum(len(b) for b in buf)
                if buf and (para.role != buf_role or para.role == "definition"
                            or size >= min_chars or size + len(piece) > max_chars):
                    emit()
                buf_role = para.role
                buf.append(piece)
        emit()
    return out
