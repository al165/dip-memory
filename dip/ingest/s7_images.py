"""Stage 7 — image library, searchable by subject and by metaphor.

Image files live in IMAGE_DIR (00_Data/images/). Each one has a markdown page in IMAGE_WIKI_DIR
(data/wiki/images/) with its credit, a description, its subjects and the metaphors it can carry:

    ---
    title: Mycelium under a forest floor
    file: mycelium.jpg
    source: https://commons.wikimedia.org/wiki/File:...
    creator: Jane Doe (2019)
    license: CC BY-SA 4.0
    credit: Jane Doe, Mycelium, 2019, CC BY-SA 4.0, via Wikimedia Commons
    locked: false
    ---
    # Mycelium under a forest floor
    ## Description
    ...
    ## Subjects
    - fungi, forest floor, roots
    ## Metaphors
    - networks: decentralised connection with no single centre

The page is the source of truth: the `images` table is rebuilt from the pages on every run.
A missing page is drafted: the vision model describes the picture, then the LLM proposes
metaphors. Credit comes from `<file>.source.txt` next to the image if there is one (a URL, a
credit line, or `source:`/`creator:`/`license:`/`credit:`/`description:` lines); the model
can't know it. An unlocked page is redrafted when its image file changes; set `locked: true`
once you've edited a page. Delete a page to have it drafted again.

Images are kept out of `nodes`, so they don't affect clusters, edges or the text wiki.
"""
import hashlib
import json
import re
import time
from pathlib import Path

from dip import config, db, llm
from dip.ingest.common import log, progress
from dip.ingest.s6_wiki import _is_locked, slugify

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
CREDIT_FIELDS = ("source", "creator", "license", "credit")

DESCRIBE_PROMPT = (
    "Describe this image for a research archive, as JSON: `title` (a short title), "
    "`description` (one paragraph describing what is shown, including any visible text) and "
    "`subjects` (3-8 short phrases naming the things, people, places and materials shown)."
    "{hint}"
)
DESCRIBE_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "description": {"type": "string"},
                   "subjects": {"type": "array", "items": {"type": "string"}}},
    "required": ["title", "description", "subjects"],
}

DESCRIBE_FROM_TEXT_PROMPT = """An image in an archive comes with this catalogue text:

{text}

From it, write as JSON: `title` (a short title), `description` (one paragraph describing what
the image most likely shows, without inventing details) and `subjects` (3-8 short phrases
naming the things, people, places and materials shown)."""

METAPHOR_PROMPT = """An image shows:

{title}. {description}
Subjects: {subjects}

A design researcher wants to use this image as a METAPHOR next to texts about technology,
society and culture. What ideas, relationships, structures or processes could it stand for?
Think about how its parts relate (connection, growth, hierarchy, exchange, decay, care, scale,
control, memory...), not about what it literally is.

Give `metaphors`: 4-6 items, each with `keyword` (1-3 words, e.g. "networks", "mutual aid")
and `gloss` (one short sentence saying what the image says about that idea, e.g. "many small
threads exchanging resources with no centre"). Do not just repeat the subjects."""
METAPHOR_SCHEMA = {
    "type": "object",
    "properties": {"metaphors": {"type": "array", "minItems": 3, "maxItems": 6, "items": {
        "type": "object",
        "properties": {"keyword": {"type": "string"}, "gloss": {"type": "string"}},
        "required": ["keyword", "gloss"]}}},
    "required": ["metaphors"],
}


# --- pages --------------------------------------------------------------------------------

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_source_file(img: Path) -> dict:
    """Credit (and an optional catalogue description) from `<file>.source.txt` next to the image."""
    p = img.with_name(img.name + ".source.txt")
    out = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        m = re.match(r"^(source|creator|license|credit|description):\s*(.*)$", line, flags=re.I)
        if m:
            out[m.group(1).lower()] = m.group(2).strip()
        elif re.match(r"^https?://", line) and "source" not in out:
            out["source"] = line
        elif line and "credit" not in out:
            out["credit"] = line
    return out


def parse_page(text: str) -> dict:
    """Front matter fields + description, subjects and metaphors from an image page."""
    out = {"description": "", "subjects": [], "metaphors": []}
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, flags=re.S)
    head, body = (m.group(1), m.group(2)) if m else ("", text)
    for line in head.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    sections, cur = {}, None
    for line in body.splitlines():
        h = re.match(r"^##\s+(.*)$", line)
        if h:
            cur = h.group(1).strip().lower()
            sections[cur] = []
        elif cur:
            sections[cur].append(line)
    out["description"] = " ".join(" ".join(sections.get("description", [])).split())
    for line in sections.get("subjects", []):
        if line.strip().startswith(("-", "*")):
            out["subjects"] += [s.strip() for s in line.strip()[1:].split(",") if s.strip()]
    for line in sections.get("metaphors", []):
        if line.strip().startswith(("-", "*")):
            item = line.strip()[1:].strip()
            kw, _, gloss = item.partition(":")
            if kw.strip():
                out["metaphors"].append({"keyword": kw.strip(), "gloss": gloss.strip()})
    return out


def render_page(d: dict) -> str:
    head = [f"title: {d['title']}", f"file: {d['file']}"]
    head += [f"{k}: {d.get(k) or 'TODO'}" for k in CREDIT_FIELDS]
    head.append(f"locked: {'true' if d.get('locked') == 'true' else 'false'}")
    lines = ["---", *head, "---", f"# {d['title']}", "", "## Description", d["description"], "",
             "## Subjects", "- " + ", ".join(d["subjects"]), "", "## Metaphors"]
    lines += [f"- {m['keyword']}: {m['gloss']}" for m in d["metaphors"]]
    return "\n".join(lines) + "\n"


def has_credit(d: dict) -> bool:
    return any((d.get(k) or "TODO") != "TODO" for k in ("credit", "source"))


def draft(img: Path, rel: str) -> dict:
    """Describe the image (vision model; else the .source.txt description) and propose metaphors."""
    src = read_source_file(img)
    hint = f"\nCatalogue text for context: {src['description']}" if src.get("description") else ""
    t0 = time.time()
    try:
        d = llm.generate(DESCRIBE_PROMPT.format(hint=hint), schema=DESCRIBE_SCHEMA,
                         images=[img.read_bytes()], model=config.VLM_MODEL)
        how = f"described by {config.VLM_MODEL}"
    except llm.LLMError as e:
        if not src.get("description"):
            raise
        log(f"    {img.name}: vision model unavailable ({e}); describing from .source.txt instead")
        d = llm.generate(DESCRIBE_FROM_TEXT_PROMPT.format(text=src["description"]), schema=DESCRIBE_SCHEMA)
        how = f"described from .source.txt by {config.LLM_MODEL}"
    how += f" {time.time() - t0:.1f}s"
    t0 = time.time()
    d = {"title": d["title"].strip() or img.stem, "description": " ".join(d["description"].split()),
         "subjects": [s.strip() for s in d["subjects"] if s.strip()]}
    out = llm.generate(METAPHOR_PROMPT.format(title=d["title"], description=d["description"], subjects=", ".join(d["subjects"])),
                       schema=METAPHOR_SCHEMA, temperature=0.5)
    d["metaphors"] = [{"keyword": m["keyword"].strip().lower(), "gloss": m["gloss"].strip()}
                      for m in out["metaphors"] if m["keyword"].strip()]
    how += f", {len(d['metaphors'])} metaphors by {config.LLM_MODEL} {time.time() - t0:.1f}s"
    return {**d, "file": rel, **{k: src.get(k, "") for k in CREDIT_FIELDS}, "_how": how}


def fill_credit(text: str, src: dict) -> str:
    """Put .source.txt credit into a page's empty (TODO) credit fields."""
    for k in CREDIT_FIELDS:
        if src.get(k):
            text = re.sub(rf"^{k}:\s*(TODO)?\s*$", f"{k}: {src[k]}", text, count=1, flags=re.M)
    return text


# --- stage --------------------------------------------------------------------------------

def _dur(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}m {seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def _eta(done: int, total: int, elapsed: float) -> str:
    return "" if done >= total else f" · ~{_dur(elapsed / done * (total - done))} left"


def image_files() -> list[Path]:
    if not config.IMAGE_DIR.exists():
        return []
    return sorted(p for p in config.IMAGE_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _slugs(files) -> dict[Path, str]:
    out, used = {}, set()
    for f in files:
        base = slugify(str(f.relative_to(config.IMAGE_DIR).with_suffix("")))
        slug, n = base, 2
        while slug in used:
            slug, n = f"{base}-{n}", n + 1
        used.add(slug)
        out[f] = slug
    return out


def sync_pages(conn) -> list[tuple[Path, Path]]:
    """Draft missing pages (and redraft unlocked ones whose image changed). Returns (image, page)."""
    wiki = config.IMAGE_WIKI_DIR
    wiki.mkdir(parents=True, exist_ok=True)
    old_sha = {r["slug"]: r["file_sha"] for r in conn.execute("SELECT slug, file_sha FROM images")}
    files = _slugs(image_files())
    new, changed, locked, todo = 0, 0, 0, set()
    for img, slug in files.items():
        page = wiki / f"{slug}.md"
        if not page.exists():
            new += 1
            todo.add(img)
        elif slug in old_sha and old_sha[slug] != _sha(img.read_bytes()):
            if _is_locked(page):
                locked += 1
            else:
                changed += 1
                todo.add(img)
    log(f"stage 7: {len(files)} image files in {config.IMAGE_DIR}: {len(todo)} page(s) to draft "
        f"({new} new, {changed} changed image)" + (f", {locked} changed but locked" if locked else "")
        + f", {len(files) - len(todo)} up to date")
    if todo:
        log(f"stage 7: drafting with {config.VLM_MODEL} (vision) + {config.LLM_MODEL} (metaphors) "
            f"at {config.OLLAMA_URL}")
    pairs, done, t0 = [], 0, time.time()
    for img, slug in files.items():
        page = wiki / f"{slug}.md"
        rel = str(img.relative_to(config.IMAGE_DIR))
        if img in todo:
            t1 = time.time()
            try:
                d = draft(img, rel)
                page.write_text(render_page(d), encoding="utf-8")
                done += 1
                log(f"  drafted {done}/{len(todo)} {img.name} in {time.time() - t1:.1f}s ({d.get('_how', 'drafted')})"
                    + _eta(done, len(todo), time.time() - t0))
            except llm.LLMError as e:
                done += 1
                log(f"  FAILED {done}/{len(todo)} {img.name}: could not draft a page: {e}")
                continue
        elif not _is_locked(page):
            text = page.read_text(encoding="utf-8")
            filled = fill_credit(text, read_source_file(img))
            if filled != text:
                page.write_text(filled, encoding="utf-8")
                log(f"  {img.name}: credit filled in from .source.txt")
        pairs.append((img, page))
    if todo:
        log(f"stage 7: drafting done in {_dur(time.time() - t0)}, pages in {wiki}")
    return pairs


def load_pages(conn, pairs) -> int:
    """Rebuild the images table from the pages. Unchanged pages keep their rows (and vectors)."""
    rows = {r["slug"]: r for r in conn.execute("SELECT * FROM images")}
    keep, changed = set(), 0
    with db.tx(conn):
        for img, page in pairs:
            slug, text = page.stem, page.read_text(encoding="utf-8")
            file_sha, page_sha = _sha(img.read_bytes()), _sha(text.encode())
            keep.add(slug)
            old = rows.get(slug)
            if old and old["file_sha"] == file_sha and old["page_sha"] == page_sha:
                continue
            if old:
                conn.execute("DELETE FROM images WHERE id=?", (old["id"],))
            d = parse_page(text)
            iid = conn.execute(
                "INSERT INTO images(slug, path, page, title, description, subjects_json, source, creator, "
                "license, credit, file_sha, page_sha) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (slug, str(img), str(page), d.get("title") or slug, d["description"], json.dumps(d["subjects"]),
                 *[None if (d.get(k) or "TODO") == "TODO" else d[k] for k in CREDIT_FIELDS],
                 file_sha, page_sha)).lastrowid
            conn.executemany("INSERT INTO image_metaphors(image_id, keyword, gloss) VALUES (?,?,?)",
                             [(iid, m["keyword"], m["gloss"]) for m in d["metaphors"]])
            changed += 1
        for slug, r in rows.items():
            if slug not in keep:
                conn.execute("DELETE FROM images WHERE id=?", (r["id"],))
        db.delete_orphan_vectors(conn)
    return changed


def desc_text(r) -> str:
    subjects = ", ".join(json.loads(r["subjects_json"] or "[]"))
    return f"{r['title']}\n{r['description']}\nSubjects: {subjects}"


def metaphor_text(r) -> str:
    return f"{r['keyword']}: {r['gloss']}" if r["gloss"] else r["keyword"]


def _missing(conn, owner, table):
    return conn.execute(f"SELECT t.* FROM {table} t LEFT JOIN vectors v ON v.owner=? AND v.owner_id=t.id "
                        f"WHERE v.owner_id IS NULL", (owner,)).fetchall()


def embed_missing(conn, batch: int = 8) -> int:
    desc = _missing(conn, "image_desc", "images")
    meta = _missing(conn, "image_metaphor", "image_metaphors")
    # the picture itself; only gemma puts images and text in one space
    pics = [r for r in _missing(conn, "image_visual", "images") if Path(r["path"]).exists()] \
        if config.EMBED_MULTIMODAL else []
    if not (desc or meta or pics):
        log("stage 7: all image vectors up to date")
        return 0
    log(f"stage 7: embedding {len(desc)} description(s), {len(meta)} metaphor line(s), {len(pics)} picture(s) "
        f"with {llm.embed_backend_info()}")
    if not config.EMBED_MULTIMODAL:
        log("  pictures skipped: this backend embeds text only (use st or remote for gemma)")
    for owner, rows, text in (("image_desc", desc, desc_text), ("image_metaphor", meta, metaphor_text)):
        if rows:
            t0 = time.time()
            vecs = llm.embed([text(r) for r in rows])
            with db.tx(conn):
                db.put_vectors(conn, owner, [r["id"] for r in rows], vecs)
            log(f"  {owner}: {len(rows)} in {time.time() - t0:.1f}s")
    t0 = time.time()
    for i in range(0, len(pics), batch):
        chunk = pics[i:i + batch]
        vecs = llm.embed_images([r["path"] for r in chunk])
        with db.tx(conn):
            db.put_vectors(conn, "image_visual", [r["id"] for r in chunk], vecs)
        done = i + len(chunk)
        log(f"  image_visual: {done}/{len(pics)} in {time.time() - t0:.1f}s" + _eta(done, len(pics), time.time() - t0))
    return len(desc) + len(meta) + len(pics)


def write_index(conn):
    lines = ["# Images", "", "_One page per image in the library. Edit freely; set `locked: true` "
             "to stop a page being redrafted. Credit is shown wherever an image is used._", ""]
    for r in conn.execute("SELECT * FROM images ORDER BY slug"):
        kws = ", ".join(m["keyword"] for m in conn.execute(
            "SELECT keyword FROM image_metaphors WHERE image_id=?", (r["id"],)))
        lines.append(f"- [[images/{r['slug']}]] {r['title']} · _{kws}_" + ("" if r["credit"] or r["source"]
                                                                           else " · **credit missing**"))
    (config.IMAGE_WIKI_DIR / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(conn):
    t0 = time.time()
    log(f"stage 7: database {config.DB_PATH} · embeddings: {llm.embed_backend_info()}")
    pairs = sync_pages(conn)
    changed = load_pages(conn, pairs)
    log(f"stage 7: {changed} page(s) new or edited since the last run")
    n = embed_missing(conn)
    write_index(conn)
    for r in conn.execute("SELECT slug FROM images WHERE credit IS NULL AND source IS NULL ORDER BY slug"):
        log(f"  WARNING credit missing for image {r['slug']}: fill in source/credit on its page")
    total = conn.execute("SELECT count(*) FROM images").fetchone()[0]
    log(f"stage 7: {total} images, {n} vectors embedded, done in {_dur(time.time() - t0)}")


def precompute_concepts(conn):
    """Draw the abstract concepts of every passage now, so the Images action never waits on the LLM."""
    from concurrent.futures import ThreadPoolExecutor

    from dip.query import actions
    rows = conn.execute("SELECT text FROM nodes WHERE kind='chunk'").fetchall()
    cached = {r["text"] for r in conn.execute("SELECT text FROM concept_cache WHERE model=?", (config.LLM_MODEL,))}
    todo = [r["text"] for r in rows if r["text"] not in cached]
    log(f"concepts: {len(rows) - len(todo)} cached, {len(todo)} to draw ({config.LLM_CONCURRENCY} at a time)")

    def draw(text):
        try:
            return actions.llm_concepts(text)
        except llm.LLMError as e:
            log(f"  concepts failed: {e}")
            return None

    with ThreadPoolExecutor(max_workers=config.LLM_CONCURRENCY) as pool:
        for i, (text, concepts) in enumerate(zip(todo, pool.map(draw, todo)), 1):
            if concepts:
                actions.cache_concepts(conn, text, concepts)  # written here, on the main thread
            progress(i, len(todo), "concepts", every=25)

    # their query vectors too (plus the no-LLM concepts of clusters and themes), so the Images
    # action also works on a server without the embedding model (DIP_OFFLINE)
    concepts = {c for r in conn.execute("SELECT concepts_json FROM concept_cache WHERE model=?",
                                        (config.LLM_MODEL,)) for c in json.loads(r["concepts_json"])}
    for n in conn.execute("SELECT * FROM nodes WHERE kind IN ('cluster','theme')").fetchall():
        concepts.update(actions._tree_concepts(conn, n))
    concepts = sorted(concepts)
    have = 0
    for i in range(0, len(concepts), 256):
        have += len(actions.concept_vectors(conn, concepts[i:i + 256]))
    log(f"concept vectors: {have} of {len(concepts)} stored")
