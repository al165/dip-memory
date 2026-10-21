"""Source loaders. Each loader turns a file into a `Parsed` record: a type, a title and the
text that the chunker will structure. To support a new format, write a loader function and
register its extensions in LOADERS (see 02_Documentation/adding-data.md)."""
import datetime as dt
import functools
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from dip import config, llm


@dataclass
class Parsed:
    type: str           # text | pdf | html | web | image
    title: str | None   # None = let the chunker take the first heading
    text: str


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def load_text(path: Path) -> Parsed:
    return Parsed("text", None, path.read_text(encoding="utf-8", errors="replace"))


def load_markdown(path: Path) -> Parsed:
    """Markdown, with any YAML front matter removed from the text (its `title`, if any, is used)."""
    import frontmatter  # python-frontmatter

    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        post = frontmatter.loads(raw)
    except Exception:  # malformed front matter: index the file as it is
        return Parsed("text", None, raw)
    title = post.metadata.get("title")
    return Parsed("text", str(title).strip() if title else None, post.content)


def _has_text_layer(path: Path) -> bool:
    import pymupdf
    with pymupdf.open(path) as doc:
        return any(page.get_text().strip() for page in doc)


@functools.cache
def _pdf_converter(ocr: bool):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions(do_ocr=ocr, do_table_structure=True)
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def load_pdf(path: Path) -> Parsed:
    """PDF -> markdown via Docling: a local layout model finds headings, paragraphs, lists,
    tables and captions in reading order (multi-column pages included), and drops running
    headers/footers and page numbers. Scanned PDFs (no text layer) are OCR'd; OCR is skipped
    otherwise, since it is the slow part. Models download from Hugging Face on first use."""
    import pymupdf  # imported lazily: only needed for PDFs

    with pymupdf.open(path) as doc:
        title = (doc.metadata or {}).get("title") or None
    result = _pdf_converter(not _has_text_layer(path)).convert(path)
    text = result.document.export_to_markdown(image_placeholder="")
    text = text.replace("\ufffd", "")  # glyphs from icon fonts
    if not text.strip():
        raise ValueError(f"no text found in {path.name}")
    return Parsed("pdf", title, text)


def is_url(source) -> bool:
    return isinstance(source, str) and source.startswith(("http://", "https://"))


def _extract_html(html: str, url: str | None = None):
    """Main text of an HTML page as markdown, extracted with trafilatura (navigation, ads,
    comments and footers removed, headings, lists and tables kept), plus its metadata."""
    import trafilatura  # imported lazily: only needed for web pages

    text = trafilatura.extract(html, url=url, output_format="markdown", include_formatting=True,
                               include_links=False, include_tables=True, include_comments=False)
    return text, trafilatura.extract_metadata(html, default_url=url)


def load_html(path: Path) -> Parsed:
    """A saved web page or other local HTML file, extracted like a web article."""
    text, meta = _extract_html(path.read_text(encoding="utf-8", errors="replace"))
    if not text:
        raise ValueError(f"no main text found in {path.name}")
    return Parsed("html", (meta.title if meta else None) or None, text)


def load_url(url: str) -> Parsed:
    """Web article -> markdown. The page is fetched once and its main text extracted locally
    (see `_extract_html`). A copy is saved to data/sources/ so the indexed text can always be
    read back, even if the page changes or disappears."""
    import trafilatura

    html = trafilatura.fetch_url(url)
    if not html:
        raise ValueError(f"could not fetch {url}")
    text, meta = _extract_html(html, url)
    if not text or len(text) < 200:
        raise ValueError(f"no article text found at {url} (paywall, JS-only page or not an article?)")
    title = (meta.title if meta else None) or url

    config.SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "article"
    header = (f"---\nsource: {url}\ntitle: {title}\nauthor: {getattr(meta, 'author', '') or ''}\n"
              f"date: {getattr(meta, 'date', '') or ''}\nfetched: {dt.date.today()}\n---\n\n")
    (config.SOURCES_DIR / f"{slug}.md").write_text(header + text, encoding="utf-8")
    return Parsed("web", title, text)


IMAGE_PROMPT = (
    "Describe this image for a research archive, as JSON: `title` (a short title), "
    "`description` (a paragraph describing what is shown) and `text` (any visible text, "
    "transcribed verbatim; empty if none)."
)
IMAGE_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "description": {"type": "string"}, "text": {"type": "string"}},
    "required": ["title", "description", "text"],
}


def load_image(path: Path) -> Parsed:
    """Images are indexed through a local vision-language model: its description (+ any OCR'd
    text) becomes the item's text, so images take part in every query type."""
    out = llm.generate(IMAGE_PROMPT, schema=IMAGE_SCHEMA, images=[path.read_bytes()], model=config.VLM_MODEL)
    body = out["description"].strip()
    if out["text"].strip():
        body += "\n\nText in image: " + out["text"].strip()
    return Parsed("image", out["title"].strip() or path.stem, body)


LOADERS = {
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_text,
    ".pdf": load_pdf,
    ".html": load_html,
    ".htm": load_html,
    ".png": load_image,
    ".jpg": load_image,
    ".jpeg": load_image,
    ".webp": load_image,
}


def load(path: Path) -> Parsed:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"No loader for {path.suffix!r} ({path.name}). Supported: {sorted(LOADERS)}")
    return loader(path)
