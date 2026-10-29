"""FastAPI server for the query actions.

uvicorn dip.api:app --port 8000        (test UI at http://localhost:8000/, API docs at /docs)
"""

import hashlib
import json
import re
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dip import config, db, llm
from dip.query import actions

app = FastAPI(
    title="DIP memory",
    version="0.1",
    description="Local-first document memory: neighbours, counter, evidence, zoom, question.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount(
    "/ui", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui"
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("ui/")  # relative, so it also works under a path prefix


def get_conn():
    conn = db.connect()
    try:
        yield conn
    finally:
        conn.close()


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except actions.NotFound as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except llm.LLMError as e:
        raise HTTPException(503, f"local model unavailable: {e}") from e


@app.get("/health")
def health(conn=Depends(get_conn)):
    counts = {
        r["kind"]: r["n"]
        for r in conn.execute("SELECT kind, count(*) n FROM nodes GROUP BY kind")
    }
    edges = {
        r["type"]: r["n"]
        for r in conn.execute("SELECT type, count(*) n FROM edges GROUP BY type")
    }
    return {
        "db": str(config.DB_PATH),
        "status": "ok",
        "nodes": counts,
        "edges": edges,
        "claims": conn.execute("SELECT count(*) FROM claims").fetchone()[0],
        "wiki_pages": conn.execute("SELECT count(*) FROM wiki_pages").fetchone()[0],
        "images": conn.execute("SELECT count(*) FROM images").fetchone()[0],
        "models": {
            "llm": config.LLM_MODEL,
            "embed": config.EMBED_MODEL,
            "nli": config.NLI_MODEL,
        },
        "ollama": llm.available(),
        "offline": config.OFFLINE,
        "editing": "token" if config.EDIT_TOKEN else "open",
    }


@app.get("/items")
def items(conn=Depends(get_conn)):
    return [
        dict(r)
        for r in conn.execute(
            "SELECT i.*, n.id AS document_node FROM items i JOIN nodes n ON n.item_id=i.id AND n.kind='document'"
        )
    ]


@app.get("/nodes")
def list_nodes(
    kind: str = "chunk", limit: int = 50, offset: int = 0, conn=Depends(get_conn)
):
    rows = conn.execute(
        "SELECT * FROM nodes WHERE kind=? ORDER BY item_id, ord, id LIMIT ? OFFSET ?",
        (kind, limit, offset),
    ).fetchall()
    return [db.node_dict(r) for r in rows]


@app.get("/nodes/{node_id}")
def get_node(node_id: int, conn=Depends(get_conn)):
    row = db.node(conn, node_id)
    if row is None:
        raise HTTPException(404, f"node {node_id} not found")
    out = db.node_dict(row)
    out["children"] = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM nodes WHERE parent_id=? OR cluster_id=? ORDER BY ord, id",
            (node_id, node_id),
        )
    ]
    out["claims"] = [
        dict(r)
        for r in conn.execute("SELECT id, text FROM claims WHERE node_id=?", (node_id,))
    ]
    out["edges"] = [
        dict(r)
        for r in conn.execute(
            "SELECT src_node, dst_node, type, weight, method FROM edges WHERE src_node=? OR dst_node=?",
            (node_id, node_id),
        )
    ]
    return out


@app.get("/search")
def search(q: str, k: int = 10, kind: str = "chunk", conn=Depends(get_conn)):
    return _run(actions.search, conn, q, k, tuple(kind.split(",")))


def _action_endpoint(name):
    fn = actions.ACTIONS[name]

    def endpoint(
        node_id: int | None = Query(None, description="node to start from"),
        text: str | None = Query(
            None, description="or free text (mapped to its closest chunk)"
        ),
        k: int = Query(config.DEFAULT_K, ge=1, le=50),
        conn=Depends(get_conn),
    ):
        return _run(fn, conn, node_id=node_id, text=text, k=k)

    endpoint.__name__ = name.replace("-", "_")
    endpoint.__doc__ = (fn.__doc__ or "").strip() or f"{name} for a node"
    return endpoint


for _name in actions.ACTIONS:
    app.get(f"/{_name}", summary=_name)(_action_endpoint(_name))


class QuestionIn(BaseModel):
    question: str
    node_id: int | None = None
    k: int = 8


@app.post("/question")
def question(body: QuestionIn, conn=Depends(get_conn)):
    return _run(actions.question, conn, body.question, body.node_id, body.k)


@app.get("/wiki")
def wiki_index(conn=Depends(get_conn)):
    return [
        {k: r[k] for k in ("id", "slug", "title", "summary", "source_node")}
        | {"node_ids": json.loads(r["node_ids_json"])}
        for r in conn.execute("SELECT * FROM wiki_pages ORDER BY id")
    ]


def _moved(path: Path, folder: Path) -> Path:
    """Stored paths are those of the machine that built the database; on another server (the
    demo) look for the same file name in the configured folder."""
    return path if path.exists() else Path(folder) / path.name


def _image_row(conn, slug):
    row = conn.execute("SELECT * FROM images WHERE slug=?", (slug,)).fetchone()
    if row is None:
        raise HTTPException(404, f"no image {slug!r}")
    return row


@app.get("/image-library")
def image_library(conn=Depends(get_conn)):
    """Every image in the library with its description, subjects, metaphors and credit."""
    metaphors = {}
    for m in conn.execute(
        "SELECT image_id, keyword, gloss FROM image_metaphors ORDER BY id"
    ):
        metaphors.setdefault(m["image_id"], []).append(
            {"keyword": m["keyword"], "gloss": m["gloss"]}
        )
    return [
        {
            "id": r["id"],
            "slug": r["slug"],
            "title": r["title"],
            "description": r["description"],
            "subjects": json.loads(r["subjects_json"] or "[]"),
            "metaphors": metaphors.get(r["id"], []),
            "credit": r["credit"],
            "source": r["source"],
            "creator": r["creator"],
            "license": r["license"],
            "url": f"/image-file/{r['slug']}",
            "page": f"/image-page/{r['slug']}",
        }
        for r in conn.execute("SELECT * FROM images ORDER BY title COLLATE NOCASE")
    ]


@app.get("/image-passages/{slug}")
def image_passages(
    slug: str, k: int = Query(config.DEFAULT_K, ge=1, le=50), conn=Depends(get_conn)
):
    """The reverse of Images: passages this image fits, by subject and by metaphor."""
    return _run(actions.image_passages, conn, slug, k)


@app.get("/image-file/{slug}", response_class=FileResponse)
def image_file(slug: str, conn=Depends(get_conn)):
    """The image file itself (only images registered by stage 7 are served)."""
    path = _moved(Path(_image_row(conn, slug)["path"]), config.IMAGE_DIR)
    if not path.exists():
        raise HTTPException(404, f"file for image {slug!r} is missing")
    return FileResponse(path)


@app.get("/image-page/{slug}", response_class=PlainTextResponse)
def image_page(slug: str, conn=Depends(get_conn)):
    """The image's wiki page: description, subjects, metaphors and credit."""
    path = _moved(Path(_image_row(conn, slug)["page"]), config.IMAGE_WIKI_DIR)
    if not path.exists():
        raise HTTPException(404, f"page for image {slug!r} is missing")
    return path.read_text(encoding="utf-8")


@app.get("/wiki/{slug}", response_class=PlainTextResponse)
def wiki_page(slug: str):
    path = (config.WIKI_DIR / f"{slug}.md").resolve()
    if path.parent != Path(config.WIKI_DIR).resolve() or not path.exists():
        raise HTTPException(404, f"no wiki page {slug!r}")
    return path.read_text(encoding="utf-8")


# --- wiki editor ---------------------------------------------------------------------
# Topic pages (one per cluster/theme, stage 6) and image pages (stage 7). Only pages the
# database knows are listed and editable; nothing is created or deleted here.


def _editable(conn, kind: str, name: str) -> Path:
    if kind == "topic":
        known = {
            Path(r["path"]).name for r in conn.execute("SELECT path FROM wiki_pages")
        }
        folder = config.WIKI_DIR
    elif kind == "image":
        known = {Path(r["page"]).name for r in conn.execute("SELECT page FROM images")}
        folder = config.IMAGE_WIKI_DIR
    else:
        raise HTTPException(404, f"no page kind {kind!r}")
    path = Path(folder) / name
    if name not in known or not path.is_file():
        raise HTTPException(404, f"no {kind} page {name!r}")
    return path


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _front(text: str, key: str):
    m = re.match(r"---\n(.*?)\n---", text, flags=re.S)
    hit = m and re.search(rf"^{key}:\s*(.*?)\s*$", m.group(1), flags=re.M)
    return hit.group(1) if hit else None


def _set_locked(text: str, locked: bool) -> str:
    """Set `locked:` in the front matter (added if missing), so a rebuild keeps the edit."""
    m = re.match(r"---\n(.*?)\n---", text, flags=re.S)
    if not m:
        return f"---\nlocked: {str(locked).lower()}\n---\n" + text
    head = m.group(1)
    if re.search(r"^locked:", head, flags=re.M):
        head = re.sub(
            r"^locked:.*$", f"locked: {str(locked).lower()}", head, flags=re.M
        )
    else:
        head += f"\nlocked: {str(locked).lower()}"
    return f"---\n{head}\n---" + text[m.end() :]


@app.get("/wiki-files")
def wiki_files(conn=Depends(get_conn)):
    """Every editable page: topic pages, then image pages."""
    out = []
    for r in conn.execute(
        "SELECT slug, title, path FROM wiki_pages ORDER BY title COLLATE NOCASE"
    ):
        p = Path(config.WIKI_DIR) / Path(r["path"]).name
        if p.is_file():
            out.append(
                {
                    "kind": "topic",
                    "name": p.name,
                    "slug": r["slug"],
                    "title": r["title"],
                    "locked": _front(p.read_text(encoding="utf-8"), "locked") == "true",
                }
            )
    for r in conn.execute(
        "SELECT slug, title, page FROM images ORDER BY title COLLATE NOCASE"
    ):
        p = Path(config.IMAGE_WIKI_DIR) / Path(r["page"]).name
        if p.is_file():
            out.append(
                {
                    "kind": "image",
                    "name": p.name,
                    "slug": r["slug"],
                    "title": r["title"],
                    "locked": _front(p.read_text(encoding="utf-8"), "locked") == "true",
                }
            )
    return out


@app.get("/wiki-files/{kind}/{name}")
def wiki_file(kind: str, name: str, conn=Depends(get_conn)):
    text = _editable(conn, kind, name).read_text(encoding="utf-8")
    return {"kind": kind, "name": name, "text": text, "hash": _hash(text)}


class WikiEdit(BaseModel):
    text: str
    base_hash: (
        str  # the hash the editor loaded; a mismatch means someone else saved first
    )
    lock: bool | None = None


@app.put("/wiki-files/{kind}/{name}")
def save_wiki_file(
    kind: str,
    name: str,
    body: WikiEdit,
    conn=Depends(get_conn),
    x_edit_token: str | None = Header(None),
):
    """Save an edited page. Topic pages: the page's title, summary and cited passages are updated
    in the database at once (Zoom out, Question); set lock to keep the edit through rebuilds.
    Image pages are read again by the next stage 7 run."""
    if config.EDIT_TOKEN and not secrets.compare_digest(
        x_edit_token or "", config.EDIT_TOKEN
    ):
        raise HTTPException(403, "editing needs the edit token")
    path = _editable(conn, kind, name)
    current = path.read_text(encoding="utf-8")
    if _hash(current) != body.base_hash:
        raise HTTPException(
            409,
            "this page changed since you opened it; reload it to see the other edit",
        )
    text = body.text.replace("\r\n", "\n")
    if body.lock is not None:
        text = _set_locked(text, body.lock)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    note = "saved; stage 7 reads image pages again on its next run"
    if kind == "topic":
        row = conn.execute(
            "SELECT * FROM wiki_pages WHERE path LIKE ?", (f"%{name}",)
        ).fetchone()
        if row is not None:
            chunks = db.ids_of_kind(conn, "chunk")
            cited = {int(x) for x in re.findall(r"\[\[chunk:(\d+)\]\]", text)} & chunks
            title = _front(text, "title") or row["title"]
            summary = next(
                (ln[2:].strip() for ln in text.splitlines() if ln.startswith("> ")),
                row["summary"],
            )
            node_ids = sorted(set(json.loads(row["node_ids_json"])) | cited)
            conn.execute(
                "UPDATE wiki_pages SET title=?, summary=?, node_ids_json=? WHERE id=?",
                (title, summary, json.dumps(node_ids), row["id"]),
            )
            try:  # the page vector, used by Question; kept as it was without an embedding model
                db.put_vectors(
                    conn, "wiki", [row["id"]], llm.embed([f"{title}\n{summary}"])
                )
            except llm.LLMError:
                pass
            conn.commit()
            db.bump_version(conn)
        locked = _front(text, "locked") == "true"
        note = (
            "saved and locked: rebuilds keep it"
            if locked
            else "saved; not locked, so the next stage 6 rebuild will rewrite it"
        )
    return {
        "kind": kind,
        "name": name,
        "hash": _hash(text),
        "note": note,
        "locked": _front(text, "locked") == "true",
    }
