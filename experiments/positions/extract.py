"""Step 1: per passage, extract role (incl. `meta`) and 0-3 positions with voice and holder.
The LLM sees the previous and next passage of the same document as context, so a view set up in
one chunk and rejected in the next is understood. Reads data/dip.sqlite read-only.

    python extract.py [model] [items] [out]      defaults: qwen3:14b 2,3,4,6,7,8 positions.json
"""
import json, sqlite3, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dip import llm

HERE = Path(__file__).parent
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3:14b"
ITEMS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "2,3,4,6,7,8").split(",")]
OUT = HERE / (sys.argv[3] if len(sys.argv) > 3 else "positions.json")
ROLES = ["claim", "evidence", "example", "definition", "provocation", "question", "caption", "link", "meta"]
SCHEMA = {"type": "object", "properties": {
    "role": {"type": "string", "enum": ROLES},
    "positions": {"type": "array", "maxItems": 3, "items": {"type": "object", "properties": {
        "statement": {"type": "string"},
        "voice": {"type": "string", "enum": ["argued", "rejected", "reported"]},
        "holder": {"type": "string"}}, "required": ["statement", "voice", "holder"]}}},
    "required": ["role", "positions"]}
PROMPT = """You are reading "{doc}". The text's overall argument, for orientation:
{thesis}

Analyse the passage marked CURRENT. The passages before and after it are context only.

[BEFORE] (section "{s0}")
{t0}

[CURRENT] (section "{s1}")
{t1}

[AFTER] (section "{s2}")
{t2}

Return JSON:
- role: the CURRENT passage's main function: claim = an argument or position; evidence = facts, data or research findings; example = a concrete case, project or story; definition; provocation = a challenging question or quote; question; caption = describes an image; link = navigation or links; meta = about the text itself rather than its subject (acknowledgements, credits, contents, overviews of what the report, series or section will cover, methodology, author bios).
- positions: 0-3 debatable positions that appear in the CURRENT passage (use the context only to understand them). For each:
  - statement: the position as one plain sentence that someone could hold. It must stand alone: name the subject, never write "the passage", "the author", "this" or "it". State the position itself, not that someone holds it: write "Data-driven predictability makes the future controllable", not "Big tech believes data makes the future controllable".
  - voice: "argued" if the text argues for this position; "rejected" if the text presents it in order to argue against or criticise it; "reported" if the text reports someone's view without taking a side. Judge the voice against the text's overall argument: when a passage describes a trend, ideology or system that the text as a whole criticises (e.g. a passage describing the drive for frictionless, predictable technology in a text that argues for friction), the positions behind that trend are "rejected", held by whoever promotes it, even if this passage only describes it.
  - holder: who holds the position: "author" if the text's author argues it, otherwise the person or group named or implied (e.g. "David Bell", "big tech", "AI developers"), or "" if unclear.
Leave positions empty for meta, link and caption passages, and for plain facts or descriptions that take no arguable position."""
SYSTEM = "You map the arguments in a research corpus. Be precise and terse."

c = sqlite3.connect("file:" + str(HERE.parent.parent / "data" / "dip.sqlite") + "?mode=ro", uri=True)
c.row_factory = sqlite3.Row
rows = c.execute(f"""SELECT n.id, n.item_id, n.text, s.title sec, d.title doc, d.text thesis FROM nodes n
    JOIN nodes s ON s.id=n.parent_id JOIN nodes d ON d.id=s.parent_id
    WHERE n.kind='chunk' AND n.item_id IN ({','.join(map(str, ITEMS))}) ORDER BY n.item_id, n.id""").fetchall()
jobs = []
for i, r in enumerate(rows):
    prev = rows[i - 1] if i and rows[i - 1]["item_id"] == r["item_id"] else None
    nxt = rows[i + 1] if i + 1 < len(rows) and rows[i + 1]["item_id"] == r["item_id"] else None
    jobs.append((r, prev, nxt))

done = json.loads(OUT.read_text()) if OUT.exists() else {}
todo = [j for j in jobs if str(j[0]["id"]) not in done]
print(f"{MODEL}: {len(jobs)} passages, {len(todo)} to do", flush=True)


def work(job):
    r, p, n = job
    t = time.time()
    o = llm.generate(PROMPT.format(doc=r["doc"], thesis=r["thesis"], s0=p["sec"] if p else "-", t0=p["text"] if p else "(start of text)",
                                   s1=r["sec"], t1=r["text"], s2=n["sec"] if n else "-",
                                   t2=n["text"] if n else "(end of text)"),
                     schema=SCHEMA, system=SYSTEM, model=MODEL)
    return r, o, time.time() - t


t0 = time.time()
with ThreadPoolExecutor(4) as pool:
    futs = [pool.submit(work, j) for j in todo]
    for i, f in enumerate(as_completed(futs), 1):
        try:
            r, o, dt = f.result()
        except llm.LLMError as e:
            print("skip", e, flush=True)
            continue
        pos = [p for p in o["positions"] if len(p["statement"]) > 10] if o["role"] not in ("meta", "link", "caption") else []
        done[str(r["id"])] = {"item": r["item_id"], "role": o["role"], "positions": pos}
        if i % 25 == 0 or i == len(todo):
            OUT.write_text(json.dumps(done, indent=1, ensure_ascii=False))
            print(f"  {i}/{len(todo)}  {(time.time() - t0) / i:.2f}s/passage", flush=True)
OUT.write_text(json.dumps(done, indent=1, ensure_ascii=False))
print(f"done in {time.time() - t0:.0f}s")
