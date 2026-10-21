"""Prototype step A: extract (claim, issue, stance) positions per chunk. Reads a DB snapshot, writes positions.json."""
import json, sqlite3, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, "/home/arran/Projects/AIxDESIGN/DIP/01_Initial_Tests/dip-memory")
from dip import llm

HERE = Path(__file__).parent
ITEMS = tuple(int(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else (2, 3, 4, 6, 7, 8)
SCHEMA = {"type": "object", "properties": {"positions": {"type": "array", "maxItems": 3, "items": {
    "type": "object", "properties": {
        "claim": {"type": "string"}, "issue": {"type": "string"},
        "stance": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["claim", "issue", "stance"]}}}, "required": ["positions"]}
PROMPT = """Analyse this passage from "{doc}", section "{section}".

PASSAGE:
{text}

Return JSON with `positions`: 0-3 positions this passage takes on questions that reasonable people could disagree about. For each:
- claim: one sentence stating the position. It must stand alone: name the subject, never write "the passage", "the author", "this" or "it".
- issue: the debatable question the claim answers, as a neutral yes/no question that a different text could answer the other way. Make it general enough to be shared across texts (no names of this text's own projects, people or sections) but specific enough to be a real disagreement. Example: "Do individual digital habits, like deleting files, meaningfully reduce emissions?"
- stance: "yes" or "no", this passage's answer to the issue.

Skip plain facts, descriptions, examples without an argument, methodology, acknowledgements and statements about the text itself. Return an empty list if the passage takes no debatable position."""
SYSTEM = "You map the arguments in a research corpus. Be precise and terse."

conn = sqlite3.connect("file:" + str(HERE.parent.parent / "data" / "dip.sqlite") + "?mode=ro", uri=True); conn.row_factory = sqlite3.Row
rows = conn.execute(f"""SELECT n.id, n.item_id, n.text, n.title sec, d.title doc FROM nodes n
  JOIN nodes s ON s.id=n.parent_id JOIN nodes d ON d.id=s.parent_id
  WHERE n.kind='chunk' AND n.item_id IN ({",".join(map(str, ITEMS))}) AND coalesce(n.role,'')!='link' ORDER BY n.id""").fetchall()
out_path = HERE / (sys.argv[2] if len(sys.argv) > 2 else "positions.json")
done = json.loads(out_path.read_text()) if out_path.exists() else {}
todo = [r for r in rows if str(r["id"]) not in done]
print(f"{len(rows)} chunks, {len(todo)} to do", flush=True)
t0 = time.time()
def work(r):
    t = time.time()
    o = llm.generate(PROMPT.format(doc=r["doc"], section=r["sec"], text=r["text"]), schema=SCHEMA, system=SYSTEM)
    return r, o["positions"], time.time() - t
with ThreadPoolExecutor(4) as pool:
    futs = [pool.submit(work, r) for r in todo]
    for i, f in enumerate(as_completed(futs), 1):
        try:
            r, pos, dt = f.result()
        except Exception as e:
            print("skip", e, flush=True); continue
        done[str(r["id"])] = {"item": r["item_id"], "positions": pos, "secs": round(dt, 1)}
        if i % 25 == 0 or i == len(todo):
            out_path.write_text(json.dumps(done, indent=1))
            print(f"{i}/{len(todo)} {(time.time()-t0)/i:.2f}s/chunk wall", flush=True)
out_path.write_text(json.dumps(done, indent=1))
print("wall", round(time.time() - t0), "s")
