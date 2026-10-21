"""Label a test set's passage pairs yourself. Your labels override Claude's in the eval scripts.

    python label.py                      # the representative test set (../testset/gold.json), unlabelled pairs
    python label.py --all                # every pair, showing Claude's label after you answer
    python label.py --gold gold.json     # an older test set (this folder's, or ../positions/gold.json)

Keys: o opposed · q qualifies (partial pushback) · c compatible / same side · d different question
      s supports · enter skip · x save and quit
"""

import json
import sqlite3
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).parent
KEYS = {"o": "O", "q": "Q", "c": "C", "d": "D", "s": "S"}

gold_path = (
    Path(sys.argv[sys.argv.index("--gold") + 1]).resolve()
    if "--gold" in sys.argv
    else HERE.parent / "testset" / "gold.json"
)
gold = json.loads(gold_path.read_text())
db = sqlite3.connect(
    "file:" + str(HERE.parent.parent / "data" / "dip.sqlite") + "?mode=ro", uri=True
)

print(f"\nTest set: {gold_path}")
if gold.get("rule"):
    print("\nRule: " + textwrap.fill(gold["rule"], 94, subsequent_indent="      "))
print("""
Keys: o opposed · q qualifies (partial pushback) · c compatible / same side · d different question
      s supports · enter skip · x save and quit
""")


def show(nid):
    text, sec, doc = db.execute(
        """SELECT n.text, s.title, i.title FROM nodes n JOIN nodes s ON s.id=n.parent_id
                                   JOIN items i ON i.id=n.item_id WHERE n.id=?""",
        (nid,),
    ).fetchone()
    body = "\n".join(
        textwrap.fill(p, 100) for p in text.strip().split("\n") if p.strip()
    )
    return f"[{doc[:50]} › {sec}]  node {nid}\n{body}"


todo = [it for it in gold["items"] if "--all" in sys.argv or not it["user"]]
for n, it in enumerate(todo, 1):
    a, b = it["pair"]
    print("\n" + "=" * 100 + f"\n{n}/{len(todo)}\n\nA {show(a)}\n\nB {show(b)}\n")
    k = input("o/q/c/d/s, enter = skip, x = quit > ").strip().lower()

    if k == "x":
        break

    if k in KEYS:
        it["user"] = KEYS[k]

        print(
            f"  saved {gold['labels'][it['user']]}  (Claude said {gold['labels'][it['claude']]}"
            + (f": {it['note']}" if it.get("note") else "")
            + ")"
        )
        gold_path.write_text(json.dumps(gold, indent=1, ensure_ascii=False))
print(
    f"\n{sum(1 for it in gold['items'] if it['user'])}/{len(gold['items'])} labelled by you"
)
