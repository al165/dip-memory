"""Seed the image library (config.IMAGE_DIR) with freely licensed images from Wikimedia Commons,
chosen to test metaphor search against this corpus. Each image gets `<file>.source.txt` with its
Commons page, creator, license, credit line and catalogue description; stage 7 reads that.

    python experiments/image_metaphor/seed_images.py      # skips images already downloaded
    python -m dip.ingest.run --stages 7
"""
import html
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dip import config  # noqa: E402

API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "dip-memory-seed/0.1 (local research prototype; image library seeding)"}
FREE = re.compile(r"public domain|^pd|cc0|cc by|cc-by", re.I)

# (slug, Commons search). Literal subjects of the corpus and images that can carry its ideas.
SEEDS = [
    ("mycelium", "Mycelium"),
    ("murmuration", "Starling murmuration"),
    ("peasant-wedding", "Pieter Bruegel the Elder The Peasant Wedding"),
    ("american-progress", "John Gast American Progress 1872 painting"),
    ("tower-of-babel", "Pieter Bruegel the Elder The Tower of Babel Vienna"),
    ("cumulus-cloud", "cumulus cloud blue sky"),
    ("server-racks", "data center server racks aisle"),
    ("submarine-cable", "submarine communications cable cross section"),
    ("open-pit-mine", "open pit copper mine aerial"),
    ("fjord-coastline", "Norway fjord coastline aerial"),
    ("allotment-garden", "allotment garden vegetable plots"),
    ("honeycomb", "honeybees on honeycomb"),
    ("jacquard-loom", "Jacquard loom punched cards"),
    ("switchboard", "telephone switchboard operators women historical"),
    ("card-catalogue", "library card catalog drawers"),
    ("panopticon", "Presidio Modelo"),
    ("river-delta", "river delta aerial satellite braided"),
    ("old-world-map", "Mercator world map 1569"),
    ("earth-at-night", "Earth at night city lights NASA"),
    ("spider-web", "spider web dew drops"),
]


def _plain(s: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", s or "")).split())


def get(client, url, **kw):
    """GET with a pause between calls and backoff on 429 (Commons rate-limits bursts)."""
    for attempt in range(5):
        time.sleep(1.5)
        r = client.get(url, **kw)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        time.sleep(10 * (attempt + 1))
    r.raise_for_status()


def find(client, query):
    r = get(client, API, params={
        "action": "query", "format": "json", "generator": "search", "gsrsearch": f"{query} filetype:bitmap",
        "gsrnamespace": 6, "gsrlimit": 10, "prop": "imageinfo", "iiprop": "url|extmetadata|mime",
        "iiurlwidth": 1280})
    pages = sorted(r.json().get("query", {}).get("pages", {}).values(), key=lambda p: p["index"])
    for p in pages:
        info = (p.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        lic = _plain(meta.get("LicenseShortName", {}).get("value", ""))
        if info.get("mime") in ("image/jpeg", "image/png") and FREE.search(lic):
            return p["title"], info, meta, lic
    return None


def main():
    out = config.IMAGE_DIR
    out.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True) as client:
        for slug, query in SEEDS:
            if any(out.glob(f"{slug}.*")):
                print(f"skip {slug} (exists)")
                continue
            hit = find(client, query)
            if not hit:
                print(f"NOT FOUND {slug}: {query}")
                continue
            title, info, meta, lic = hit
            ext = ".png" if info["mime"] == "image/png" else ".jpg"
            img = out / f"{slug}{ext}"
            img.write_bytes(get(client, info.get("thumburl") or info["url"]).content)
            artist = _plain(meta.get("Artist", {}).get("value", "")) or "unknown"
            date = _plain(meta.get("DateTimeOriginal", {}).get("value", ""))[:40]
            name = title.removeprefix("File:").rsplit(".", 1)[0]
            creator = f"{artist} ({date})" if date else artist
            credit = f"{artist}, {name}, {lic}, via Wikimedia Commons"
            desc = _plain(meta.get("ImageDescription", {}).get("value", ""))[:800]
            img.with_name(img.name + ".source.txt").write_text(
                f"source: {info['descriptionurl']}\ncreator: {creator}\nlicense: {lic}\ncredit: {credit}\n"
                f"description: {desc or name}\n", encoding="utf-8")
            print(f"ok   {slug}: {title} [{lic}]")


if __name__ == "__main__":
    main()
