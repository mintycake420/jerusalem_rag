import time, requests
from pathlib import Path

SEARCH = "https://archive.org/advancedsearch.php"
META = "https://archive.org/metadata/"
HEADERS = {"User-Agent": "JerusalemRAG/1.0"}

def search_items(q, rows=40):
    params = {
        "q": q,
        "fl[]": ["identifier", "title"],
        "rows": rows,
        "page": 1,
        "output": "json"
    }
    r = requests.get(SEARCH, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()["response"]["docs"]

def get_metadata(identifier):
    r = requests.get(META + identifier, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

def is_english(meta) -> bool:
    lang = meta.get("metadata", {}).get("language")
    if isinstance(lang, str):
        langs = [lang]
    else:
        langs = lang or []
    langs = [str(l).lower() for l in langs]
    return any(l.startswith("en") or l == "eng" or "english" in l for l in langs)

def download_txt(identifier, out_dir: Path) -> bool:
    meta = get_metadata(identifier)
    if not is_english(meta):
        print("Skip non-English:", identifier)
        return False
    files = meta.get("files", [])
    txts = [f for f in files if str(f.get("name","")).lower().endswith(".txt")]
    if not txts:
        return False

    fname = txts[0]["name"]
    url = f"https://archive.org/download/{identifier}/{fname}"
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()

    title = meta.get("metadata", {}).get("title", identifier)
    (out_dir / f"{identifier}.txt").write_text(
        f"TITLE: {title}\nIDENTIFIER: {identifier}\nURL: {url}\n\n" + r.text,
        encoding="utf-8"
    )
    print("Downloaded:", title)
    return True

def build(out_dir="data/raw/archive", max_items=20):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    queries = [
        '("William of Tyre" OR "Historia rerum in partibus transmarinis") AND mediatype:texts',
        '("Fulcher of Chartres" OR "Gesta Francorum") AND mediatype:texts',
        '("Kingdom of Jerusalem" AND crusade) AND mediatype:texts',
    ]

    saved = 0
    seen = set()

    for q in queries:
        for d in search_items(q):
            if saved >= max_items: break
            ident = d.get("identifier")
            if not ident or ident in seen: 
                continue
            seen.add(ident)

            try:
                if download_txt(ident, out):
                    saved += 1
            except Exception as e:
                print("Skip:", ident, e)

            time.sleep(0.3)

    print("Done. Saved:", saved)

if __name__ == "__main__":
    build()
