import time, requests
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "JerusalemRAG/1.0"}

KEYWORDS = [
    "Jerusalem", "Crusade", "Crusader", "Outremer", "Acre", "Hattin",
    "Baldwin", "Templar", "Hospitaller", "Latin", "Frankish"
]

def api_get(params):
    r = requests.get(API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

def get_extract(title: str) -> str:
    data = api_get({
        "action": "query", "format": "json",
        "prop": "extracts", "explaintext": 1,
        "titles": title
    })
    page = next(iter(data["query"]["pages"].values()))
    return page.get("extract", "")

def get_links(title: str, limit=80):
    out, cont = [], None
    while len(out) < limit:
        params = {
            "action": "query", "format": "json",
            "titles": title, "prop": "links",
            "plnamespace": 0, "pllimit": "max",
        }
        if cont: params["plcontinue"] = cont
        data = api_get(params)
        page = next(iter(data["query"]["pages"].values()))
        out += [l["title"] for l in page.get("links", [])]
        cont = data.get("continue", {}).get("plcontinue")
        if not cont: break
    return out[:limit]

def ok_title(t: str) -> bool:
    return any(k.lower() in t.lower() for k in KEYWORDS)

def save_corpus(seeds, max_pages=50, out_dir="data/raw/wiki"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    queue = list(seeds)
    seen = set()

    while queue and len(seen) < max_pages:
        title = queue.pop(0)
        if title in seen: 
            continue
        seen.add(title)

        try:
            text = get_extract(title)
            if text.strip():
                filename = title.replace("/", "_").replace(" ", "_")
                (out_path / f"{filename}.txt").write_text(
                    f"TITLE: {title}\n\n{text}", encoding="utf-8"
                )
                print("Saved:", title)
        except Exception as e:
            print("Skip:", title, e)

        try:
            for l in get_links(title):
                if l not in seen and ok_title(l):
                    queue.append(l)
        except:
            pass

        time.sleep(0.2)

    print("Done. Files:", len(list(out_path.glob("*.txt"))))

if __name__ == "__main__":
    seeds = [
        "Kingdom of Jerusalem",
        "Acre, Israel",
        "Baldwin IV of Jerusalem",
        "Battle of Hattin",
        "First Crusade"
    ]
    save_corpus(seeds, max_pages=60)
