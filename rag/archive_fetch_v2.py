"""Fetch multilingual Crusade manuscripts from Archive.org.

This fetcher downloads Latin, French, Arabic, and Greek sources from the
"Recueil des historiens des croisades" collection and related materials.

Unlike the v1 fetcher, this one does NOT filter by English only.
"""

import time
import requests
from pathlib import Path
from typing import Optional

SEARCH = "https://archive.org/advancedsearch.php"
META = "https://archive.org/metadata/"
HEADERS = {"User-Agent": "JerusalemRAG/2.0 (scholarly research)"}

# Curated list of multilingual Crusade sources on Archive.org
CRUSADE_MANUSCRIPTS = [
    # Historiens occidentaux (Latin chronicles)
    {
        "identifier": "RecueilDesHistoriensDesCroisadesOcc4",
        "language": "la",
        "description": "Latin chronicles including William of Tyre",
    },
    {
        "identifier": "RecueilDesHistoriensDesCroisadesOccidentaux12",
        "language": "la",
        "description": "Latin chronicles Vol 1-2",
    },
    {
        "identifier": "RecueilDesHistoriensDesCroisadesOccidentaux2",
        "language": "la",
        "description": "Latin chronicles Vol 2",
    },
    {
        "identifier": "RecueilDesHistoriensDesCroisadesOccidentaux11",
        "language": "la",
        "description": "Latin chronicles Vol 1 part 1",
    },
    {
        "identifier": "RecueilDesHistoriensDesCroisadesOccidentaux3",
        "language": "la",
        "description": "Latin chronicles Vol 3",
    },
    # Historiens orientaux (Arabic sources with French translations)
    {
        "identifier": "recueildeshistor01acad",
        "language": "ar",
        "description": "Arabic chronicles Vol 1 - Ibn al-Athir and others",
    },
    {
        "identifier": "recueildeshistor02acad",
        "language": "ar",
        "description": "Arabic chronicles Vol 2",
    },
    {
        "identifier": "recueildeshistor03acad",
        "language": "ar",
        "description": "Arabic chronicles Vol 3",
    },
    {
        "identifier": "recueildeshistor04acad_0",
        "language": "ar",
        "description": "Arabic chronicles Vol 4",
    },
    {
        "identifier": "recueildeshistor05acad_0",
        "language": "ar",
        "description": "Arabic chronicles Vol 5",
    },
    # Historiens grecs (Byzantine Greek sources)
    {
        "identifier": "ldpd_10824499_002",
        "language": "el",
        "description": "Byzantine Greek chronicles",
    },
    {
        "identifier": "RecueilDesHistoriensDesCroisadesGrecs1",
        "language": "el",
        "description": "Greek historians Vol 1 - Anna Comnena and others",
    },
    # Documents arméniens (Armenian sources)
    {
        "identifier": "RecueilDesHistoriensDesCroisadesDocumentsArmeniensTomePremier",
        "language": "hy",
        "description": "Armenian documents Vol 1",
    },
    {
        "identifier": "RecueilDesHistoriensDesCroisadesDocumentsArmeniensTomeSecond",
        "language": "hy",
        "description": "Armenian documents Vol 2",
    },
    # Assises de Jérusalem (Old French legal texts)
    {
        "identifier": "AssisesDeJerusalemBeugnotVol1",
        "language": "fr",
        "description": "Laws of the Kingdom of Jerusalem Vol 1 (Old French)",
    },
    {
        "identifier": "AssisesDeJerusalemBeugnotVol2",
        "language": "fr",
        "description": "Laws of the Kingdom of Jerusalem Vol 2 (Old French)",
    },
    # Hayton's Flor des estoires
    {
        "identifier": "hetum",
        "language": "fr",
        "description": "Hayton - La Flor des estoires de la terre d'Orient",
    },
]


def get_metadata(identifier: str) -> dict:
    """Fetch metadata for an Archive.org item."""
    r = requests.get(META + identifier, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def get_language_name(code: str) -> str:
    """Get full language name from code."""
    names = {
        "la": "Latin",
        "ar": "Arabic",
        "el": "Greek",
        "fr": "French",
        "hy": "Armenian",
        "en": "English",
    }
    return names.get(code, code)


def download_manuscript(
    identifier: str,
    language: str,
    description: str,
    out_dir: Path,
) -> Optional[Path]:
    """Download a manuscript's text file from Archive.org."""
    print(f"Fetching: {identifier}...")

    try:
        meta = get_metadata(identifier)
    except Exception as e:
        print(f"  Failed to get metadata: {e}")
        return None

    # Find text files
    files = meta.get("files", [])
    txts = [f for f in files if str(f.get("name", "")).lower().endswith(".txt")]

    if not txts:
        print(f"  No text file found for {identifier}")
        return None

    # Download first text file (usually the OCR djvu.txt)
    fname = txts[0]["name"]
    url = f"https://archive.org/download/{identifier}/{fname}"

    try:
        r = requests.get(url, headers=HEADERS, timeout=120)
        r.raise_for_status()
        text = r.text
    except Exception as e:
        print(f"  Failed to download: {e}")
        return None

    # Skip if too short
    if len(text) < 1000:
        print(f"  Text too short ({len(text)} chars), skipping")
        return None

    # Get title from metadata
    title = meta.get("metadata", {}).get("title", identifier)
    if isinstance(title, list):
        title = title[0]

    # Build header with metadata
    header = f"""TITLE: {title}
IDENTIFIER: {identifier}
LANGUAGE: {language}
LANGUAGE_NAME: {get_language_name(language)}
SOURCE: Archive.org
URL: https://archive.org/details/{identifier}
DESCRIPTION: {description}

---

"""

    # Save file
    filepath = out_dir / f"{identifier}.txt"
    filepath.write_text(header + text, encoding="utf-8")
    print(f"  Saved: {filepath.name} ({len(text):,} chars)")
    return filepath


def fetch_all(out_dir: str = "data/raw/archive_v2", delay: float = 1.0) -> list[Path]:
    """Fetch all curated multilingual manuscripts.

    Args:
        out_dir: Directory to save files
        delay: Seconds between requests (Archive.org is generally more permissive)

    Returns:
        List of saved file paths
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved = []
    total = len(CRUSADE_MANUSCRIPTS)

    print(f"Fetching {total} multilingual manuscripts from Archive.org...")
    print(f"Languages: Latin, Arabic, Greek, Armenian, French\n")

    for i, manuscript in enumerate(CRUSADE_MANUSCRIPTS):
        print(f"[{i+1}/{total}] ", end="")

        path = download_manuscript(
            identifier=manuscript["identifier"],
            language=manuscript["language"],
            description=manuscript.get("description", ""),
            out_dir=out,
        )

        if path:
            saved.append(path)

        if i < total - 1:
            time.sleep(delay)

    # Summary by language
    print(f"\nFetched {len(saved)}/{total} manuscripts")

    lang_counts = {}
    for m in CRUSADE_MANUSCRIPTS:
        lang = m["language"]
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    print("By language:")
    for lang, count in lang_counts.items():
        print(f"  {get_language_name(lang)}: {count}")

    return saved


if __name__ == "__main__":
    fetch_all()
