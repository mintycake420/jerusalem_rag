"""Fetch medieval manuscripts from Gallica (BnF - Bibliothèque nationale de France).

Gallica provides free access to digitized manuscripts with OCR text.
No API key required.

Note: Gallica has strict rate limits. This fetcher uses exponential backoff
and long delays between requests to avoid 429 errors.
"""

import time
import re
import random
from pathlib import Path
import requests
from typing import Optional

# Key Crusade-related manuscripts available on Gallica
GALLICA_MANUSCRIPTS = [
    {
        "ark": "bpt6k5765751w",
        "title": "Recueil des historiens des croisades - Historiens occidentaux - Tome 1",
        "author": "Various (Latin chronicles)",
        "language": "la",
        "description": "Collection of Latin crusade chronicles including William of Tyre",
    },
    {
        "ark": "bpt6k57657528",
        "title": "Recueil des historiens des croisades - Historiens occidentaux - Tome 2",
        "author": "Various (Latin chronicles)",
        "language": "la",
        "description": "Continuation of Latin crusade chronicles",
    },
    {
        "ark": "bpt6k5765753n",
        "title": "Recueil des historiens des croisades - Historiens orientaux - Tome 1",
        "author": "Various (Arabic chronicles)",
        "language": "ar",
        "description": "Arabic sources on the Crusades with French translations",
    },
    {
        "ark": "bpt6k5765769s",
        "title": "Recueil des historiens des croisades - Historiens grecs - Tome 1",
        "author": "Various (Byzantine chronicles)",
        "language": "el",
        "description": "Byzantine Greek sources on the Crusades",
    },
    {
        "ark": "bpt6k5454923d",
        "title": "Historia rerum in partibus transmarinis gestarum (William of Tyre)",
        "author": "William of Tyre",
        "language": "la",
        "description": "William of Tyre's history of the Kingdom of Jerusalem in Latin",
    },
    {
        "ark": "bpt6k111294n",
        "title": "Gesta Francorum et aliorum Hierosolimitanorum",
        "author": "Anonymous",
        "language": "la",
        "description": "Anonymous account of the First Crusade",
    },
    {
        "ark": "bpt6k5765755c",
        "title": "Fulcherii Carnotensis Historia Hierosolymitana",
        "author": "Fulcher of Chartres",
        "language": "la",
        "description": "Fulcher of Chartres' chronicle of the First Crusade",
    },
]


class GallicaFetcher:
    """Fetch manuscripts from Gallica's free OCR text endpoint."""

    BASE_URL = "https://gallica.bnf.fr"

    def __init__(self, out_dir: str = "data/raw/gallica"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "JerusalemRAG/2.0 (scholarly research; contact: github.com/jerusalem_rag)"

    def get_text_url(self, ark: str) -> str:
        """Get URL for plain text OCR of a document."""
        return f"{self.BASE_URL}/ark:/12148/{ark}/texteBrut"

    def get_document_url(self, ark: str) -> str:
        """Get URL for viewing the document."""
        return f"{self.BASE_URL}/ark:/12148/{ark}"

    def fetch_text(self, ark: str, timeout: int = 90, max_retries: int = 5) -> Optional[str]:
        """Fetch OCR text for a Gallica document with retry logic.

        Uses exponential backoff to handle rate limiting (429 errors).
        """
        url = self.get_text_url(ark)

        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, timeout=timeout)

                # Handle rate limiting with exponential backoff
                if resp.status_code == 429:
                    wait_time = (2 ** attempt) * 30 + random.uniform(5, 15)
                    print(f"  Rate limited (429). Waiting {wait_time:.0f}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue

                resp.raise_for_status()

                # Gallica returns HTML with text in <pre> or plain text
                text = resp.text
                # Try to extract text from HTML if present
                if "<pre>" in text.lower():
                    match = re.search(r"<pre[^>]*>(.*?)</pre>", text, re.DOTALL | re.IGNORECASE)
                    if match:
                        text = match.group(1)
                # Clean HTML entities
                text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
                return text.strip()

            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 10 + random.uniform(1, 5)
                    print(f"  Error: {e}. Retrying in {wait_time:.0f}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Error fetching {ark} after {max_retries} attempts: {e}")
                    return None

        return None

    def save_manuscript(self, manuscript: dict) -> Optional[Path]:
        """Download and save a manuscript with metadata header."""
        ark = manuscript["ark"]
        print(f"Fetching: {manuscript['title']}...")

        text = self.fetch_text(ark)
        if not text:
            print(f"  Failed to fetch text for {ark}")
            return None

        # Skip if too short (likely failed OCR)
        if len(text) < 1000:
            print(f"  Text too short ({len(text)} chars), skipping")
            return None

        # Build metadata header
        header = f"""TITLE: {manuscript['title']}
AUTHOR: {manuscript.get('author', 'Unknown')}
LANGUAGE: {manuscript['language']}
SOURCE: Gallica (BnF)
ARK: {ark}
URL: {self.get_document_url(ark)}
DESCRIPTION: {manuscript.get('description', '')}

---

"""
        # Save file
        safe_name = re.sub(r"[^\w\-]", "_", ark)
        filepath = self.out_dir / f"{safe_name}.txt"
        filepath.write_text(header + text, encoding="utf-8")
        print(f"  Saved: {filepath.name} ({len(text):,} chars)")
        return filepath

    def fetch_all(self, delay: float = 30.0) -> list[Path]:
        """Fetch all configured manuscripts with conservative rate limiting.

        Args:
            delay: Seconds to wait between requests (default: 30s to avoid 429s)
        """
        saved = []
        total = len(GALLICA_MANUSCRIPTS)

        print(f"Fetching {total} manuscripts from Gallica...")
        print(f"Using {delay}s delay between requests to avoid rate limiting.")
        print("This will take a while. Be patient!\n")

        for i, manuscript in enumerate(GALLICA_MANUSCRIPTS):
            print(f"[{i+1}/{total}] ", end="")
            path = self.save_manuscript(manuscript)
            if path:
                saved.append(path)

            # Don't wait after the last one
            if i < total - 1:
                # Add some randomness to seem more human
                actual_delay = delay + random.uniform(5, 15)
                print(f"  Waiting {actual_delay:.0f}s before next request...")
                time.sleep(actual_delay)

        print(f"\nFetched {len(saved)}/{total} manuscripts")
        return saved

    def fetch_by_ark(self, ark: str, title: str = "Unknown", language: str = "la") -> Optional[Path]:
        """Fetch a specific manuscript by ARK identifier."""
        manuscript = {
            "ark": ark,
            "title": title,
            "language": language,
        }
        return self.save_manuscript(manuscript)


if __name__ == "__main__":
    fetcher = GallicaFetcher()
    fetcher.fetch_all()
