"""Focused translation script for Assises de Jérusalem.

Run overnight to translate the medieval French legal texts.
Usage: python translate_assises.py
"""

import json
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from rag.translation import Translator
from rag.injest_v2 import MultilingualIngestor

# Priority documents to translate
PRIORITY_DOCS = [
    "AssisesDeJerusalemBeugnotVol1.txt",
    "AssisesDeJerusalemBeugnotVol2.txt",
    "hetum.txt",  # Hayton's chronicle
]


def main():
    print("=" * 60)
    print("Assises de Jérusalem Translation Script")
    print("=" * 60)

    # Load and chunk just the priority documents
    ingestor = MultilingualIngestor(
        data_dir="data/raw/archive_v2",
        translate_non_english=False,  # We'll translate manually
    )

    # Load files
    data_dir = Path("data/raw/archive_v2")
    all_chunks = []

    for doc_name in PRIORITY_DOCS:
        filepath = data_dir / doc_name
        if filepath.exists():
            print(f"\nLoading: {doc_name}")
            chunks = ingestor.load_file(filepath)
            print(f"  Created {len(chunks)} chunks")
            all_chunks.extend(chunks)
        else:
            print(f"  Not found: {doc_name}")

    print(f"\nTotal chunks to translate: {len(all_chunks)}")

    # Filter to non-English chunks
    non_english = [c for c in all_chunks if c.get("language", "en") != "en"]
    print(f"Non-English chunks: {len(non_english)}")

    if not non_english:
        print("No non-English chunks found!")
        return

    # Initialize translator
    translator = Translator(requests_per_minute=4)

    # Check how many are already cached
    cached_count = 0
    for chunk in non_english:
        if translator.cache.get(chunk["text"], chunk.get("language", "fr")):
            cached_count += 1

    print(f"Already cached: {cached_count}")
    print(f"Need to translate: {len(non_english) - cached_count}")

    if cached_count == len(non_english):
        print("\nAll chunks already translated!")
        return

    print(f"\nStarting translation (4 req/min = ~15s per chunk)...")
    print("Estimated time:", f"{(len(non_english) - cached_count) * 15 / 60:.1f} minutes")
    print("Press Ctrl+C to stop (progress is saved)\n")

    # Translate with progress bar
    translated_count = 0
    skipped_count = 0

    for chunk in tqdm(non_english, desc="Translating"):
        lang = chunk.get("language", "fr")
        text = chunk["text"]

        # Check cache first (translator does this too, but we want accurate counts)
        if translator.cache.get(text, lang):
            skipped_count += 1
            continue

        # Translate
        result = translator.translate(text, lang)
        if result:
            translated_count += 1

    print(f"\n{'=' * 60}")
    print(f"Translation complete!")
    print(f"  Newly translated: {translated_count}")
    print(f"  Skipped (cached): {skipped_count}")
    print(f"  Total cached now: {cached_count + translated_count}")
    print(f"\nRun 'python -m rag.injest_v2' to rebuild the index with translations.")


if __name__ == "__main__":
    main()
