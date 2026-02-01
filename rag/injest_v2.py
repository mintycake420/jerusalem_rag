"""Multilingual ingestion pipeline for Jerusalem RAG v2.

Handles:
- Multiple source languages (Latin, Arabic, Greek, English)
- Pre-translation during ingestion
- Enhanced metadata per chunk
- Bilingual storage (original + translated text)
"""

import re
import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

# Load .env file for GEMINI_API_KEY
load_dotenv()

from .models import ChunkMeta, get_language_name
from .translation import Translator, is_english


def extract_metadata_from_header(text: str) -> dict:
    """Extract metadata from document header."""
    metadata = {}
    lines = text[:2000].split("\n")

    for line in lines:
        if line.startswith("---"):
            break
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()
            if key == "title":
                metadata["title"] = value
            elif key == "author":
                metadata["author"] = value
            elif key == "language":
                metadata["language"] = value
            elif key == "url":
                metadata["source_url"] = value
            elif key == "source":
                metadata["source_repository"] = value.lower()

    return metadata


def detect_repository(filepath: Path) -> str:
    """Detect source repository from file path."""
    path_str = str(filepath).lower()
    if "gallica" in path_str:
        return "gallica"
    if "archive" in path_str:
        return "archive"
    if "wiki" in path_str:
        return "wiki"
    if "vatican" in path_str:
        return "vatican"
    return "unknown"


class MultilingualIngestor:
    """Ingest multilingual documents with translation support."""

    def __init__(
        self,
        data_dir: str = "data/raw",
        index_dir: str = "data/index_v2",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 2000,
        overlap: int = 300,
        translate_non_english: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.translate_non_english = translate_non_english

        self.model = SentenceTransformer(model_name)
        self.translator = Translator() if translate_non_english else None

        self.chunks: list[dict] = []
        self.embeddings = None
        self.index = None

    def chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        i = 0
        while i < len(text):
            part = text[i : i + self.chunk_size].strip()
            if part:
                chunks.append(part)
            i += self.chunk_size - self.overlap
        return chunks

    def load_file(self, filepath: Path) -> list[dict]:
        """Load a single file and create chunks with metadata."""
        text = filepath.read_text(encoding="utf-8", errors="ignore")

        # Extract metadata from header
        metadata = extract_metadata_from_header(text)
        repository = detect_repository(filepath)

        # Detect or use declared language
        declared_lang = metadata.get("language", "").lower()
        if declared_lang in ("la", "latin"):
            lang = "la"
        elif declared_lang in ("ar", "arabic"):
            lang = "ar"
        elif declared_lang in ("el", "greek"):
            lang = "el"
        elif declared_lang in ("fr", "french"):
            lang = "fr"
        elif is_english(text):
            lang = "en"
        else:
            # Try to detect
            lang = self.translator.detect_language(text) if self.translator else "en"

        # Create chunks
        prefix = re.sub(r"[^\w\-]", "_", filepath.stem)
        raw_chunks = self.chunk_text(text)

        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk = {
                "chunk_id": f"{prefix}_chunk_{i:03d}",
                "source": str(filepath),
                "text": chunk_text,
                "language": lang,
                "language_name": get_language_name(lang),
                "is_translation": False,
                "original_language": None,
                "original_text": None,
                "author": metadata.get("author"),
                "title": metadata.get("title"),
                "source_url": metadata.get("source_url"),
                "source_repository": repository,
            }
            chunks.append(chunk)

        return chunks

    def load_files(self):
        """Load all text files from data directory."""
        files = list(self.data_dir.rglob("*.txt"))
        print(f"Found {len(files)} text files")

        for filepath in tqdm(files, desc="Loading files"):
            file_chunks = self.load_file(filepath)
            self.chunks.extend(file_chunks)

        print(f"Created {len(self.chunks)} chunks")

        # Count by language
        lang_counts = {}
        for c in self.chunks:
            lang = c.get("language", "en")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        print("Chunks by language:", lang_counts)

    def translate_chunks(self, max_chunks: int = None):
        """Translate non-English chunks.

        Args:
            max_chunks: Maximum number of chunks to translate (None = all).
                       With Gemini 3 Flash free tier (20 RPD), use ~15-18 max.
        """
        if not self.translator:
            print("Translation disabled, skipping")
            return

        non_english = [c for c in self.chunks if c.get("language", "en") != "en"]
        if not non_english:
            print("No non-English chunks to translate")
            return

        # Limit chunks if specified
        if max_chunks and max_chunks < len(non_english):
            print(f"Limiting translation to {max_chunks} chunks (of {len(non_english)} non-English)")
            non_english = non_english[:max_chunks]

        print(f"Translating {len(non_english)} non-English chunks...")
        print("This may take a while due to API rate limits (4 RPM for free tier).")

        translated_count = 0
        for i, chunk in enumerate(tqdm(non_english, desc="Translating")):
            lang = chunk.get("language", "en")
            original_text = chunk["text"]

            # Translate
            translated = self.translator.translate(original_text, lang)

            if translated:
                # Store original and set text to translation for embedding
                chunk["original_text"] = original_text
                chunk["text"] = translated
                chunk["is_translation"] = True
                chunk["original_language"] = lang
                translated_count += 1
                # Keep language as original for display purposes

        print(f"Translation complete: {translated_count}/{len(non_english)} chunks translated")

    def embed_chunks(self):
        """Generate embeddings for all chunks."""
        # Embed the 'text' field (which is translated for non-English)
        texts = [c["text"] for c in self.chunks]
        print(f"Embedding {len(texts)} chunks...")

        self.embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        self.embeddings = np.array(self.embeddings, dtype="float32")

    def build_index(self):
        """Build FAISS index."""
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        print(f"FAISS index built with {self.index.ntotal} vectors")

    def save(self):
        """Save index and metadata."""
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(self.index_dir / "faiss.index"))

        # Save chunks with full metadata
        with open(self.index_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        # Save config
        config = {
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dim": int(self.embeddings.shape[1]),
            "total_chunks": len(self.chunks),
        }
        with open(self.index_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"Index and metadata saved to {self.index_dir}")

    def run(self, skip_translation: bool = False, max_translate: int = None):
        """Run full ingestion pipeline.

        Args:
            skip_translation: Skip translation entirely
            max_translate: Max chunks to translate (None = all, 15 recommended for free tier)
        """
        self.load_files()

        if not skip_translation:
            self.translate_chunks(max_chunks=max_translate)

        self.embed_chunks()
        self.build_index()
        self.save()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents for RAG v2")
    parser.add_argument(
        "--skip-translation",
        action="store_true",
        help="Skip translation step (faster, but non-English won't embed well)",
    )
    parser.add_argument(
        "--max-translate",
        type=int,
        default=15,
        help="Max chunks to translate (default: 15, set to 0 for unlimited). Free tier allows ~20/day.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing source documents",
    )
    parser.add_argument(
        "--index-dir",
        default="data/index_v2",
        help="Directory to save index",
    )

    args = parser.parse_args()

    ingestor = MultilingualIngestor(
        data_dir=args.data_dir,
        index_dir=args.index_dir,
    )

    max_trans = args.max_translate if args.max_translate > 0 else None
    ingestor.run(skip_translation=args.skip_translation, max_translate=max_trans)
