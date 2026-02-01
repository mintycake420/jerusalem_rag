"""Retrieval module for Jerusalem RAG v2.

Supports:
- Language-aware retrieval and formatting
- Optional language filtering
- Display of original text alongside translations
"""

import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .models import get_language_name, get_language_flag


class Retriever:
    """Retrieve relevant chunks from the multilingual index."""

    def __init__(
        self,
        index_dir: str = "data/index_v2",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.index_dir = Path(index_dir)
        self.model = SentenceTransformer(model_name)

        # Load index
        index_path = self.index_dir / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found at {index_path}")
        self.index = faiss.read_index(str(index_path))

        # Load chunks
        chunks_path = self.index_dir / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks not found at {chunks_path}")
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def retrieve(
        self,
        question: str,
        top_k: int = 4,
        languages: list[str] = None,
    ) -> list[tuple[float, dict]]:
        """Retrieve top-k relevant chunks.

        Args:
            question: Query string
            top_k: Number of results to return
            languages: Optional list of language codes to filter by (e.g., ["en", "la"])

        Returns:
            List of (score, chunk_dict) tuples
        """
        # Encode query
        q_emb = self.model.encode([question], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")

        # Search - get more if filtering
        search_k = top_k * 3 if languages else top_k
        scores, ids = self.index.search(q_emb, search_k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]

            # Apply language filter
            if languages:
                chunk_lang = chunk.get("language", "en")
                # Also include if original_language matches
                orig_lang = chunk.get("original_language")
                if chunk_lang not in languages and orig_lang not in languages:
                    continue

            results.append((float(score), chunk))

            if len(results) >= top_k:
                break

        return results


def format_context(
    results: list[tuple[float, dict]],
    include_original: bool = False,
    max_chunk_len: int = None,
) -> str:
    """Format retrieved chunks into context for LLM.

    Args:
        results: List of (score, chunk) tuples
        include_original: Include original text for translations
        max_chunk_len: Optional max length per chunk

    Returns:
        Formatted context string
    """
    parts = []

    for score, chunk in results:
        chunk_id = chunk.get("chunk_id", "unknown")
        lang = chunk.get("language", "en")
        lang_name = get_language_name(lang)
        is_trans = chunk.get("is_translation", False)
        orig_lang = chunk.get("original_language")

        # Build header
        header = f"[{chunk_id}] (score: {score:.3f})"
        if orig_lang and orig_lang != "en":
            orig_name = get_language_name(orig_lang)
            header += f" [Original: {orig_name}, Translated to English]"
        elif lang != "en":
            header += f" [{lang_name}]"

        # Get text
        text = chunk.get("text", "")
        if max_chunk_len and len(text) > max_chunk_len:
            text = text[:max_chunk_len] + "..."

        part = f"{header}\n{text}"

        # Include original if requested
        if include_original and is_trans:
            original = chunk.get("original_text", "")
            if original:
                if max_chunk_len and len(original) > max_chunk_len:
                    original = original[:max_chunk_len] + "..."
                part += f"\n\n[Original {get_language_name(orig_lang)} text:]\n{original}"

        parts.append(part)

    return "\n\n---\n\n".join(parts)


def format_sources_summary(results: list[tuple[float, dict]]) -> list[dict]:
    """Create summary of sources for display.

    Returns list of dicts with source metadata for UI display.
    """
    sources = []
    for score, chunk in results:
        sources.append({
            "chunk_id": chunk.get("chunk_id", "unknown"),
            "score": score,
            "language": chunk.get("language", "en"),
            "language_name": get_language_name(chunk.get("language", "en")),
            "flag": get_language_flag(chunk.get("language", "en")),
            "is_translation": chunk.get("is_translation", False),
            "original_language": chunk.get("original_language"),
            "author": chunk.get("author"),
            "title": chunk.get("title"),
            "source_repository": chunk.get("source_repository", "unknown"),
            "preview": chunk.get("text", "")[:200] + "...",
        })
    return sources


# Convenience function matching v1 API
def retrieve(question: str, top_k: int = 4, index_dir: str = "data/index_v2"):
    """Simple retrieve function for backward compatibility."""
    retriever = Retriever(index_dir=index_dir)
    return retriever.retrieve(question, top_k=top_k)
