"""Data models for Jerusalem RAG v2."""

from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import date


@dataclass
class ChunkMeta:
    """Metadata for a text chunk in the RAG corpus."""

    chunk_id: str
    source: str
    text: str

    # Language info
    language: str = "en"  # ISO 639-1: en, la, ar, el, fr
    language_name: str = "English"
    is_translation: bool = False
    original_language: Optional[str] = None
    original_text: Optional[str] = None

    # Source attribution
    author: Optional[str] = None
    title: Optional[str] = None
    date_composed: Optional[str] = None
    source_url: Optional[str] = None
    source_repository: str = "unknown"  # archive, wiki, gallica

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ChunkMeta":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Language code mappings
LANGUAGE_NAMES = {
    "en": "English",
    "la": "Latin",
    "ar": "Arabic",
    "el": "Greek",
    "fr": "French",
    "he": "Hebrew",
    "it": "Italian",
    "de": "German",
}

LANGUAGE_FLAGS = {
    "en": "🇬🇧",
    "la": "🇻🇦",  # Vatican for Latin
    "ar": "🇸🇦",
    "el": "🇬🇷",
    "fr": "🇫🇷",
    "he": "🇮🇱",
    "it": "🇮🇹",
    "de": "🇩🇪",
}


def get_language_name(code: str) -> str:
    """Get human-readable language name from ISO code."""
    return LANGUAGE_NAMES.get(code, code.upper())


def get_language_flag(code: str) -> str:
    """Get flag emoji for language code."""
    return LANGUAGE_FLAGS.get(code, "🌐")
