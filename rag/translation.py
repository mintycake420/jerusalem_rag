"""Translation pipeline for multilingual corpus.

Handles batch translation during ingestion using Gemini API.
Designed to work within Gemini 3 Flash free tier limits (5 RPM, 20 RPD).
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Optional

from langdetect import detect, LangDetectException
from google import genai
from google.genai import types

from .models import LANGUAGE_NAMES


class TranslationCache:
    """Simple file-based cache for translations."""

    def __init__(self, cache_dir: str = "data/translations"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()

    def _load_index(self) -> dict:
        """Load cache index from disk."""
        if self.index_file.exists():
            return json.loads(self.index_file.read_text(encoding="utf-8"))
        return {}

    def _save_index(self):
        """Save cache index to disk."""
        self.index_file.write_text(
            json.dumps(self.index, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _hash_text(self, text: str) -> str:
        """Create hash for cache key."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get(self, text: str, source_lang: str) -> Optional[str]:
        """Get cached translation if available."""
        key = f"{source_lang}_{self._hash_text(text)}"
        if key in self.index:
            cache_file = self.cache_dir / f"{key}.txt"
            if cache_file.exists():
                return cache_file.read_text(encoding="utf-8")
        return None

    def put(self, text: str, source_lang: str, translation: str):
        """Cache a translation."""
        key = f"{source_lang}_{self._hash_text(text)}"
        cache_file = self.cache_dir / f"{key}.txt"
        cache_file.write_text(translation, encoding="utf-8")
        self.index[key] = {
            "source_lang": source_lang,
            "original_len": len(text),
            "translated_len": len(translation),
        }
        self._save_index()


class Translator:
    """Translate text using Gemini API with rate limiting."""

    def __init__(
        self,
        cache_dir: str = "data/translations",
        requests_per_minute: int = 4,  # Stay under 5 RPM limit for Gemini 3 Flash
    ):
        self.cache = TranslationCache(cache_dir)
        self.min_delay = 60.0 / requests_per_minute
        self.last_request_time = 0
        self._client = None

    @property
    def client(self):
        """Lazy-load Gemini client."""
        if self._client is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def detect_language(self, text: str) -> str:
        """Detect language of text. Returns ISO 639-1 code."""
        # Check for explicit language markers in header
        text_lower = text[:500].lower()
        if "language: la" in text_lower:
            return "la"
        if "language: ar" in text_lower:
            return "ar"
        if "language: el" in text_lower or "language: greek" in text_lower:
            return "el"

        # Use langdetect for automatic detection
        try:
            # Sample middle of text to avoid headers
            sample_start = min(500, len(text) // 4)
            sample = text[sample_start : sample_start + 2000]
            detected = detect(sample)
            # Map some common detections
            if detected == "hr" or detected == "sr":  # Often misdetects Latin
                # Check for Latin patterns
                if any(w in text.lower() for w in ["rex", "deus", "anno", "ecclesia"]):
                    return "la"
            return detected
        except LangDetectException:
            return "en"  # Default to English if detection fails

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self.last_request_time = time.time()

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str = "en",
        use_cache: bool = True,
    ) -> str:
        """Translate text using Gemini API.

        Args:
            text: Text to translate
            source_lang: ISO 639-1 source language code
            target_lang: Target language (default: English)
            use_cache: Whether to use translation cache

        Returns:
            Translated text
        """
        # Check cache first
        if use_cache:
            cached = self.cache.get(text, source_lang)
            if cached:
                return cached

        # Rate limit
        self._rate_limit()

        # Build translation prompt
        lang_name = LANGUAGE_NAMES.get(source_lang, source_lang.upper())
        prompt = f"""You are a scholarly translator specializing in medieval texts.
Translate the following {lang_name} text to English.

IMPORTANT:
- Preserve proper nouns (names of people, places) in their common English forms
- Keep historical terms with brief clarification if needed
- Maintain the tone and style of medieval chronicles
- If text contains OCR errors, do your best to interpret the intended meaning

TEXT TO TRANSLATE:
{text}

ENGLISH TRANSLATION:"""

        max_retries = 5
        for attempt in range(max_retries):
            try:
                config = types.GenerateContentConfig(
                    temperature=0.1,  # Low temperature for accurate translation
                    max_output_tokens=4096,
                    top_p=0.95,
                )

                response = self.client.models.generate_content(
                    model="gemini-3-flash-preview",  # Free tier: 5 RPM, 250K TPM, 20 RPD
                    contents=prompt,
                    config=config,
                )

                translation = response.text or ""

                # Cache the result
                if use_cache and translation:
                    self.cache.put(text, source_lang, translation)

                return translation

            except Exception as e:
                error_str = str(e)
                # Retry on 503 (overloaded) or 429 (rate limit)
                if "503" in error_str or "429" in error_str or "overloaded" in error_str.lower():
                    wait_time = (2 ** attempt) * 30 + 10  # 40s, 70s, 130s, 250s, 490s
                    print(f"  API overloaded/rate-limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Translation error: {e}")
                    return ""

        print(f"  Failed after {max_retries} retries")
        return ""

    def translate_chunks(
        self,
        chunks: list[dict],
        source_lang: str,
        progress_callback=None,
    ) -> list[dict]:
        """Translate a list of chunks, adding translations to each.

        Args:
            chunks: List of chunk dicts with 'text' field
            source_lang: Source language code
            progress_callback: Optional callback(current, total)

        Returns:
            Chunks with 'translated_text' field added
        """
        total = len(chunks)
        translated = []

        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            translation = self.translate(text, source_lang)

            new_chunk = chunk.copy()
            new_chunk["translated_text"] = translation
            new_chunk["original_language"] = source_lang
            new_chunk["is_translation"] = False
            translated.append(new_chunk)

            if progress_callback:
                progress_callback(i + 1, total)

            if (i + 1) % 10 == 0:
                print(f"  Translated {i + 1}/{total} chunks")

        return translated


def is_english(text: str) -> bool:
    """Check if text is primarily English."""
    try:
        sample_start = min(500, len(text) // 4)
        sample = text[sample_start : sample_start + 2000]
        return detect(sample) == "en"
    except LangDetectException:
        return True  # Assume English if detection fails
