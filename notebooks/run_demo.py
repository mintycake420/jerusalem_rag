"""Run Jerusalem RAG Demo - Q&A with Gemini"""
import os
import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).parent.parent / ".env")

INDEX_DIR = Path(__file__).parent.parent / "data/index_v2"
LANGUAGE_NAMES = {"en": "English", "la": "Latin", "ar": "Arabic", "el": "Greek", "fr": "French", "hy": "Armenian"}

def get_lang_name(code):
    return LANGUAGE_NAMES.get(code, code.upper())

print("=" * 60)
print("JERUSALEM RAG EXPLORER - DEMO")
print("=" * 60)
print()

# Load resources
print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
print(f"  Index contains {index.ntotal} vectors")

print("Loading chunks...")
with open(INDEX_DIR / "chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"  Loaded {len(chunks)} chunks")

# Count by language
lang_counts = {}
for c in chunks:
    lang = c.get("language", "en")
    lang_counts[lang] = lang_counts.get(lang, 0) + 1

print("\nChunks by language:")
for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
    print(f"  {get_lang_name(lang)}: {count}")

SYSTEM_PROMPT = """You are a scholarly historian specializing in the Crusades (1095-1291 CE).
RULES:
1. Answer ONLY using the provided CONTEXT
2. EVERY claim must cite [ChunkID]
3. If insufficient information, say so
Keep your answer concise."""

def retrieve(question, top_k=4):
    q_emb = model.encode([question], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype="float32")
    scores, ids = index.search(q_emb, top_k)
    results = []
    for i in range(top_k):
        if ids[0][i] >= 0 and ids[0][i] < len(chunks):
            results.append((float(scores[0][i]), chunks[ids[0][i]]))
    return results

def format_context(results):
    parts = []
    for score, chunk in results:
        text = chunk["text"][:700]  # Truncate to reduce tokens
        parts.append(f"[{chunk['chunk_id']}]\n{text}")
    return "\n---\n".join(parts)

def ask_question(question, top_k=4):
    results = retrieve(question, top_k=top_k)
    context = format_context(results)
    prompt = f"{SYSTEM_PROMPT}\n\nQUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER:"

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = types.GenerateContentConfig(temperature=0.3, max_output_tokens=1024)

    resp = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config=config
    )
    return resp.text or "", results

print()
print("=" * 60)
print("QUESTION ANSWERING DEMO")
print("=" * 60)

questions = [
    "What happened at the Battle of Hattin?",
    "Who was Baldwin IV of Jerusalem?",
]

for i, question in enumerate(questions, 1):
    print()
    print(f"Q{i}: {question}")
    print()

    try:
        answer, sources = ask_question(question)

        print("ANSWER:")
        print("-" * 40)
        # Sanitize for Windows console
        safe_answer = answer.encode("ascii", "replace").decode("ascii")
        print(safe_answer)

        print()
        print("SOURCES USED:")
        for score, chunk in sources:
            lang = get_lang_name(chunk.get("language", "en"))
            cid = chunk["chunk_id"]
            print(f"  [{lang}] {cid[:50]} (score: {score:.3f})")

    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg:
            print("Rate limit error - waiting before next question...")
            import time
            time.sleep(15)
        else:
            print(f"Error: {err_msg[:200]}")

print()
print("=" * 60)
print("DEMO COMPLETE!")
print("=" * 60)
