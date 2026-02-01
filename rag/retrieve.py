import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("data/index/faiss.index")
chunks = json.loads(open("data/index/chunks.json", encoding="utf-8").read())

def retrieve(question, top_k=4):
    q = model.encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, top_k)

    results = []
    for rank, cid in enumerate(ids[0]):
        c = chunks[int(cid)]
        results.append((float(scores[0][rank]), c))
    return results

def format_context(results):
    parts = []
    for score, c in results:
        parts.append(f"[{c['chunk_id']}] (score {score:.3f})\n{c['text']}")
    return "\n\n---\n\n".join(parts)
