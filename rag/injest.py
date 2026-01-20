from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class Ingestor:
    def __init__(
        self,
        data_dir="data/raw",
        index_dir="data/index",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=2000,
        overlap=300,
    ):
        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.model = SentenceTransformer(model_name)

        self.chunks = []
        self.embeddings = None
        self.index = None

    def chunk_text(self, text: str):
        chunks = []
        i = 0
        while i < len(text):
            part = text[i : i + self.chunk_size].strip()
            if part:
                chunks.append(part)
            i += self.chunk_size - self.overlap
        return chunks

    def load_files(self):
        files = list(self.data_dir.rglob("*.txt"))
        print(f"Found {len(files)} text files")

        for f in files:
            text = f.read_text(encoding="utf-8", errors="ignore")
            prefix = f.stem.replace(" ", "_")

            for i, chunk in enumerate(self.chunk_text(text)):
                self.chunks.append(
                    {
                        "chunk_id": f"{prefix}_chunk_{i:03d}",
                        "source": str(f),
                        "text": chunk,
                    }
                )

        print(f"Created {len(self.chunks)} chunks")

    def embed_chunks(self):
        texts = [c["text"] for c in self.chunks]
        print("Embedding chunks...")
        emb = self.model.encode(texts, normalize_embeddings=True)
        self.embeddings = np.array(emb, dtype="float32")

    def build_index(self):
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        print(f"FAISS index built with {self.index.ntotal} vectors")

    def save(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(self.index_dir / "faiss.index"))

        with open(self.index_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        print("Index and metadata saved")

    def run(self):
        self.load_files()
        self.embed_chunks()
        self.build_index()
        self.save()


if __name__ == "__main__":
    ingestor = Ingestor()
    ingestor.run()
