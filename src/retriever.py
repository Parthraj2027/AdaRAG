import pickle
from pathlib import Path
import faiss
import numpy as np
from config import MODELS_DIR
from src.embedder import Embedder
from src.utils import ensure_directory, logger

class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self.dimension = self.embedder.embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []

    def add_documents(self, documents: list[str]) -> None:
        if not documents:
            raise ValueError("Document list is empty.")
        embeddings = self.embedder.encode_batch(documents)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        query_embedding = self.embedder.encode(query)
        query_embedding = np.asarray([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.documents[idx], float(score)))
        return results

    def save(self) -> None:
        ensure_directory(MODELS_DIR)
        faiss.write_index(self.index, str(MODELS_DIR / "faiss.index"))
        with open(MODELS_DIR / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    def load(self) -> None:
        index_file = MODELS_DIR / "faiss.index"
        doc_file = MODELS_DIR / "documents.pkl"
        if not index_file.exists() or not doc_file.exists():
            raise FileNotFoundError("Retriever files missing. Build index first.")
        self.index = faiss.read_index(str(index_file))
        with open(doc_file, "rb") as f:
            self.documents = pickle.load(f)