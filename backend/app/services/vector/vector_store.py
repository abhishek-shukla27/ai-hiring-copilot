import faiss
import numpy as np
import os
import pickle
from typing import List, Dict

DIM = 384  # set to whatever your embedding dim is
INDEX_PATH = "data/faiss_index.bin"
META_PATH = "data/faiss_metadata.pkl"

class VectorStore:
    def __init__(self, dim: int = DIM, index_path: str = INDEX_PATH, meta_path: str = META_PATH):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path

        # We'll use inner product on normalized vectors -> equivalent to cosine similarity
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata: List[Dict] = []

        # If files exist, load
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "rb") as f:
                    self.metadata = pickle.load(f)
            except Exception as e:
                # start fresh if load fails
                print("Failed to load index/metadata:", e)
                self.index = faiss.IndexFlatIP(self.dim)
                self.metadata = []

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        # L2 normalize rows
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def add_vector(self, vector: List[float], meta: Dict):
        v = np.array(vector, dtype="float32").reshape(1, -1)
        if v.shape[1] != self.dim:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dim}, got {v.shape[1]}")
        v = self._normalize(v)
        self.index.add(v)
        self.metadata.append(meta)

    def add_bulk(self, vectors: List[List[float]], metas: List[Dict]):
        arr = np.array(vectors, dtype="float32")
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError("Invalid vectors shape")
        arr = self._normalize(arr)
        self.index.add(arr)
        self.metadata.extend(metas)

    def search(self, query_vec: List[float], top_k: int = 5):
        q = np.array(query_vec, dtype="float32").reshape(1, -1)
        q = self._normalize(q)
        scores, ids = self.index.search(q, top_k)
        results = []
        for i, idx in enumerate(ids[0]):
            if idx < len(self.metadata) and idx != -1:
                results.append({
                    "score": float(scores[0][i]),
                    "metadata": self.metadata[idx],
                    "id": int(idx)
                })
        return results

    def save(self):
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def clear(self):
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = []
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.meta_path):
            os.remove(self.meta_path)
