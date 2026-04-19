"""Sentence embeddings for deduplication and similarity scoring."""

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper around sentence-transformers for embedding generation."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into dense vectors."""
        return self.model.encode(texts, convert_to_numpy=True)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        a_flat = a.flatten()
        b_flat = b.flatten()
        dot = np.dot(a_flat, b_flat)
        norm = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
        if norm == 0:
            return 0.0
        return float(dot / norm)
