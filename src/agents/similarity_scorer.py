"""SimilarityScorer — filters semantically redundant QA pairs.

Paper 1, Section 3.3: 'A SimilarityScorer Agent filters out newly generated
pairs that are semantically redundant with existing data, thereby maintaining
dataset diversity.'

Uses sentence-transformers embeddings, no LLM needed.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.data_models import QAPair
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SimilarityScorer:
    """Embedding-based deduplication for QA pairs."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.85):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self._existing_embeddings: list[np.ndarray] = []
        self._existing_ids: list[str] = []

    def is_duplicate(self, qa: QAPair) -> bool:
        """Check if qa.question is semantically redundant with existing questions."""
        if not self._existing_embeddings:
            return False

        new_emb = self.model.encode([qa.question], convert_to_numpy=True)[0]

        for existing_emb in self._existing_embeddings:
            sim = self._cosine_similarity(new_emb, existing_emb)
            if sim >= self.threshold:
                return True
        return False

    def add_to_index(self, qa: QAPair) -> None:
        """Add a QA pair to the existing set for future dedup checks."""
        emb = self.model.encode([qa.question], convert_to_numpy=True)[0]
        self._existing_embeddings.append(emb)
        self._existing_ids.append(qa.id)

    def deduplicate_batch(self, qas: list[QAPair]) -> list[QAPair]:
        """Remove duplicates within a batch and against existing index."""
        unique = []
        for qa in qas:
            if not self.is_duplicate(qa):
                self.add_to_index(qa)
                unique.append(qa)
            else:
                logger.info(f"Filtered duplicate QA: {qa.id}")
        return unique

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)
