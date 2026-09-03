# REFACTOR: Optional semantic relevance layered on the deterministic lexical
# gates. Lexical bigram coverage is brittle for paraphrases and cross-language
# pairs (an English query against a Mandarin snippet), so cosine similarity acts
# as a rescue-and-rerank signal. It never overrides a hard evidence gate.
from __future__ import annotations

import logging
import math
import threading
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_SEMANTIC_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_MIN_SIMILARITY = 0.35
# Bounded so a strong semantic match can rescue a result that lexical scoring
# put just under the usability gate, without outweighing hard constraints.
SEMANTIC_MAX_BONUS = 25
SEMANTIC_TEXT_MAX_CHARS = 1000


@runtime_checkable
class SemanticScorer(Protocol):
    """Minimal similarity surface used by discovery and the answer node."""

    def similarities(self, query: str, texts: Sequence[str]) -> list[float]:
        """Return cosine similarity in ``[-1, 1]`` for each text against ``query``."""


class EmbeddingSemanticScorer:
    """Cosine similarity over a lazily loaded local embedding model."""

    def __init__(self, model_name: str = DEFAULT_SEMANTIC_MODEL) -> None:
        self.model_name = model_name
        self._embeddings: Any | None = None
        self._lock = threading.Lock()

    def _model(self) -> Any | None:
        if self._embeddings is not None:
            return self._embeddings
        with self._lock:
            if self._embeddings is None:
                from ..rag.hf_embeddings import build_huggingface_embeddings

                self._embeddings = build_huggingface_embeddings(self.model_name)
        return self._embeddings

    def similarities(self, query: str, texts: Sequence[str]) -> list[float]:
        candidates = [str(text or "")[:SEMANTIC_TEXT_MAX_CHARS] for text in texts]
        if not query.strip() or not any(text.strip() for text in candidates):
            return [0.0] * len(candidates)

        try:
            model = self._model()
            if model is None:
                return [0.0] * len(candidates)
            query_vector = model.embed_query(query)
            document_vectors = model.embed_documents(candidates)
        except Exception as exc:
            logger.warning("Semantic relevance unavailable; using lexical scores only: %s", exc)
            return [0.0] * len(candidates)

        return [cosine_similarity(query_vector, vector) for vector in document_vectors]


_default_scorer: EmbeddingSemanticScorer | None = None
_default_scorer_lock = threading.Lock()


def default_semantic_scorer(model_name: str = DEFAULT_SEMANTIC_MODEL) -> EmbeddingSemanticScorer:
    """Return a process-wide scorer so the model loads at most once."""

    global _default_scorer
    with _default_scorer_lock:
        if _default_scorer is None or _default_scorer.model_name != model_name:
            _default_scorer = EmbeddingSemanticScorer(model_name)
        return _default_scorer


def build_semantic_scorer(settings: Any) -> SemanticScorer | None:
    """Return a scorer when semantic filtering is enabled, else ``None``."""

    if not getattr(settings, "web_search_semantic_filter_enabled", False):
        return None
    return default_semantic_scorer(
        str(getattr(settings, "web_search_semantic_model", DEFAULT_SEMANTIC_MODEL))
        or DEFAULT_SEMANTIC_MODEL
    )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity, or ``0.0`` for empty or zero vectors."""

    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (left_norm * right_norm)


def semantic_bonus(similarity: float, min_similarity: float = DEFAULT_MIN_SIMILARITY) -> int:
    """Convert a similarity into a bounded, non-negative score bonus.

    Similarities at or below the threshold contribute nothing, so the semantic
    layer can only add recall. It never penalizes a lexically strong result.
    """

    threshold = max(0.0, min(1.0, float(min_similarity)))
    if similarity <= threshold:
        return 0
    span = 1.0 - threshold
    if span <= 0:
        return SEMANTIC_MAX_BONUS
    return round(SEMANTIC_MAX_BONUS * min(1.0, (similarity - threshold) / span))


__all__ = [
    "DEFAULT_MIN_SIMILARITY",
    "DEFAULT_SEMANTIC_MODEL",
    "SEMANTIC_MAX_BONUS",
    "EmbeddingSemanticScorer",
    "SemanticScorer",
    "build_semantic_scorer",
    "cosine_similarity",
    "default_semantic_scorer",
    "semantic_bonus",
]
