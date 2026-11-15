"""Second-stage reranking of a first-stage candidate list with a cross-encoder."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .fusion import normalize
from .ranking import Ranked, sort_desc

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class PairScorer(Protocol):
    """Scores (query, document) pairs; higher means more relevant."""

    name: str

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray: ...


class CrossEncoderReranker:
    """``sentence_transformers.CrossEncoder`` reading query and document together.

    ``max_length`` caps query + document tokens (abstracts beyond it are truncated); 256
    keeps CPU latency manageable and covers the first ~180 words of an abstract.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
        max_length: int = 256,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import CrossEncoder  # heavy import, done lazily

        self.name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        if not documents:
            return np.empty(0, dtype=np.float64)
        pairs = [(query, d) for d in documents]
        out = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return np.asarray(out, dtype=np.float64)


def apply_rerank(first_stage: Ranked, scores: np.ndarray) -> Ranked:
    """Reorder the first ``len(scores)`` candidates by ``scores``; keep the tail as it was.

    Tail documents keep their first-stage order below every reranked one, so Recall@k for
    k beyond the rerank depth is unchanged. Returned scores are ranks turned into
    descending numbers (cross-encoder logits and first-stage scores are not comparable).
    """
    depth = len(scores)
    head = sort_desc(first_stage.indices[:depth], np.asarray(scores, dtype=np.float64)).indices
    order = np.concatenate([head, first_stage.indices[depth:]])
    return Ranked(order, -np.arange(order.size, dtype=np.float64))


def interpolate(ce_scores: np.ndarray, first_stage_scores: np.ndarray, weight: float) -> np.ndarray:
    """``weight * z(cross-encoder) + (1 - weight) * z(first stage)`` over the reranked head.

    ``weight = 1`` gives the pure cross-encoder order, ``weight = 0`` keeps the first stage.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    return weight * normalize(ce_scores, "zscore") + (1.0 - weight) * normalize(
        first_stage_scores, "zscore"
    )
