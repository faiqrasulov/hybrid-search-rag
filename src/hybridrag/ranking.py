"""A ranked list of corpus rows and the top-k selection every retriever shares."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Ranked:
    """Results for one query: corpus row indices, best first, and their scores."""

    indices: np.ndarray
    scores: np.ndarray

    def __len__(self) -> int:
        return len(self.indices)

    def head(self, k: int) -> Ranked:
        return Ranked(self.indices[:k], self.scores[:k])


def sort_desc(indices: np.ndarray, scores: np.ndarray) -> Ranked:
    """Sort by score descending; ties broken by corpus row (deterministic)."""
    order = np.lexsort((indices, -scores))
    return Ranked(indices[order], scores[order])


def top_k(scores: np.ndarray, k: int, positive_only: bool = False) -> Ranked:
    """Top-k of one score vector over the whole corpus.

    ``positive_only`` drops documents scoring <= 0, as an inverted index would for
    documents that share no term with the query.
    """
    k = min(k, scores.shape[0])
    cand = np.argpartition(-scores, k - 1)[:k] if k < scores.shape[0] else np.arange(k)
    # include every document tied with the k-th score so tie-breaking is by row id
    kth = scores[cand].min()
    cand = np.flatnonzero(scores >= kth)
    ranked = sort_desc(cand, scores[cand]).head(k)
    if positive_only:
        keep = ranked.scores > 0
        ranked = Ranked(ranked.indices[keep], ranked.scores[keep])
    return ranked


def top_k_rows(scores: np.ndarray, k: int, positive_only: bool = False) -> list[Ranked]:
    """Row-wise :func:`top_k` for a (queries x documents) score matrix."""
    return [top_k(row, k, positive_only) for row in np.asarray(scores)]
