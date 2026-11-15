"""Combine ranked lists from several retrievers: Reciprocal Rank Fusion and convex combination."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .ranking import Ranked, sort_desc


def rrf(rankings: Sequence[Ranked], k: int = 60) -> Ranked:
    """Reciprocal Rank Fusion (Cormack et al. 2009): ``score(d) = sum_r 1 / (k + rank_r(d))``.

    Ranks start at 1; a document missing from a list gets nothing from it. Uses only ranks,
    so it needs no score calibration and no tuning beyond ``k``.
    """
    fused: dict[int, float] = {}
    for ranked in rankings:
        for rank, doc in enumerate(ranked.indices.tolist(), start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    docs = np.fromiter(fused.keys(), dtype=np.int64, count=len(fused))
    scores = np.fromiter(fused.values(), dtype=np.float64, count=len(fused))
    return sort_desc(docs, scores)


def normalize(scores: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Per-query score normalisation: ``minmax`` to [0, 1] or ``zscore`` (mean 0, sd 1).

    A constant list maps to all zeros under both methods.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return s
    if method == "minmax":
        span = s.max() - s.min()
        return (s - s.min()) / span if span > 0 else np.zeros_like(s)
    if method == "zscore":
        sd = s.std()
        return (s - s.mean()) / sd if sd > 0 else np.zeros_like(s)
    raise ValueError(f"unknown normalisation {method!r} (use 'minmax' or 'zscore')")


def convex(lexical: Ranked, dense: Ranked, weight: float, method: str = "minmax") -> Ranked:
    """Convex combination ``weight * dense + (1 - weight) * lexical`` of normalised scores.

    Each list is normalised over its own top-k candidates, as in a production system that
    only sees each retriever's top k. A document missing from one list gets that list's
    lowest normalised score (it ranked below everything that list returned).
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    parts = []
    for ranked in (lexical, dense):
        norm = normalize(ranked.scores, method)
        floor = norm.min() if norm.size else 0.0
        parts.append((dict(zip(ranked.indices.tolist(), norm.tolist(), strict=True)), floor))
    (lex, lex_floor), (den, den_floor) = parts
    docs = np.array(sorted(set(lex) | set(den)), dtype=np.int64)
    scores = np.array(
        [
            weight * den.get(d, den_floor) + (1.0 - weight) * lex.get(d, lex_floor)
            for d in docs.tolist()
        ],
        dtype=np.float64,
    )
    return sort_desc(docs, scores)
