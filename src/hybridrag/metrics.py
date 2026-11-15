"""Retrieval metrics (nDCG@k, Recall@k, MRR@k) and paired bootstrap statistics, from scratch.

Conventions follow trec_eval / pytrec_eval as used by BEIR: linear gains (gain = relevance
grade), discount ``1 / log2(rank + 1)``, the ideal DCG built from *all* judged relevant
documents of the query, and Recall@k = relevant retrieved in the top k / all relevant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

Run = Mapping[str, Sequence[str]]  # query id -> ranked doc ids, best first
Qrels = Mapping[str, Mapping[str, int]]  # query id -> {doc id: relevance grade}


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain of gains listed in rank order."""
    g = np.asarray(gains, dtype=np.float64)
    return float(np.sum(g / np.log2(np.arange(2, g.size + 2))))


def ndcg_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int = 10) -> float:
    """nDCG@k for one query."""
    ideal = dcg(sorted((r for r in rels.values() if r > 0), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return dcg([max(rels.get(d, 0), 0) for d in ranked[:k]]) / ideal


def recall_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int = 100) -> float:
    """Share of the relevant documents that appear in the top k."""
    relevant = {d for d, r in rels.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def mrr_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int = 10) -> float:
    """Reciprocal rank of the first relevant document within the top k (0 if none)."""
    for i, d in enumerate(ranked[:k], start=1):
        if rels.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


METRICS = {
    "ndcg@10": lambda r, q: ndcg_at_k(r, q, 10),
    "recall@10": lambda r, q: recall_at_k(r, q, 10),
    "recall@100": lambda r, q: recall_at_k(r, q, 100),
    "mrr@10": lambda r, q: mrr_at_k(r, q, 10),
}


def per_query(run: Run, qrels: Qrels, qids: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    """Every metric in :data:`METRICS` for each query, as arrays aligned with ``qids``.

    Queries missing from the run score 0, as in trec_eval with ``-c``.
    """
    qids = list(qrels) if qids is None else list(qids)
    return {
        name: np.array([fn(list(run.get(q, ())), qrels[q]) for q in qids])
        for name, fn in METRICS.items()
    }


def recall_curve(run: Run, qrels: Qrels, ks: Sequence[int]) -> np.ndarray:
    """Mean Recall@k for each k in ``ks``."""
    return np.array(
        [np.mean([recall_at_k(list(run.get(q, ())), qrels[q], k) for q in qrels]) for k in ks]
    )


def bootstrap_indices(n: int, n_boot: int = 10_000, seed: int = 0) -> np.ndarray:
    """(n_boot x n) query resamples; reuse them across systems so comparisons are paired."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_ci(x: np.ndarray, idx: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """Mean of per-query scores and its percentile bootstrap CI."""
    means = np.asarray(x, dtype=np.float64)[idx].mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(means, [a, 1 - a])
    return float(np.mean(x)), float(lo), float(hi)


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, idx: np.ndarray, level: float = 0.95
) -> dict[str, float]:
    """Compare system ``a`` with ``b`` on the same queries.

    Returns the mean difference ``a - b``, its percentile bootstrap CI and a two-sided
    p-value from the bootstrap distribution shifted to the null of no difference
    (Efron & Tibshirani 1993, sec. 16.4; the "paired bootstrap test" of Sakai 2006).
    """
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    observed = float(d.mean())
    boot = d[idx].mean(axis=1)
    alpha = (1 - level) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    p = float(np.mean(np.abs(boot - observed) >= abs(observed) - 1e-12))
    return {"diff": observed, "ci_low": float(lo), "ci_high": float(hi), "p_value": p}


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted
