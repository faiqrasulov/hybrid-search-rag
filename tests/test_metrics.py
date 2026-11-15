import math

import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from hybridrag.metrics import (
    bootstrap_ci,
    bootstrap_indices,
    dcg,
    holm,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap,
    per_query,
    recall_at_k,
    recall_curve,
)


def test_ndcg_hand_computed_binary():
    # relevant d1 at rank 2 and d2 at rank 3; ideal puts them at ranks 1 and 2
    ranked = ["d3", "d1", "d2"]
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(ranked, {"d1": 1, "d2": 1}, k=10) == pytest.approx(expected)
    assert expected == pytest.approx(0.693426, abs=1e-6)


def test_ndcg_hand_computed_graded():
    # gains 1 then 2 versus ideal 2 then 1 (linear gains, as in trec_eval)
    value = ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, k=10)
    assert value == pytest.approx((1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3)))
    assert value == pytest.approx(0.859719, abs=1e-6)


def test_ndcg_cutoff_and_edge_cases():
    rels = {f"r{i}": 1 for i in range(15)}
    perfect = [f"r{i}" for i in range(15)]
    assert ndcg_at_k(perfect, rels, k=10) == pytest.approx(1.0)  # ideal truncated at k
    assert ndcg_at_k(["x"] * 10 + perfect, rels, k=10) == 0.0  # relevant only after k
    assert ndcg_at_k(["a"], {}, k=10) == 0.0
    assert ndcg_at_k(["a"], {"a": 0}, k=10) == 0.0  # grade 0 is not relevant


def test_ndcg_matches_sklearn_reference():
    rng = np.random.default_rng(0)
    for _ in range(25):
        n = 40
        grades = rng.integers(0, 3, size=n) * (rng.random(n) < 0.3)
        if grades.sum() == 0:
            continue
        scores = rng.permutation(n).astype(float)  # distinct scores: no ties
        ranked = [f"d{i}" for i in np.argsort(-scores)]
        rels = {f"d{i}": int(g) for i, g in enumerate(grades) if g > 0}
        ref = ndcg_score(grades[None, :], scores[None, :], k=10)
        assert ndcg_at_k(ranked, rels, k=10) == pytest.approx(ref, abs=1e-12)


def test_recall_and_mrr_known_answers():
    ranked = ["a", "b", "c", "d", "e"]
    rels = {"b": 1, "e": 1, "z": 1}
    assert recall_at_k(ranked, rels, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(ranked, rels, k=5) == pytest.approx(2 / 3)
    assert mrr_at_k(ranked, rels, k=10) == pytest.approx(0.5)
    assert mrr_at_k(ranked, {"e": 1}, k=4) == 0.0
    assert dcg([1, 0, 1]) == pytest.approx(1 + 0.5)


def test_per_query_scores_missing_queries_as_zero():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    out = per_query({"q1": ["a", "c"]}, qrels, qids=["q1", "q2"])
    assert out["ndcg@10"].tolist() == [1.0, 0.0]
    assert out["recall@100"].tolist() == [1.0, 0.0]
    assert recall_curve({"q1": ["c", "a"]}, {"q1": {"a": 1}}, [1, 2]).tolist() == [0.0, 1.0]


def test_bootstrap_ci_and_paired_test():
    rng = np.random.default_rng(1)
    idx = bootstrap_indices(200, n_boot=4000, seed=0)
    a = rng.normal(0.5, 0.2, 200)
    mean, lo, hi = bootstrap_ci(a, idx)
    assert lo < mean < hi
    se = a.std(ddof=1) / np.sqrt(200)
    assert hi - lo == pytest.approx(2 * 1.96 * se, rel=0.15)  # matches the normal interval

    same = paired_bootstrap(a, a, idx)
    assert same["diff"] == 0.0 and same["p_value"] == 1.0

    better = paired_bootstrap(a + 0.1 + rng.normal(0, 0.05, 200), a, idx)
    assert better["ci_low"] > 0 and better["p_value"] < 0.001

    noise = paired_bootstrap(a + rng.normal(0, 0.05, 200), a, idx)
    assert noise["ci_low"] < 0 < noise["ci_high"] and noise["p_value"] > 0.05


def test_holm_known_values():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(holm([0.5, 0.9]), [1.0, 1.0])
