import numpy as np
import pytest

from hybridrag.fusion import convex, normalize, rrf
from hybridrag.ranking import Ranked, top_k
from hybridrag.rerank import apply_rerank, interpolate


def ranked(indices, scores=None):
    indices = np.asarray(indices, dtype=np.int64)
    if scores is None:
        scores = -np.arange(len(indices), dtype=float)
    return Ranked(indices, np.asarray(scores, dtype=float))


def test_rrf_hand_computed():
    fused = rrf([ranked([0, 1, 2]), ranked([2, 0, 3])], k=60)
    expected = {0: 1 / 61 + 1 / 62, 1: 1 / 62, 2: 1 / 63 + 1 / 61, 3: 1 / 63}
    assert fused.indices.tolist() == [0, 2, 1, 3]
    np.testing.assert_allclose(fused.scores, [expected[d] for d in fused.indices])


def test_rrf_single_list_keeps_order_and_k_controls_flatness():
    one = ranked([5, 3, 9])
    assert rrf([one]).indices.tolist() == [5, 3, 9]
    # rank 1 in one list vs rank 2 in both: small k favours the single top hit
    lists = [ranked([0, 1]), ranked([2, 1])]
    assert rrf(lists, k=0).indices[0] == 0
    assert rrf(lists, k=60).indices[0] == 1


def test_normalize_minmax_and_zscore():
    s = np.array([2.0, 4.0, 10.0])
    np.testing.assert_allclose(normalize(s, "minmax"), [0.0, 0.25, 1.0])
    z = normalize(s, "zscore")
    assert z.mean() == pytest.approx(0) and z.std() == pytest.approx(1)
    assert normalize(np.array([3.0, 3.0]), "minmax").tolist() == [0.0, 0.0]
    assert normalize(np.array([3.0, 3.0]), "zscore").tolist() == [0.0, 0.0]
    with pytest.raises(ValueError):
        normalize(s, "softmax")


def test_convex_combination_hand_computed():
    lex = ranked([0, 1, 2], [10.0, 6.0, 2.0])  # minmax -> 1, 0.5, 0
    den = ranked([2, 3], [0.9, 0.5])  # minmax -> 1, 0
    fused = convex(lex, den, weight=0.4)
    # doc 3 is missing from the lexical list -> lexical floor 0
    expected = {0: 0.6 * 1.0 + 0.4 * 0.0, 1: 0.6 * 0.5, 2: 0.6 * 0.0 + 0.4 * 1.0, 3: 0.0}
    assert fused.indices.tolist() == [0, 2, 1, 3]
    np.testing.assert_allclose(fused.scores, [expected[d] for d in fused.indices])


def test_convex_endpoints_reproduce_each_retriever():
    rng = np.random.default_rng(0)
    lex = top_k(rng.random(50), 20)
    den = top_k(rng.random(50), 20)
    # the last document of a list sits at the floor, tied with documents the list missed,
    # so the endpoints reproduce each retriever down to its second-to-last document
    assert convex(lex, den, 0.0).indices[:19].tolist() == lex.indices[:19].tolist()
    assert convex(lex, den, 1.0, "zscore").indices[:19].tolist() == den.indices[:19].tolist()
    with pytest.raises(ValueError):
        convex(lex, den, 1.5)


def test_top_k_sorts_breaks_ties_by_row_and_drops_nonpositive():
    scores = np.array([0.5, 2.0, 0.5, 0.0, 3.0])
    r = top_k(scores, 4)
    assert r.indices.tolist() == [4, 1, 0, 2]
    assert top_k(scores, 10, positive_only=True).indices.tolist() == [4, 1, 0, 2]


def test_apply_rerank_reorders_head_only():
    first = ranked([7, 3, 9, 1, 4])
    out = apply_rerank(first, np.array([0.1, 0.9, 0.5]))
    assert out.indices.tolist() == [3, 9, 7, 1, 4]  # tail (1, 4) untouched
    assert np.all(np.diff(out.scores) < 0)


def test_interpolate_endpoints():
    ce = np.array([3.0, 1.0, 2.0])
    first = np.array([0.1, 0.9, 0.5])
    assert np.argsort(-interpolate(ce, first, 1.0)).tolist() == [0, 2, 1]
    assert np.argsort(-interpolate(ce, first, 0.0)).tolist() == [1, 2, 0]
    with pytest.raises(ValueError):
        interpolate(ce, first, -0.1)
