import math
from collections import Counter

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from hybridrag.lexical import BM25, TermCounts, TfidfCosine
from hybridrag.text import Tokenizer

PLAIN = Tokenizer(stopwords=None, stemmer=None)


def test_tokenizer_lowercases_drops_stopwords_and_stems():
    tok = Tokenizer()
    assert tok("The cell's RUNNING studies of IL-6 and TNF-α") == [
        "cell",
        "run",
        "studi",
        "il",
        "6",
        "tnf",
        "α",
    ]
    assert Tokenizer(stemmer=None)("The running studies") == ["running", "studies"]
    assert PLAIN("The cat") == ["the", "cat"]


def test_bm25_hand_computed_tiny_corpus():
    # N = 3, df(cat) = 2, |d| = [2, 3, 2], avgdl = 7/3, k1 = 1.2, b = 0.75
    bm25 = BM25.fit(["cat sat", "cat cat dog", "dog bird"], tokenizer=PLAIN)
    scores = bm25.score(["cat"])[0]
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    assert idf == pytest.approx(0.470004, abs=1e-6)
    assert scores[0] == pytest.approx(0.499176, abs=1e-6)  # tf=1, |d|=2
    assert scores[1] == pytest.approx(0.598187, abs=1e-6)  # tf=2, |d|=3
    assert scores[2] == 0.0
    ranked = bm25.rank(["cat"], k=10)[0]
    assert ranked.indices.tolist() == [1, 0]  # the doc without "cat" is not returned


def _reference_bm25(docs, query, k1, b):
    """Textbook BM25 with plain Python loops (independent of the sparse implementation)."""
    toks = [PLAIN(d) for d in docs]
    n = len(docs)
    avgdl = sum(len(t) for t in toks) / n
    df = Counter(term for t in toks for term in set(t))
    out = []
    for t in toks:
        tf = Counter(t)
        s = 0.0
        for term in PLAIN(query):
            if term not in df:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            f = tf[term]
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(t) / avgdl))
        out.append(s)
    return np.array(out)


@pytest.mark.parametrize("k1,b", [(1.2, 0.75), (0.9, 0.4), (2.0, 1.0), (1.5, 0.0)])
def test_bm25_matches_reference_formula(k1, b):
    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(30)]
    docs = [" ".join(rng.choice(words, size=rng.integers(3, 25))) for _ in range(40)]
    queries = [" ".join(rng.choice(words, size=4)) + " unseen" for _ in range(6)]
    bm25 = BM25.fit(docs, tokenizer=PLAIN, k1=k1, b=b)
    got = bm25.score(queries)
    for qi, q in enumerate(queries):
        np.testing.assert_allclose(got[qi], _reference_bm25(docs, q, k1, b), rtol=1e-10)
    # re-weighting the same index equals fitting from scratch
    again = BM25.fit(docs, tokenizer=PLAIN).with_params(k1, b).score(queries)
    np.testing.assert_allclose(again, got, rtol=1e-12)


def test_tfidf_matches_sklearn():
    rng = np.random.default_rng(1)
    words = [f"w{i}" for i in range(25)]
    docs = [" ".join(rng.choice(words, size=rng.integers(2, 20))) for _ in range(30)]
    queries = [" ".join(rng.choice(words, size=3)) for _ in range(5)]
    ours = TfidfCosine.fit(docs, tokenizer=PLAIN).score(queries)
    vec = TfidfVectorizer(
        tokenizer=PLAIN, lowercase=False, token_pattern=None, sublinear_tf=True, norm="l2"
    )
    d = vec.fit_transform(docs)
    ref = (vec.transform(queries) @ d.T).toarray()
    np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_term_counts_drop_out_of_vocabulary_query_terms():
    counts = TermCounts(["a b b", "c"], PLAIN)
    q = counts.queries(["b b zzz", "zzz"]).toarray()
    assert q[0, counts.vocab["b"]] == 2 and q.sum() == 2
    assert counts.matrix.toarray()[0, counts.vocab["b"]] == 2
