"""Lexical retrievers from scratch on scipy sparse matrices: Okapi BM25 and TF-IDF cosine."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy import sparse

from .ranking import Ranked, top_k_rows
from .text import Tokenizer


class TermCounts:
    """Vocabulary plus a (documents x terms) CSR matrix of raw term counts."""

    def __init__(self, documents: Sequence[str], tokenizer: Callable[[str], list[str]]) -> None:
        self.tokenizer = tokenizer
        self.vocab: dict[str, int] = {}
        indptr, indices = [0], []
        for doc in documents:
            for term in tokenizer(doc):
                indices.append(self.vocab.setdefault(term, len(self.vocab)))
            indptr.append(len(indices))
        data = np.ones(len(indices), dtype=np.float64)
        shape = (len(documents), len(self.vocab))
        counts = sparse.csr_matrix((data, np.asarray(indices, dtype=np.int64), indptr), shape)
        counts.sum_duplicates()  # merges repeated terms into counts, sorts indices
        self.matrix = counts

    @property
    def n_docs(self) -> int:
        return self.matrix.shape[0]

    def queries(self, queries: Sequence[str]) -> sparse.csr_matrix:
        """(queries x terms) count matrix; out-of-vocabulary terms are dropped."""
        rows, cols = [], []
        for i, q in enumerate(queries):
            for term in self.tokenizer(q):
                j = self.vocab.get(term)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        data = np.ones(len(rows), dtype=np.float64)
        m = sparse.csr_matrix((data, (rows, cols)), shape=(len(queries), len(self.vocab)))
        m.sum_duplicates()
        return m


class _SparseRetriever:
    """Shared search logic: scores = query_vectors @ term_document_weights."""

    name = "sparse"
    _weights: sparse.csr_matrix  # (terms x documents)
    counts: TermCounts

    def _query_vectors(self, q: sparse.csr_matrix) -> sparse.csr_matrix:
        return q

    def score(self, queries: Sequence[str]) -> np.ndarray:
        """Dense (queries x documents) score matrix."""
        q = self._query_vectors(self.counts.queries(queries))
        return np.asarray((q @ self._weights).todense(), dtype=np.float64)

    def rank(self, queries: Sequence[str], k: int = 100) -> list[Ranked]:
        """Top-k documents per query (documents sharing no term are never returned)."""
        return top_k_rows(self.score(queries), k, positive_only=True)


class BM25(_SparseRetriever):
    """Okapi BM25 with Lucene's idf, ``log(1 + (N - df + 0.5) / (df + 0.5))``.

    score(q, d) = sum over query terms t of
        idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl)).
    A query term that appears twice counts twice. Defaults k1=1.2, b=0.75 match
    Elasticsearch/Lucene.
    """

    name = "bm25"

    def __init__(self, counts: TermCounts, k1: float = 1.2, b: float = 0.75) -> None:
        self.counts, self.k1, self.b = counts, k1, b
        tf = counts.matrix.tocoo()
        n_docs = counts.n_docs
        df = np.bincount(tf.col, minlength=len(counts.vocab))
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        doc_len = np.asarray(counts.matrix.sum(axis=1)).ravel()
        self.avgdl = float(doc_len.mean())
        norm = k1 * (1.0 - b + b * doc_len[tf.row] / self.avgdl)
        w = self.idf[tf.col] * tf.data * (k1 + 1.0) / (tf.data + norm)
        self._weights = sparse.csr_matrix((w, (tf.col, tf.row)), shape=(len(counts.vocab), n_docs))

    @classmethod
    def fit(
        cls,
        documents: Sequence[str],
        tokenizer: Callable[[str], list[str]] | None = None,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> BM25:
        return cls(TermCounts(documents, tokenizer or Tokenizer()), k1=k1, b=b)

    def with_params(self, k1: float, b: float) -> BM25:
        """Same index (no re-tokenizing), different k1/b."""
        return BM25(self.counts, k1=k1, b=b)


class TfidfCosine(_SparseRetriever):
    """TF-IDF vectors (sublinear tf ``1 + ln tf``, smoothed idf ``ln((1+N)/(1+df)) + 1``),
    L2-normalised, scored by cosine similarity. Same weighting as scikit-learn's
    ``TfidfVectorizer(sublinear_tf=True)``."""

    name = "tfidf"

    def __init__(self, counts: TermCounts) -> None:
        self.counts = counts
        tf = counts.matrix.tocsr(copy=True)
        n_docs = counts.n_docs
        df = np.bincount(tf.indices, minlength=len(counts.vocab))
        self.idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        self._weights = self._normalise(tf).T.tocsr()

    def _normalise(self, tf: sparse.csr_matrix) -> sparse.csr_matrix:
        tf = tf.astype(np.float64)
        tf.data = 1.0 + np.log(tf.data)
        x = tf.multiply(self.idf[None, :]).tocsr()
        norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        return sparse.diags(1.0 / norms) @ x

    def _query_vectors(self, q: sparse.csr_matrix) -> sparse.csr_matrix:
        return self._normalise(q)

    @classmethod
    def fit(
        cls, documents: Sequence[str], tokenizer: Callable[[str], list[str]] | None = None
    ) -> TfidfCosine:
        return cls(TermCounts(documents, tokenizer or Tokenizer()))
