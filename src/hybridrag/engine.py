"""One search interface over every retrieval system in the project."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .data import Corpus, load_scifact
from .dense import DenseRetriever, Encoder, SentenceTransformerEncoder
from .fusion import convex, rrf
from .lexical import BM25, TermCounts, TfidfCosine
from .ranking import Ranked
from .rerank import PairScorer, apply_rerank, interpolate
from .text import Tokenizer

SYSTEMS = ("tfidf", "bm25", "dense", "rrf", "hybrid", "rerank")
# Dense weight and normalisation of the convex hybrid, tuned on the SciFact *train* split by
# `make run` (see reports/metrics.json -> fusion_tuning).
DEFAULT_DENSE_WEIGHT = 0.45
DEFAULT_NORMALIZATION = "minmax"
DEFAULT_CANDIDATES = 100  # top-k taken from each first-stage retriever
DEFAULT_RERANK_DEPTH = 20
DEFAULT_RERANK_BASE = "hybrid"  # first stage under the reranker (best on train nDCG@10)
DEFAULT_RERANK_WEIGHT = 0.5  # cross-encoder weight in the z-score interpolation (train-tuned)


@dataclass(frozen=True)
class Hit:
    """One search result."""

    rank: int
    doc_id: str
    title: str
    text: str
    score: float


class SearchEngine:
    """BM25, TF-IDF, dense, RRF and convex hybrids, and cross-encoder reranking over a corpus.

    ``system`` names: ``tfidf``, ``bm25``, ``dense``, ``rrf`` (BM25 + dense, k=60),
    ``hybrid`` (convex BM25 + dense) and ``rerank`` (cross-encoder over the top
    ``rerank_depth`` candidates of the ``rerank_base`` system, its z-scored logits
    interpolated with the first-stage scores using ``rerank_weight``; 1.0 = pure
    cross-encoder order).
    """

    def __init__(
        self,
        corpus: Corpus,
        encoder: Encoder | None = None,
        reranker: PairScorer | None = None,
        tokenizer: Tokenizer | None = None,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        normalization: str = DEFAULT_NORMALIZATION,
        candidates: int = DEFAULT_CANDIDATES,
        rerank_depth: int = DEFAULT_RERANK_DEPTH,
        rerank_base: str = DEFAULT_RERANK_BASE,
        rerank_weight: float = DEFAULT_RERANK_WEIGHT,
        cache_dir: str | Path | None = None,
        verbose: bool = False,
    ) -> None:
        self.corpus = corpus
        self.documents = corpus.documents()
        counts = TermCounts(self.documents, tokenizer or Tokenizer())
        self.bm25 = BM25(counts)
        self.tfidf = TfidfCosine(counts)
        self.dense = (
            DenseRetriever.fit(encoder, self.documents, cache_dir, verbose=verbose)
            if encoder is not None
            else None
        )
        self.reranker = reranker
        self.dense_weight = dense_weight
        self.normalization = normalization
        self.candidates = candidates
        self.rerank_depth = rerank_depth
        if rerank_base not in SYSTEMS[:-1]:
            raise ValueError(f"rerank_base must be one of {SYSTEMS[:-1]}")
        self.rerank_base = rerank_base
        self.rerank_weight = rerank_weight

    @classmethod
    def from_scifact(
        cls,
        data_dir: str | Path = "data",
        dense: bool = True,
        rerank: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> SearchEngine:
        """Engine over BEIR SciFact with the default MiniLM encoder (embeddings cached)."""
        corpus = load_scifact(data_dir).corpus
        encoder = SentenceTransformerEncoder() if dense else None
        reranker = None
        if rerank:
            from .rerank import CrossEncoderReranker

            reranker = CrossEncoderReranker()
        return cls(
            corpus,
            encoder,
            reranker,
            cache_dir=Path(data_dir) / "embeddings",
            verbose=verbose,
            **kwargs,
        )

    def _need_dense(self) -> DenseRetriever:
        if self.dense is None:
            raise ValueError("this engine was built without a dense encoder")
        return self.dense

    def first_stage(self, queries: Sequence[str], system: str) -> list[Ranked]:
        """Candidate lists (top ``candidates``) for a batch of queries."""
        n = self.candidates
        if system == "tfidf":
            return self.tfidf.rank(queries, n)
        if system == "bm25":
            return self.bm25.rank(queries, n)
        if system == "dense":
            return self._need_dense().rank(queries, n)
        if system == "rerank":
            return self.first_stage(queries, self.rerank_base)
        if system in ("rrf", "hybrid"):
            lexical = self.bm25.rank(queries, n)
            dense = self._need_dense().rank(queries, n)
            if system == "rrf":
                return [rrf([a, b]).head(n) for a, b in zip(lexical, dense, strict=True)]
            return [
                convex(a, b, self.dense_weight, self.normalization).head(n)
                for a, b in zip(lexical, dense, strict=True)
            ]
        raise ValueError(f"unknown system {system!r}; choose from {SYSTEMS}")

    def rerank(self, query: str, candidates: Ranked) -> Ranked:
        """Cross-encoder rerank of the top ``rerank_depth`` candidates of one query."""
        if self.reranker is None:
            raise ValueError("this engine was built without a reranker")
        head = candidates.head(self.rerank_depth)
        ce = self.reranker.score(query, [self.documents[i] for i in head.indices])
        return apply_rerank(candidates, interpolate(ce, head.scores, self.rerank_weight))

    def rank(self, queries: Sequence[str], system: str = "hybrid") -> list[Ranked]:
        """Full ranking (up to ``candidates`` documents) for each query."""
        ranked = self.first_stage(queries, system)
        if system == "rerank":
            ranked = [self.rerank(q, r) for q, r in zip(queries, ranked, strict=True)]
        return ranked

    def search(self, query: str, system: str = "hybrid", k: int = 10) -> list[Hit]:
        """Top-k hits for one query."""
        ranked = self.rank([query], system)[0].head(k)
        c = self.corpus
        pairs = zip(ranked.indices.tolist(), ranked.scores.tolist(), strict=True)
        return [
            Hit(rank, c.doc_ids[i], c.titles[i], c.texts[i], s)
            for rank, (i, s) in enumerate(pairs, start=1)
        ]
