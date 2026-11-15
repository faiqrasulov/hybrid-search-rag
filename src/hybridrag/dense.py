"""Dense retrieval: sentence-transformer embeddings and exact cosine search in numpy."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from .ranking import Ranked, top_k_rows

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


class Encoder(Protocol):
    """Anything that maps texts to an (n x dim) float32 array of L2-normalised rows."""

    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Thin wrapper around ``sentence_transformers.SentenceTransformer`` (CPU by default).

    The model is downloaded from the Hugging Face Hub on first use.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ENCODER,
        device: str = "cpu",
        batch_size: int = 32,
        max_seq_length: int | None = None,
    ) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import SentenceTransformer  # heavy import, done lazily

        self.name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)


def _fingerprint(encoder_name: str, documents: Sequence[str]) -> str:
    h = hashlib.sha1(encoder_name.encode(), usedforsecurity=False)
    for doc in documents:
        h.update(doc.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


def embed_corpus(
    encoder: Encoder,
    documents: Sequence[str],
    cache_dir: str | Path | None = None,
    chunk_size: int = 256,
    verbose: bool = False,
) -> np.ndarray:
    """Embed documents in small chunks (bounded memory, progress); cache the result as .npy.

    The cache key covers the encoder name and the exact document texts.
    """
    path = None
    if cache_dir is not None:
        slug = encoder.name.replace("/", "__")
        path = Path(cache_dir) / f"{slug}-{_fingerprint(encoder.name, documents)}.npy"
        if path.exists():
            return np.load(path)
    out: np.ndarray | None = None
    start = time.perf_counter()
    for lo in range(0, len(documents), chunk_size):
        emb = encoder.encode(documents[lo : lo + chunk_size])
        if out is None:
            out = np.empty((len(documents), emb.shape[1]), dtype=np.float32)
        out[lo : lo + len(emb)] = emb
        if verbose:
            done = min(lo + chunk_size, len(documents))
            rate = done / (time.perf_counter() - start)
            print(f"  embedded {done}/{len(documents)} docs ({rate:.0f} docs/s)", flush=True)
    assert out is not None, "no documents to embed"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, out)
    return out


class DenseRetriever:
    """Exact (brute-force) cosine search over L2-normalised embeddings.

    One query costs a (dim x N) matrix-vector product: about 2 million multiply-adds for
    SciFact, well under a millisecond. An approximate index (FAISS IVF/HNSW) pays off from
    about a million documents, where the float32 matrix takes ~1.5 GB and brute force costs
    tens of milliseconds per query.
    """

    name = "dense"

    def __init__(self, encoder: Encoder, doc_embeddings: np.ndarray) -> None:
        self.encoder = encoder
        self.doc_embeddings = np.ascontiguousarray(doc_embeddings, dtype=np.float32)

    @classmethod
    def fit(
        cls,
        encoder: Encoder,
        documents: Sequence[str],
        cache_dir: str | Path | None = None,
        verbose: bool = False,
    ) -> DenseRetriever:
        return cls(encoder, embed_corpus(encoder, documents, cache_dir, verbose=verbose))

    def score_embeddings(self, query_embeddings: np.ndarray) -> np.ndarray:
        """Cosine similarities (queries x documents); rows are already unit length."""
        return np.asarray(query_embeddings, dtype=np.float32) @ self.doc_embeddings.T

    def score(self, queries: Sequence[str]) -> np.ndarray:
        return self.score_embeddings(self.encoder.encode(queries))

    def rank(self, queries: Sequence[str], k: int = 100) -> list[Ranked]:
        return top_k_rows(self.score(queries), k)
