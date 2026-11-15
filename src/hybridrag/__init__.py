"""hybridrag: measured hybrid retrieval (BM25, dense, fusion, reranking) with a cited RAG layer."""

from .engine import SYSTEMS, Hit, SearchEngine
from .rag import Answer, answer

__all__ = ["SYSTEMS", "Answer", "Hit", "SearchEngine", "answer"]
__version__ = "0.1.0"
