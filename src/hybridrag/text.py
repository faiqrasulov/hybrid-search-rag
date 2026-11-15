"""Tokenizer for the lexical retrievers: lowercase, split, drop stopwords, optional stemming."""

from __future__ import annotations

import re
from collections.abc import Iterable

import snowballstemmer

# Lucene's EnglishAnalyzer stop set (the "english" analyzer used by BEIR's Elasticsearch BM25).
LUCENE_ENGLISH_STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the their "
    "then there these they this to was will with".split()
)

_POSSESSIVE = re.compile(r"['’]s\b")
_TOKEN = re.compile(r"[^\W_]+")  # runs of letters/digits (Unicode-aware), split on the rest


class Tokenizer:
    """Callable text -> list of terms.

    Args:
        stopwords: terms to drop after lowercasing (default: Lucene's English stop set).
        stemmer: a Snowball algorithm name (``"porter"``, ``"english"``) or ``None``.
    """

    def __init__(
        self,
        stopwords: Iterable[str] | None = LUCENE_ENGLISH_STOPWORDS,
        stemmer: str | None = "porter",
    ) -> None:
        self.stopwords = frozenset(stopwords or ())
        self.stemmer_name = stemmer
        self._stemmer = snowballstemmer.stemmer(stemmer) if stemmer else None
        self._memo: dict[str, str] = {}

    def _stem(self, word: str) -> str:
        stem = self._memo.get(word)
        if stem is None:
            stem = self._memo[word] = self._stemmer.stemWord(word)  # type: ignore[union-attr]
        return stem

    def __call__(self, text: str) -> list[str]:
        words = _TOKEN.findall(_POSSESSIVE.sub("", text.lower()))
        words = [w for w in words if w not in self.stopwords]
        if self._stemmer is not None:
            words = [self._stem(w) for w in words]
        return words

    def __repr__(self) -> str:
        return f"Tokenizer(stopwords={len(self.stopwords)}, stemmer={self.stemmer_name!r})"
