"""Tiny synthetic corpus, a deterministic fake encoder and a fake reranker (no downloads)."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np
import pytest

from hybridrag.data import Corpus, Dataset

DOCS = [
    ("Vitamin D and bone density", "Vitamin D supplementation increases bone mineral density."),
    ("Statins and cholesterol", "Statins lower LDL cholesterol and reduce heart attacks."),
    ("Exercise and depression", "Aerobic exercise reduces symptoms of depression in adults."),
    ("Smoking and lung cancer", "Cigarette smoking causes most cases of lung cancer."),
    ("Aspirin and stroke", "Low-dose aspirin lowers the risk of ischemic stroke."),
    ("Sleep and memory", "Sleep consolidates memory after learning new tasks."),
    ("Coffee and liver", "Coffee drinking is linked to lower liver fibrosis."),
    ("Insulin and glucose", "Insulin lowers blood glucose by promoting uptake in muscle."),
    ("Antibiotics and resistance", "Overuse of antibiotics drives bacterial resistance."),
    ("Measles vaccine", "The measles vaccine prevents measles infection in children."),
    ("Salt and blood pressure", "High salt intake raises blood pressure in adults."),
    ("Gut microbiome and obesity", "The gut microbiome differs between obese and lean mice."),
]
QUERIES = {
    "q1": "vitamin D improves bone density",
    "q2": "statins reduce cholesterol",
    "q3": "exercise helps depression",
    "q4": "smoking causes lung cancer",
    "q5": "aspirin prevents stroke",
    "q6": "sleep improves memory",
    "q7": "salt increases blood pressure",
    "q8": "antibiotic overuse causes resistance",
}


class FakeEncoder:
    """Hashed bag-of-words embedding: deterministic, L2-normalised, no model download."""

    name = "fake-encoder"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.calls = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                h = int(hashlib.md5(word[:5].encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
            norm = np.linalg.norm(out[i])
            out[i] /= norm if norm > 0 else 1.0
        return out


class FakeReranker:
    """Scores a pair by the number of shared 4-letter word prefixes."""

    name = "fake-reranker"

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        q = {w[:4] for w in re.findall(r"[a-z]+", query.lower())}
        return np.array(
            [len(q & {w[:4] for w in re.findall(r"[a-z]+", d.lower())}) for d in documents],
            dtype=np.float64,
        )


@pytest.fixture
def corpus() -> Corpus:
    return Corpus(
        doc_ids=[f"d{i}" for i in range(len(DOCS))],
        titles=[t for t, _ in DOCS],
        texts=[x for _, x in DOCS],
    )


@pytest.fixture
def dataset(corpus: Corpus) -> Dataset:
    qrels = {f"q{i}": {f"d{i - 1}": 1} for i in range(1, 7)}
    qrels["q7"] = {"d10": 1}
    qrels["q8"] = {"d8": 1}
    train = {q: qrels[q] for q in ("q1", "q2", "q3", "q4")}
    test = {q: qrels[q] for q in ("q5", "q6", "q7", "q8")}
    return Dataset(corpus=corpus, queries=dict(QUERIES), qrels={"train": train, "test": test})


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def fake_reranker() -> FakeReranker:
    return FakeReranker()
