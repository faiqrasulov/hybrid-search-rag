import json
from pathlib import Path

import pytest

from hybridrag import engine as engine_module
from hybridrag.engine import SYSTEMS, SearchEngine
from hybridrag.evaluate import LABELS, run_evaluation


def test_every_system_returns_ranked_hits(corpus, fake_encoder, fake_reranker):
    engine = SearchEngine(corpus, encoder=fake_encoder, reranker=fake_reranker, rerank_depth=5)
    for system in SYSTEMS:
        hits = engine.search("aspirin lowers stroke risk", system=system, k=5)
        assert 1 <= len(hits) <= 5
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
        assert all(a.score >= b.score for a, b in zip(hits, hits[1:], strict=False))
        assert hits[0].doc_id == "d4", system


def test_engine_errors_without_models(corpus):
    engine = SearchEngine(corpus)
    assert engine.search("measles vaccine", system="bm25", k=1)[0].doc_id == "d9"
    with pytest.raises(ValueError, match="dense"):
        engine.search("measles", system="dense")
    with pytest.raises(ValueError, match="unknown system"):
        engine.search("measles", system="bm42")


def test_pipeline_smoke_on_synthetic_corpus(tmp_path, dataset, fake_encoder, fake_reranker):
    metrics = run_evaluation(
        data_dir=tmp_path / "data",
        out_dir=tmp_path / "reports",
        dataset=dataset,
        encoder=fake_encoder,
        reranker=fake_reranker,
        candidates=10,
        rerank_depth=5,
        rerank_train_queries=3,
        n_boot=200,
        verbose=False,
    )
    saved = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert saved["systems"].keys() == metrics["systems"].keys() == LABELS.keys()
    for entry in saved["systems"].values():
        t = entry["test"]["ndcg@10"]
        assert 0.0 <= t["ci95"][0] <= t["mean"] <= t["ci95"][1] <= 1.0
        assert entry["latency_ms"]["p95"] >= entry["latency_ms"]["median"] >= 0
    # the synthetic queries share words with their documents: BM25 should find them
    assert saved["systems"]["bm25"]["test"]["recall@10"]["mean"] == 1.0
    sel = saved["fusion_tuning"]["selected"]
    assert sel["normalization"] in ("minmax", "zscore") and 0 <= sel["dense_weight"] <= 1
    assert saved["rerank"]["base"] in ("bm25", "dense", "rrf", "hybrid")
    assert len(saved["recall_curves"]["k"]) == 10
    assert {c["metric"] for c in saved["comparisons"]} == {"ndcg@10", "recall@100"}
    assert all(c["p_holm"] >= c["p_value"] - 1e-9 for c in saved["comparisons"])
    figures = sorted(p.name for p in (tmp_path / "reports" / "figures").glob("*.png"))
    assert figures == [
        "quality_vs_latency.png",
        "recall_at_k.png",
        "retrieval_quality.png",
        "tuning_on_train.png",
    ]


def test_engine_defaults_match_the_train_selection_in_reports():
    path = Path(__file__).resolve().parents[1] / "reports" / "metrics.json"
    if not path.exists():
        pytest.skip("reports/metrics.json has not been generated")
    m = json.loads(path.read_text())
    assert engine_module.DEFAULT_DENSE_WEIGHT == m["fusion_tuning"]["selected"]["dense_weight"]
    assert engine_module.DEFAULT_NORMALIZATION == m["fusion_tuning"]["selected"]["normalization"]
    assert engine_module.DEFAULT_RERANK_BASE == m["rerank"]["base"]
    assert engine_module.DEFAULT_RERANK_WEIGHT == m["rerank"]["selected_weight"]
    assert engine_module.DEFAULT_RERANK_DEPTH == m["rerank"]["depth"]
