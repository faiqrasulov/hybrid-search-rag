"""Full evaluation on BEIR SciFact -> reports/metrics.json and figures.

Every tunable choice (BM25 k1/b ablation, fusion weight and normalisation, the first stage
under the reranker, the reranker interpolation weight) is selected on the *train* split's
queries; the test split is only scored.

Run with ``python -m hybridrag.evaluate`` or ``hybridrag evaluate``.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import platform
import time
from collections.abc import Callable, Sequence
from importlib import metadata
from pathlib import Path
from typing import TypeVar

import numpy as np
from threadpoolctl import threadpool_limits

from .data import SCIFACT_URL, Dataset, load_scifact
from .dense import DEFAULT_ENCODER, Encoder, SentenceTransformerEncoder
from .engine import SearchEngine
from .fusion import convex, rrf
from .lexical import BM25, best_fields
from .metrics import (
    bootstrap_ci,
    bootstrap_indices,
    holm,
    paired_bootstrap,
    per_query,
    recall_curve,
)
from .ranking import Ranked, top_k, top_k_rows
from .rerank import DEFAULT_RERANKER, PairScorer, apply_rerank, interpolate
from .text import Tokenizer

BEIR_BM25_SCIFACT_NDCG10 = 0.665  # Thakur et al. 2021, Table 2 (Elasticsearch, two fields)
# Pyserini's BEIR reproductions (Lucene BM25): "flat" = title + text as one field,
# "multifield" = separate fields. https://castorini.github.io/pyserini/2cr/beir.html
PYSERINI_SCIFACT = {
    "flat_ndcg@10": 0.679,
    "flat_recall@100": 0.925,
    "multifield_ndcg@10": 0.665,
    "multifield_recall@100": 0.908,
}
ALPHAS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
RERANK_WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.1), 1)
K1_GRID = (0.5, 0.9, 1.2, 1.5, 2.0)
B_GRID = (0.3, 0.5, 0.75, 0.9, 1.0)
LABELS = {
    "tfidf": "TF-IDF cosine",
    "bm25": "BM25",
    "dense": "Dense (MiniLM-L6)",
    "rrf": "Hybrid RRF",
    "hybrid": "Hybrid convex",
    "rerank_ce": "Hybrid + CE rerank",
    "rerank": "Hybrid + CE interp.",
}
T = TypeVar("T")
FIRST_STAGE = ("tfidf", "bm25", "dense", "rrf", "hybrid")
COMPARISONS = [
    ("bm25", "tfidf"),
    ("dense", "bm25"),
    ("rrf", "bm25"),
    ("hybrid", "bm25"),
    ("hybrid", "dense"),
    ("hybrid", "rrf"),
    ("rerank_ce", "hybrid"),
    ("rerank", "hybrid"),
]


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg, flush=True)


def _run(rankings: Sequence[Ranked], qids: Sequence[str], doc_ids: Sequence[str]) -> dict:
    return {q: [doc_ids[i] for i in r.indices] for q, r in zip(qids, rankings, strict=True)}


def _ms(times: Sequence[float]) -> dict[str, float]:
    t = np.asarray(times) * 1000.0
    return {"median": float(np.median(t)), "p95": float(np.percentile(t, 95))}


def _round(x, nd: int = 4):
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_round(v, nd) for v in x]
    if isinstance(x, float | np.floating):
        return round(float(x), nd)
    return x


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("numpy", "scipy", "torch", "sentence-transformers", "transformers"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def _single_blas_thread(fn: Callable[..., T]) -> Callable[..., T]:
    """Run ``fn`` with numpy's BLAS pinned to one thread.

    numpy's BLAS thread pool and PyTorch's oversubscribe the CPU when they alternate, as they
    do in single-query search: in a test on this machine that roughly tripled query-encoding
    time and pushed the p95 of the 0.2 ms dense search to ~6 ms.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> T:
        with threadpool_limits(limits=1, user_api="blas"):
            return fn(*args, **kwargs)

    return wrapper


@_single_blas_thread
def run_evaluation(
    data_dir: str | Path = "data",
    out_dir: str | Path = "reports",
    dataset: Dataset | None = None,
    encoder: Encoder | None = None,
    reranker: PairScorer | None = None,
    candidates: int = 100,
    rerank_depth: int = 20,
    rerank_train_queries: int | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
    make_figures: bool = True,
    verbose: bool = True,
) -> dict:
    """Run every experiment and write ``metrics.json`` (and figures) to ``out_dir``."""
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = dataset or load_scifact(data_dir)
    corpus, doc_ids = ds.corpus, ds.corpus.doc_ids
    tr_ids, tr_text, tr_qrels = ds.split("train")
    te_ids, te_text, te_qrels = ds.split("test")
    _log(f"SciFact: {len(corpus)} docs, {len(tr_ids)} train / {len(te_ids)} test queries", verbose)

    # ---- indexes -------------------------------------------------------------------------
    t0 = time.perf_counter()
    if encoder is None:
        _log(f"loading {DEFAULT_ENCODER}", verbose)
        encoder = SentenceTransformerEncoder()
    engine = SearchEngine(
        corpus,
        encoder=encoder,
        candidates=candidates,
        rerank_depth=rerank_depth,
        cache_dir=Path(data_dir) / "embeddings",
        verbose=verbose,
    )
    timings["index_and_embed_corpus_s"] = time.perf_counter() - t0
    dense = engine.dense
    assert dense is not None

    # ---- first-stage rankings (batched) ------------------------------------------------------
    t0 = time.perf_counter()
    lists: dict[str, dict[str, list[Ranked]]] = {"train": {}, "test": {}}
    for split, texts in (("train", tr_text), ("test", te_text)):
        lists[split]["tfidf"] = engine.tfidf.rank(texts, candidates)
        lists[split]["bm25"] = engine.bm25.rank(texts, candidates)
        q_emb = encoder.encode(texts)
        lists[split]["dense"] = [top_k(row, candidates) for row in dense.score_embeddings(q_emb)]
        lists[split]["rrf"] = [
            rrf([a, b]).head(candidates)
            for a, b in zip(lists[split]["bm25"], lists[split]["dense"], strict=True)
        ]
    timings["first_stage_batch_s"] = time.perf_counter() - t0

    def mean_ndcg(rankings: Sequence[Ranked], split: str) -> float:
        qids, qrels = (tr_ids, tr_qrels) if split == "train" else (te_ids, te_qrels)
        return float(per_query(_run(rankings, qids, doc_ids), qrels, qids)["ndcg@10"].mean())

    # ---- fusion tuning on train ------------------------------------------------------------
    _log("tuning the convex fusion weight on train", verbose)
    grid: dict[str, dict[str, list[float]]] = {}
    best = (-1.0, "minmax", 0.0)
    for norm in ("minmax", "zscore"):
        grid[norm] = {"dense_weight": ALPHAS.tolist(), "train_ndcg@10": [], "test_ndcg@10": []}
        for alpha in ALPHAS:
            for split in ("train", "test"):
                fused = [
                    convex(a, b, float(alpha), norm).head(candidates)
                    for a, b in zip(lists[split]["bm25"], lists[split]["dense"], strict=True)
                ]
                grid[norm][f"{split}_ndcg@10"].append(mean_ndcg(fused, split))
            score = grid[norm]["train_ndcg@10"][-1]
            if score > best[0] + 1e-12:
                best = (score, norm, float(alpha))
    _, best_norm, best_alpha = best
    engine.dense_weight, engine.normalization = best_alpha, best_norm
    for split in ("train", "test"):
        lists[split]["hybrid"] = [
            convex(a, b, best_alpha, best_norm).head(candidates)
            for a, b in zip(lists[split]["bm25"], lists[split]["dense"], strict=True)
        ]
    _log(f"  selected {best_norm}, dense weight {best_alpha}", verbose)

    # ---- BM25 ablations ----------------------------------------------------------------------
    _log("BM25 ablations (stemming; k1/b grid on train)", verbose)
    bm25_grid = []
    for k1 in K1_GRID:
        for b in B_GRID:
            variant = engine.bm25.with_params(k1, b)
            bm25_grid.append(
                {
                    "k1": k1,
                    "b": b,
                    "train_ndcg@10": mean_ndcg(variant.rank(tr_text, candidates), "train"),
                }
            )
    tuned = max(bm25_grid, key=lambda r: r["train_ndcg@10"])
    tuned_bm25 = engine.bm25.with_params(tuned["k1"], tuned["b"])
    nostem = BM25.fit(engine.documents, Tokenizer(stemmer=None))
    # BEIR's Elasticsearch setup: title and abstract as separate BM25 fields, combined with
    # multi_match best_fields (tie_breaker 0.5); here instead the two are one field.
    fields = [BM25.fit(corpus.titles), BM25.fit(corpus.texts)]
    multifield = best_fields([f.score(te_text) for f in fields], tie_breaker=0.5)
    ablation_lists = {
        "bm25_no_stemming": nostem.rank(te_text, candidates),
        "bm25_tuned_k1_b": tuned_bm25.rank(te_text, candidates),
        "bm25_multifield_beir_style": top_k_rows(multifield, candidates, positive_only=True),
    }

    # ---- reranking ---------------------------------------------------------------------------
    base_train = {s: mean_ndcg(lists["train"][s], "train") for s in FIRST_STAGE}
    rerank_base = max(("bm25", "dense", "rrf", "hybrid"), key=lambda s: base_train[s])
    engine.rerank_base = rerank_base
    if reranker is None:
        from .rerank import CrossEncoderReranker

        _log(f"loading {DEFAULT_RERANKER}", verbose)
        reranker = CrossEncoderReranker()
    engine.reranker = reranker

    rng = np.random.default_rng(seed)
    n_sub = len(tr_ids) if rerank_train_queries is None else min(rerank_train_queries, len(tr_ids))
    sub = np.sort(rng.choice(len(tr_ids), size=n_sub, replace=False))
    _log(f"cross-encoder on {n_sub} train queries x top-{rerank_depth} of {rerank_base}", verbose)
    t0 = time.perf_counter()
    sub_lists = [lists["train"][rerank_base][i] for i in sub]
    sub_ce = [
        reranker.score(tr_text[i], [engine.documents[d] for d in r.indices[:rerank_depth]])
        for i, r in zip(sub, sub_lists, strict=True)
    ]
    timings["rerank_train_s"] = time.perf_counter() - t0
    sub_ids = [tr_ids[i] for i in sub]
    sub_qrels = {q: tr_qrels[q] for q in sub_ids}

    def sub_ndcg(weight: float) -> float:
        reranked = [
            apply_rerank(r, interpolate(s, r.scores[: len(s)], weight))
            for r, s in zip(sub_lists, sub_ce, strict=True)
        ]
        return float(per_query(_run(reranked, sub_ids, doc_ids), sub_qrels)["ndcg@10"].mean())

    weight_curve = [sub_ndcg(float(w)) for w in RERANK_WEIGHTS]
    best_weight = float(RERANK_WEIGHTS[int(np.argmax(weight_curve))])
    engine.rerank_weight = best_weight
    _log(f"  selected cross-encoder weight {best_weight}", verbose)

    # ---- per-query online pass on test: latency of every stage + reranked lists ------------
    _log("timing single-query search on test (and cross-encoder reranking)", verbose)
    t0 = time.perf_counter()
    for q in te_text[:3]:  # warm-up
        engine.rank([q], "rerank")
    lat: dict[str, list[float]] = {
        k: [] for k in ("tfidf", "bm25", "encode", "dense_search", "rrf", "convex", "ce")
    }
    ce_scores, rerank_ce, rerank_interp = [], [], []
    for qi, q in enumerate(te_text):
        t = time.perf_counter()
        engine.tfidf.rank([q], candidates)
        lat["tfidf"].append(time.perf_counter() - t)
        t = time.perf_counter()
        lex = engine.bm25.rank([q], candidates)[0]
        lat["bm25"].append(time.perf_counter() - t)
        t = time.perf_counter()
        emb = encoder.encode([q])
        lat["encode"].append(time.perf_counter() - t)
        t = time.perf_counter()
        den = top_k(dense.score_embeddings(emb)[0], candidates)
        lat["dense_search"].append(time.perf_counter() - t)
        t = time.perf_counter()
        rrf([lex, den]).head(candidates)
        lat["rrf"].append(time.perf_counter() - t)
        t = time.perf_counter()
        convex(lex, den, best_alpha, best_norm).head(candidates)
        lat["convex"].append(time.perf_counter() - t)
        # rerank the batch list (identical up to float noise) so scores align with the depth curve
        base = lists["test"][rerank_base][qi]
        t = time.perf_counter()
        head = base.head(rerank_depth)
        s = reranker.score(q, [engine.documents[d] for d in head.indices])
        reranked = apply_rerank(base, interpolate(s, head.scores, best_weight))
        lat["ce"].append(time.perf_counter() - t)
        ce_scores.append(s)
        rerank_interp.append(reranked)
        rerank_ce.append(apply_rerank(base, s))
    timings["online_test_pass_s"] = time.perf_counter() - t0

    # ---- the same stages with numpy's BLAS unpinned, to quantify thread oversubscription -----
    n_check = min(30, len(te_text))
    blas_threads = os.cpu_count() or 1
    unpinned: dict[str, list[float]] = {"encode": [], "dense_search": [], "ce": []}
    with threadpool_limits(limits=blas_threads, user_api="blas"):
        for qi, q in enumerate(te_text[:n_check]):
            t = time.perf_counter()
            emb = encoder.encode([q])
            unpinned["encode"].append(time.perf_counter() - t)
            t = time.perf_counter()
            top_k(dense.score_embeddings(emb)[0], candidates)
            unpinned["dense_search"].append(time.perf_counter() - t)
            head = lists["test"][rerank_base][qi].head(rerank_depth)
            t = time.perf_counter()
            reranker.score(q, [engine.documents[d] for d in head.indices])
            unpinned["ce"].append(time.perf_counter() - t)
    oversubscription = {
        "queries": n_check,
        "blas_threads_unpinned": blas_threads,
        "pinned_1_thread_ms": {k: _ms(lat[k][:n_check]) for k in unpinned},
        "unpinned_ms": {k: _ms(v) for k, v in unpinned.items()},
    }
    lists["test"]["rerank_ce"] = rerank_ce
    lists["test"]["rerank"] = rerank_interp

    lat_arr = {k: np.asarray(v) for k, v in lat.items()}
    dense_lat = lat_arr["encode"] + lat_arr["dense_search"]
    sys_lat = {
        "tfidf": lat_arr["tfidf"],
        "bm25": lat_arr["bm25"],
        "dense": dense_lat,
        "rrf": lat_arr["bm25"] + dense_lat + lat_arr["rrf"],
        "hybrid": lat_arr["bm25"] + dense_lat + lat_arr["convex"],
    }
    sys_lat["rerank_ce"] = sys_lat[rerank_base] + lat_arr["ce"]
    sys_lat["rerank"] = sys_lat["rerank_ce"]

    # ---- metrics with bootstrap CIs --------------------------------------------------------
    idx = bootstrap_indices(len(te_ids), n_boot, seed)
    scores = {
        s: per_query(_run(r, te_ids, doc_ids), te_qrels, te_ids) for s, r in lists["test"].items()
    }
    systems = {}
    for s, label in LABELS.items():
        entry = {"label": label, "test": {}}
        for metric, values in scores[s].items():
            mean, lo, hi = bootstrap_ci(values, idx)
            entry["test"][metric] = {"mean": mean, "ci95": [lo, hi]}
        if s in lists["train"]:
            entry["train_ndcg@10"] = mean_ndcg(lists["train"][s], "train")
        entry["latency_ms"] = _ms(sys_lat[s])
        systems[s] = entry

    comparisons = []
    for metric in ("ndcg@10", "recall@100"):
        rows = []
        for a, b in COMPARISONS:
            r = paired_bootstrap(scores[a][metric], scores[b][metric], idx)
            rows.append({"a": a, "b": b, "metric": metric, **r})
        for row, p in zip(rows, holm([r["p_value"] for r in rows]), strict=True):
            row["p_holm"] = float(p)
        comparisons.extend(rows)

    ablation_scores = {
        k: per_query(_run(v, te_ids, doc_ids), te_qrels, te_ids) for k, v in ablation_lists.items()
    }
    ablation = {
        "bm25_porter_default": {
            "k1": 1.2,
            "b": 0.75,
            "test_ndcg@10": float(scores["bm25"]["ndcg@10"].mean()),
        },
        "bm25_no_stemming": {
            "test_ndcg@10": float(ablation_scores["bm25_no_stemming"]["ndcg@10"].mean()),
            "train_ndcg@10": mean_ndcg(nostem.rank(tr_text, candidates), "train"),
            "vs_porter": paired_bootstrap(
                ablation_scores["bm25_no_stemming"]["ndcg@10"], scores["bm25"]["ndcg@10"], idx
            ),
        },
        "bm25_tuned_k1_b": {
            "k1": tuned["k1"],
            "b": tuned["b"],
            "train_ndcg@10": tuned["train_ndcg@10"],
            "test_ndcg@10": float(ablation_scores["bm25_tuned_k1_b"]["ndcg@10"].mean()),
            "vs_default": paired_bootstrap(
                ablation_scores["bm25_tuned_k1_b"]["ndcg@10"], scores["bm25"]["ndcg@10"], idx
            ),
        },
        "bm25_multifield_beir_style": {
            "description": "title and abstract as separate BM25 fields, best_fields tie 0.5",
            "test_ndcg@10": float(ablation_scores["bm25_multifield_beir_style"]["ndcg@10"].mean()),
            "vs_single_field": paired_bootstrap(
                ablation_scores["bm25_multifield_beir_style"]["ndcg@10"],
                scores["bm25"]["ndcg@10"],
                idx,
            ),
        },
        "bm25_k1_b_grid_train": bm25_grid,
    }

    test_weight_curve = [  # reported for reference only; selection used train
        mean_ndcg(
            [
                apply_rerank(r, interpolate(s, r.scores[: len(s)], float(w)))
                for r, s in zip(lists["test"][rerank_base], ce_scores, strict=True)
            ],
            "test",
        )
        for w in RERANK_WEIGHTS
    ]
    depth_curve = {"depth": [], "ce_only": [], "interpolated": []}
    base_lists = lists["test"][rerank_base]
    for d in range(5, rerank_depth + 1, 5):
        depth_curve["depth"].append(d)
        for key, w in (("ce_only", 1.0), ("interpolated", best_weight)):
            reranked = [
                apply_rerank(r, interpolate(s[:d], r.scores[:d], w))
                for r, s in zip(base_lists, ce_scores, strict=True)
            ]
            depth_curve[key].append(mean_ndcg(reranked, "test"))

    ks = list(range(1, candidates + 1))
    curves = {
        s: recall_curve(_run(lists["test"][s], te_ids, doc_ids), te_qrels, ks).tolist()
        for s in LABELS
    }

    bm25_test = float(scores["bm25"]["ndcg@10"].mean())
    timings["total_s"] = time.perf_counter() - t_start
    metrics = {
        "dataset": {
            "name": "BEIR SciFact",
            "url": SCIFACT_URL,
            "documents": len(corpus),
            "train_queries": len(tr_ids),
            "test_queries": len(te_ids),
            "train_judgements": sum(len(v) for v in tr_qrels.values()),
            "test_judgements": sum(len(v) for v in te_qrels.values()),
        },
        "config": {
            "tokenizer": "lowercase, Lucene English stopwords, Porter stemmer (snowballstemmer)",
            "bm25": {"k1": 1.2, "b": 0.75, "idf": "Lucene: ln(1 + (N - df + 0.5) / (df + 0.5))"},
            "encoder": getattr(encoder, "name", "unknown"),
            "reranker": getattr(reranker, "name", "unknown"),
            "candidates_per_retriever": candidates,
            "rerank_depth": rerank_depth,
            "rrf_k": 60,
            "bootstrap_resamples": n_boot,
            "seed": seed,
            "selection_split": "train (test is never used to choose anything)",
            "blas_threads": 1,
        },
        "systems": systems,
        "comparisons": comparisons,
        "fusion_tuning": {
            "grid": grid,
            "selected": {
                "normalization": best_norm,
                "dense_weight": best_alpha,
                "train_ndcg@10": best[0],
            },
            "rrf_train_ndcg@10": base_train["rrf"],
        },
        "rerank": {
            "base_selection_train_ndcg@10": base_train,
            "base": rerank_base,
            "depth": rerank_depth,
            "train_queries": n_sub,
            "weights": RERANK_WEIGHTS.tolist(),
            "train_ndcg@10": weight_curve,
            "test_ndcg@10": test_weight_curve,
            "selected_weight": best_weight,
            "test_depth_curve": depth_curve,
            "ce_latency_ms": _ms(lat_arr["ce"]),
        },
        "bm25_ablation": ablation,
        "beir_reference": {
            "published_bm25_ndcg@10": BEIR_BM25_SCIFACT_NDCG10,
            "pyserini_lucene_bm25": PYSERINI_SCIFACT,
            "this_repo_bm25_recall@100": float(scores["bm25"]["recall@100"].mean()),
            "this_repo_bm25_ndcg@10": bm25_test,
            "difference": bm25_test - BEIR_BM25_SCIFACT_NDCG10,
        },
        "stage_latency_ms": {k: _ms(v) for k, v in lat.items()},
        "blas_oversubscription_check": oversubscription,
        "recall_curves": {"k": ks, **curves},
        "runtime_s": timings,
        "environment": _versions(),
    }
    metrics = _round(metrics, 4)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _log(f"wrote {out_dir / 'metrics.json'}", verbose)
    if make_figures:
        from .figures import make_all

        for path in make_all(metrics, out_dir / "figures"):
            _log(f"wrote {path}", verbose)
    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="reports")
    p.add_argument("--rerank-depth", type=int, default=20)
    p.add_argument(
        "--rerank-train-queries",
        type=int,
        default=None,
        help="train queries used to tune the reranker weight (default: all)",
    )
    p.add_argument("--bootstrap", type=int, default=10_000)
    args = p.parse_args(argv)
    m = run_evaluation(
        data_dir=args.data_dir,
        out_dir=args.out,
        rerank_depth=args.rerank_depth,
        rerank_train_queries=args.rerank_train_queries,
        n_boot=args.bootstrap,
    )
    print(f"\n{'system':24s} {'nDCG@10':>8s} {'R@100':>7s} {'MRR@10':>7s} {'p50 ms':>8s}")
    for s in m["systems"].values():
        t = s["test"]
        print(
            f"{s['label']:24s} {t['ndcg@10']['mean']:8.4f} {t['recall@100']['mean']:7.4f} "
            f"{t['mrr@10']['mean']:7.4f} {s['latency_ms']['median']:8.1f}"
        )


if __name__ == "__main__":
    main()
