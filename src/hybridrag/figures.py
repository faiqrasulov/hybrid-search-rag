"""Report figures, drawn from the metrics dictionary written by :mod:`hybridrag.evaluate`."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
# Fixed categorical order: a system keeps its colour in every figure.
COLORS = {
    "tfidf": "#2a78d6",
    "bm25": "#eb6834",
    "dense": "#1baf7a",
    "rrf": "#eda100",
    "hybrid": "#e87ba4",
    "rerank_ce": "#008300",
    "rerank": "#6250d6",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": INK_2,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "font.size": 10,
        }
    )


def quality_with_ci(metrics: dict, path: Path) -> Path:
    """nDCG@10 and Recall@100 per system with 95% bootstrap CIs (the hero figure)."""
    systems = metrics["systems"]
    names = list(systems)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, metric, title in (
        (axes[0], "ndcg@10", "nDCG@10"),
        (axes[1], "recall@100", "Recall@100"),
    ):
        means = [systems[s]["test"][metric]["mean"] for s in names]
        lows = [systems[s]["test"][metric]["ci95"][0] for s in names]
        highs = [systems[s]["test"][metric]["ci95"][1] for s in names]
        for y, (s, m, lo, hi) in enumerate(zip(names, means, lows, highs, strict=True)):
            ax.plot([lo, hi], [y, y], color=COLORS[s], lw=2.2, solid_capstyle="round")
            ax.plot(m, y, "o", ms=8, color=COLORS[s], mec=SURFACE, mew=2)
            ax.annotate(
                f"{m:.3f}",
                (hi, y),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=INK,
            )
        lo_all, hi_all = min(lows), max(highs)
        pad = max((hi_all - lo_all) * 0.25, 0.02)
        ax.set_xlim(lo_all - pad * 0.4, hi_all + pad)
        ax.set_title(f"{title} (test, 300 queries)", loc="left")
        ax.grid(axis="y", visible=False)
        if metric == "ndcg@10":
            ref = metrics["beir_reference"]["published_bm25_ndcg@10"]
            ax.axvline(ref, color=INK_2, lw=1, ls="--")
            ax.annotate(
                f"BEIR's published BM25: {ref}",
                (ref, len(names) - 0.3),
                xytext=(4, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8.5,
                color=INK_2,
            )
        ax.set_ylim(-0.6, len(names) - 0.05)
    axes[0].set_yticks(range(len(names)), [systems[s]["label"] for s in names])
    fig.suptitle(
        "BEIR SciFact retrieval quality with 95% bootstrap confidence intervals",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def recall_curves(metrics: dict, path: Path) -> Path:
    """Mean Recall@k for k = 1..100 for the first-stage systems."""
    curves = metrics["recall_curves"]
    ks = curves["k"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for s in ("tfidf", "bm25", "dense", "rrf", "hybrid"):
        ax.plot(ks, curves[s], color=COLORS[s], lw=2, label=metrics["systems"][s]["label"])
    ax.set_xscale("log")
    ticks = [1, 2, 5, 10, 20, 50, 100]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlabel("k (documents retrieved)")
    ax.set_ylabel("Recall@k")
    ax.set_title("Recall@k on SciFact test (first-stage systems)", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def quality_vs_latency(metrics: dict, path: Path) -> Path:
    """nDCG@10 against median single-query latency (whiskers reach the p95)."""
    systems = metrics["systems"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    offsets = {"rerank_ce": (8, -12), "rerank": (8, 6), "tfidf": (8, -10)}
    for s, entry in systems.items():
        med, p95 = entry["latency_ms"]["median"], entry["latency_ms"]["p95"]
        y = entry["test"]["ndcg@10"]["mean"]
        ax.plot([med, p95], [y, y], color=COLORS[s], lw=1.6, alpha=0.6)
        ax.plot(med, y, "o", ms=9, color=COLORS[s], mec=SURFACE, mew=2)
        ax.annotate(
            entry["label"],
            (med, y),
            xytext=offsets.get(s, (8, 4)),
            textcoords="offset points",
            fontsize=9,
            color=INK,
        )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.margins(y=0.12)
    ax.set_xlabel("Latency per query, ms (dot = median, line to p95; CPU, log scale)")
    ax.set_ylabel("nDCG@10 (test)")
    ax.set_title("Quality vs single-query latency (CPU)", loc="left")
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi * 3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def tuning_curves(metrics: dict, path: Path) -> Path:
    """Left: fusion weight on train (test dashed, for reference). Right: reranker weight."""
    fusion = metrics["fusion_tuning"]
    sel = fusion["selected"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    norm_colors = {"minmax": COLORS["tfidf"], "zscore": COLORS["hybrid"]}
    for norm, g in fusion["grid"].items():
        c = norm_colors[norm]
        ax1.plot(g["dense_weight"], g["train_ndcg@10"], color=c, lw=2, label=f"{norm}, train")
        ax1.plot(
            g["dense_weight"],
            g["test_ndcg@10"],
            color=c,
            lw=1.4,
            ls="--",
            alpha=0.8,
            label=f"{norm}, test (not used to choose)",
        )
    ax1.axhline(fusion["rrf_train_ndcg@10"], color=INK_2, lw=1, ls=":", label="RRF k=60, train")
    ax1.axvline(sel["dense_weight"], color=INK, lw=1)
    ax1.set_xlabel("Dense weight in the convex combination (0 = BM25 only, 1 = dense only)")
    ax1.set_ylabel("nDCG@10")
    ax1.set_title(
        f"Fusion weight, chosen on train: {sel['normalization']}, {sel['dense_weight']}", loc="left"
    )
    ax1.legend(loc="lower center", fontsize=8.5)

    rr = metrics["rerank"]
    ax2.plot(
        rr["weights"],
        rr["train_ndcg@10"],
        color=COLORS["rerank"],
        lw=2,
        marker="o",
        ms=5,
        label="train",
    )
    ax2.plot(
        rr["weights"],
        rr["test_ndcg@10"],
        color=COLORS["rerank"],
        lw=1.4,
        ls="--",
        alpha=0.8,
        label="test (not used to choose)",
    )
    ax2.legend(loc="lower left", fontsize=8.5)
    ax2.axvline(rr["selected_weight"], color=INK, lw=1)
    ax2.set_xlabel("Cross-encoder weight (0 = first stage only, 1 = cross-encoder only)")
    ax2.set_ylabel("nDCG@10")
    ax2.set_title(
        f"Reranker weight, chosen on train: {rr['selected_weight']} (top-{rr['depth']})",
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_all(metrics: dict, out_dir: str | Path) -> list[Path]:
    """Write every figure to ``out_dir`` and return the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _style()
    return [
        quality_with_ci(metrics, out_dir / "retrieval_quality.png"),
        recall_curves(metrics, out_dir / "recall_at_k.png"),
        quality_vs_latency(metrics, out_dir / "quality_vs_latency.png"),
        tuning_curves(metrics, out_dir / "tuning_on_train.png"),
    ]


if __name__ == "__main__":
    import json
    import sys

    src = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/metrics.json")
    for out in make_all(json.loads(src.read_text()), src.parent / "figures"):
        print(f"wrote {out}")
