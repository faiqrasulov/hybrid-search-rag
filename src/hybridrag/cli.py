"""Command line: `hybridrag search | answer | evaluate | download`."""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from collections.abc import Sequence

from .engine import SYSTEMS, Hit, SearchEngine


def _build_engine(data_dir: str, system: str) -> SearchEngine:
    """SciFact engine with only the models the chosen system needs."""
    return SearchEngine.from_scifact(
        data_dir, dense=system not in ("bm25", "tfidf"), rerank=system == "rerank"
    )


def _print_hits(hits: Sequence[Hit], width: int = 100, chars: int = 300) -> None:
    for h in hits:
        print(f"{h.rank:>2}. [{h.doc_id}] {h.title}  (score {h.score:.3f})")
        snippet = h.text if len(h.text) <= chars else h.text[:chars].rsplit(" ", 1)[0] + " ..."
        print(textwrap.indent(textwrap.fill(snippet, width - 4), "    "))


def cmd_search(args: argparse.Namespace) -> int:
    engine = _build_engine(args.data_dir, args.system)
    _print_hits(engine.search(args.query, system=args.system, k=args.k))
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    from .rag import answer, resolve_model

    engine = _build_engine(args.data_dir, args.system)
    model = resolve_model(args.model)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        _print_hits(engine.search(args.question, system=args.system, k=args.k))
        print(
            "\nNote: ANTHROPIC_API_KEY is not set, so no answer was generated. These are the "
            f"passages that would be sent to {model}; set the key to get an answer that "
            "cites them by [doc_id]."
        )
        return 0

    import anthropic

    try:
        result = answer(args.question, k=args.k, system=args.system, engine=engine, model=model)
    except anthropic.AuthenticationError:
        print("Error: the Anthropic API rejected ANTHROPIC_API_KEY.", file=sys.stderr)
        return 1
    except anthropic.RateLimitError:
        print("Error: rate limited by the Anthropic API; retry later.", file=sys.stderr)
        return 1
    except anthropic.APIStatusError as exc:
        print(f"Error: Anthropic API returned {exc.status_code}: {exc.message}", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError:
        print("Error: could not reach the Anthropic API (network).", file=sys.stderr)
        return 1

    print(textwrap.fill(result.text, 100) if "\n" not in result.text else result.text)
    print(f"\nModel: {result.model}   cited: {', '.join(result.citations) or 'none'}")
    if result.invalid_citations:
        print(f"Warning: cited ids not among the passages: {', '.join(result.invalid_citations)}")
    print("\nSources:")
    _print_hits(result.passages)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import main as evaluate_main

    evaluate_main(args.rest)
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from .data import download_scifact

    print(f"SciFact ready in {download_scifact(args.data_dir)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hybridrag", description="Measured hybrid retrieval + RAG.")
    p.add_argument("--data-dir", default=os.environ.get("HYBRIDRAG_DATA", "data"))
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("search", help="search SciFact with one retrieval system")
    s.add_argument("query")
    s.add_argument("--system", choices=SYSTEMS, default="hybrid")
    s.add_argument("-k", type=int, default=10)
    s.set_defaults(func=cmd_search)

    a = sub.add_parser("answer", help="retrieve passages and answer with Claude, citing them")
    a.add_argument("question")
    a.add_argument("--system", choices=SYSTEMS, default="hybrid")
    a.add_argument("-k", type=int, default=5)
    a.add_argument("--model", default=None, help="Claude model id (default $HYBRIDRAG_MODEL)")
    a.set_defaults(func=cmd_answer)

    e = sub.add_parser("evaluate", help="full evaluation -> reports/ (same as make run)")
    e.add_argument("rest", nargs=argparse.REMAINDER, help="options for hybridrag.evaluate")
    e.set_defaults(func=cmd_evaluate)

    d = sub.add_parser("download", help="download BEIR SciFact into the data directory")
    d.set_defaults(func=cmd_download)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
