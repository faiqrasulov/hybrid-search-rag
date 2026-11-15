"""Download and load the BEIR SciFact benchmark (corpus, queries, relevance judgements)."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
# MD5 published by BEIR for this archive (integrity check, not security).
SCIFACT_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"
_EXPECTED_MEMBERS = (
    "scifact/corpus.jsonl",
    "scifact/queries.jsonl",
    "scifact/qrels/train.tsv",
    "scifact/qrels/test.tsv",
)

Qrels = dict[str, dict[str, int]]  # query id -> {doc id: graded relevance}


@dataclass
class Corpus:
    """Documents in a fixed order; row ``i`` of every index refers to ``doc_ids[i]``."""

    doc_ids: list[str]
    titles: list[str]
    texts: list[str]

    def __len__(self) -> int:
        return len(self.doc_ids)

    def documents(self) -> list[str]:
        """Title and abstract joined, the text every retriever indexes."""
        return [f"{t}. {x}" if t else x for t, x in zip(self.titles, self.texts, strict=True)]


@dataclass
class Dataset:
    """A BEIR-style retrieval benchmark."""

    corpus: Corpus
    queries: dict[str, str]
    qrels: dict[str, Qrels] = field(default_factory=dict)  # split -> qrels

    def split(self, name: str) -> tuple[list[str], list[str], Qrels]:
        """Return (query ids, query texts, qrels) for the judged queries of a split."""
        qrels = self.qrels[name]
        qids = [q for q in qrels if q in self.queries]
        return qids, [self.queries[q] for q in qids], {q: qrels[q] for q in qids}


def _md5(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_scifact(data_dir: str | Path = "data", url: str = SCIFACT_URL) -> Path:
    """Download and unpack SciFact into ``data_dir/scifact`` (cached; no-op if present)."""
    data_dir = Path(data_dir)
    target = data_dir / "scifact"
    if all((data_dir / m).exists() for m in _EXPECTED_MEMBERS):
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "scifact.zip"
    if not archive.exists():
        part = archive.with_suffix(".zip.part")
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, part.open("wb") as out:
                shutil.copyfileobj(resp, out)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            part.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download SciFact from {url} ({exc}). Check your internet "
                f"connection, or download the file manually and save it as {archive}."
            ) from exc
        part.rename(archive)
    digest = _md5(archive)
    if url == SCIFACT_URL and digest != SCIFACT_MD5:
        archive.unlink()
        raise RuntimeError(
            f"Checksum mismatch for {archive} (got {digest}); the file was deleted, "
            "please retry the download."
        )
    with zipfile.ZipFile(archive) as zf:
        for member in _EXPECTED_MEMBERS:  # extract only known files: no path traversal
            dest = data_dir / member
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
    return target


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_qrels(path: str | Path) -> Qrels:
    """Read a BEIR qrels TSV (header ``query-id corpus-id score``)."""
    qrels: Qrels = {}
    with Path(path).open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for qid, did, score in reader:
            qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def load_beir(folder: str | Path) -> Dataset:
    """Load a BEIR dataset folder (corpus.jsonl, queries.jsonl, qrels/*.tsv)."""
    folder = Path(folder)
    rows = _read_jsonl(folder / "corpus.jsonl")
    corpus = Corpus(
        doc_ids=[str(r["_id"]) for r in rows],
        titles=[r.get("title", "") or "" for r in rows],
        texts=[r.get("text", "") or "" for r in rows],
    )
    queries = {str(r["_id"]): r["text"] for r in _read_jsonl(folder / "queries.jsonl")}
    qrels = {p.stem: read_qrels(p) for p in sorted((folder / "qrels").glob("*.tsv"))}
    return Dataset(corpus=corpus, queries=queries, qrels=qrels)


def load_scifact(data_dir: str | Path = "data") -> Dataset:
    """Download (if needed) and load BEIR SciFact."""
    return load_beir(download_scifact(data_dir))
