# Changelog

Notable changes to this project, newest first.

## 0.1.0 (2026-11-17)

First tagged release. It contains everything below.

## 2026-11-16

- Package versions pinned in `requirements.lock`, with a `make install-locked` target.
- README: the reranker reads 256 tokens per passage; the tuned fusion score is not a zero-shot BEIR number.

## 2026-11-15

- First version: BM25 from scratch, dense retrieval, RRF and convex fusion, cross-encoder reranking on SciFact, tuned on train only.
- README: the answer layer has not been run against the API.
