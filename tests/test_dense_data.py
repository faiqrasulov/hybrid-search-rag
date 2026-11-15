import json
import zipfile

import numpy as np
import pytest

from hybridrag.data import download_scifact, load_beir, read_qrels
from hybridrag.dense import DenseRetriever, embed_corpus


def test_dense_exact_cosine_search(corpus, fake_encoder):
    docs = corpus.documents()
    dense = DenseRetriever.fit(fake_encoder, docs)
    assert dense.doc_embeddings.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(dense.doc_embeddings, axis=1), 1.0, atol=1e-6)
    # each document is its own nearest neighbour
    top = [r.indices[0] for r in dense.rank(docs, k=3)]
    assert top == list(range(len(docs)))
    sims = dense.score(["vitamin bone"])
    assert sims.shape == (1, len(docs)) and np.all(sims <= 1 + 1e-6)


def test_embedding_cache_is_reused_and_keyed_on_texts(tmp_path, corpus, fake_encoder):
    docs = corpus.documents()
    first = embed_corpus(fake_encoder, docs, cache_dir=tmp_path, chunk_size=5)
    calls = fake_encoder.calls
    assert calls == 3  # 12 docs in chunks of 5
    again = embed_corpus(fake_encoder, docs, cache_dir=tmp_path)
    assert fake_encoder.calls == calls  # loaded from the .npy cache
    np.testing.assert_array_equal(first, again)
    embed_corpus(fake_encoder, docs[:-1], cache_dir=tmp_path)
    assert fake_encoder.calls > calls  # different texts -> different cache key
    assert len(list(tmp_path.glob("*.npy"))) == 2


def _write_beir(folder):
    (folder / "qrels").mkdir(parents=True)
    rows = [{"_id": "10", "title": "T", "text": "alpha"}, {"_id": "11", "title": "", "text": "b"}]
    (folder / "corpus.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (folder / "queries.jsonl").write_text(json.dumps({"_id": "1", "text": "q one"}) + "\n")
    (folder / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\n1\t10\t1\n")
    (folder / "qrels" / "train.tsv").write_text("query-id\tcorpus-id\tscore\n9\t11\t1\n")


def test_load_beir_folder(tmp_path):
    _write_beir(tmp_path / "toy")
    ds = load_beir(tmp_path / "toy")
    assert ds.corpus.documents() == ["T. alpha", "b"]
    qids, texts, qrels = ds.split("test")
    assert (qids, texts, qrels) == (["1"], ["q one"], {"1": {"10": 1}})
    assert ds.split("train")[0] == []  # judged query without text is skipped
    assert read_qrels(tmp_path / "toy" / "qrels" / "test.tsv") == {"1": {"10": 1}}


def test_download_extracts_known_members_from_a_local_archive(tmp_path):
    src = tmp_path / "src"
    _write_beir(src / "scifact")
    archive = tmp_path / "scifact.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in (src / "scifact").rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(src))
        zf.writestr("../evil.txt", "should never be written")
    out = download_scifact(tmp_path / "data", url=archive.as_uri())
    assert (out / "corpus.jsonl").exists() and (out / "qrels" / "test.tsv").exists()
    assert not (tmp_path / "evil.txt").exists()


def test_download_fails_with_clear_message_when_offline(tmp_path):
    with pytest.raises(RuntimeError, match="Could not download SciFact"):
        download_scifact(tmp_path, url="http://127.0.0.1:9/scifact.zip")
    assert not list(tmp_path.glob("*.part"))
