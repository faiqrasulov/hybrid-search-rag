import pytest

from hybridrag import cli
from hybridrag.engine import SearchEngine


@pytest.fixture
def tiny_engine(monkeypatch, corpus, fake_encoder):
    engine = SearchEngine(corpus, encoder=fake_encoder)
    monkeypatch.setattr(cli, "_build_engine", lambda data_dir, system: engine)
    return engine


def test_search_prints_ranked_hits(tiny_engine, capsys):
    assert (
        cli.main(
            ["search", "measles vaccine for children and adults", "--system", "bm25", "-k", "2"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert out.lstrip().startswith("1. [d9] Measles vaccine")
    assert " 2. [" in out


def test_answer_without_api_key_shows_passages_and_never_calls_the_api(
    tiny_engine, monkeypatch, capsys
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def boom(*args, **kwargs):
        raise AssertionError("answer() must not be called without a key")

    monkeypatch.setattr("hybridrag.rag.answer", boom)
    assert cli.main(["answer", "does smoking cause lung cancer?", "-k", "3"]) == 0
    out = capsys.readouterr().out
    assert "[d3] Smoking and lung cancer" in out
    assert "ANTHROPIC_API_KEY is not set" in out
