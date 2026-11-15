from types import SimpleNamespace

import pytest

from hybridrag.engine import SearchEngine
from hybridrag.rag import (
    DEFAULT_MODEL,
    FALLBACK_BETA,
    NOT_ENOUGH_EVIDENCE,
    SYSTEM_PROMPT,
    answer,
    parse_citations,
)


class FakeMessages:
    def __init__(self, text="", stop_reason="end_turn", stop_details=None):
        self.text, self.stop_reason, self.stop_details = text, stop_reason, stop_details
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = [SimpleNamespace(type="thinking", thinking="")]
        if self.text:
            content.append(SimpleNamespace(type="text", text=self.text))
        return SimpleNamespace(
            content=content,
            stop_reason=self.stop_reason,
            stop_details=self.stop_details,
            model=kwargs["model"],
        )


class FakeClient:
    """Stands in for anthropic.Anthropic(): records requests, never touches the network."""

    def __init__(self, **kwargs):
        self.beta = SimpleNamespace(messages=FakeMessages(**kwargs))

    @property
    def calls(self):
        return self.beta.messages.calls


@pytest.fixture
def engine(corpus, fake_encoder):
    return SearchEngine(corpus, encoder=fake_encoder)


def test_prompt_contains_passages_and_citation_rules(engine, monkeypatch):
    monkeypatch.delenv("HYBRIDRAG_MODEL", raising=False)
    client = FakeClient(text="Aspirin lowers ischemic stroke risk [d4].")
    result = answer("Does aspirin prevent stroke?", k=3, engine=engine, client=client)

    (request,) = client.calls
    user = request["messages"][0]["content"]
    for hit in result.passages:
        assert f'<passage id="{hit.doc_id}">' in user and hit.text in user
    assert user.rstrip().endswith("Question: Does aspirin prevent stroke?")
    assert "[4983]" in request["system"] and NOT_ENOUGH_EVIDENCE in request["system"]
    assert request["system"] == SYSTEM_PROMPT
    assert request["model"] == DEFAULT_MODEL
    assert request["output_config"] == {"effort": "medium"}
    assert "temperature" not in request  # sampling params are rejected by current models
    assert request["betas"] == [FALLBACK_BETA]
    assert request["extra_body"] == {"fallbacks": "default"}

    assert len(result.passages) == 3 and result.passages[0].doc_id == "d4"
    assert result.citations == ["d4"] and result.invalid_citations == []
    assert result.text == "Aspirin lowers ischemic stroke risk [d4]."
    assert not result.insufficient_evidence


def test_model_from_env_var_and_no_fallback_for_other_models(engine, monkeypatch):
    monkeypatch.setenv("HYBRIDRAG_MODEL", "claude-haiku-4-5")
    client = FakeClient(text="x [d1]")
    result = answer("sleep memory", engine=engine, client=client, effort=None)
    request = client.calls[0]
    assert request["model"] == "claude-haiku-4-5" == result.model
    assert "betas" not in request and "extra_body" not in request
    assert "output_config" not in request  # effort omitted for models without it
    answer("sleep memory", engine=engine, client=client, model="claude-sonnet-5-5")
    assert client.calls[1]["model"] == "claude-sonnet-5-5"


def test_citations_parsing_flags_ids_not_retrieved(engine):
    client = FakeClient(text="Salt raises pressure [d10, d99]; see also [d10][doc d2].")
    result = answer("salt and blood pressure", k=2, engine=engine, client=client)
    retrieved = [h.doc_id for h in result.passages]
    assert "d10" in retrieved
    assert result.citations[0] == "d10"
    assert "d99" in result.invalid_citations


def test_parse_citations_formats():
    cited, invalid = parse_citations(
        "A [1, 2]. B [3;4] C [2][5] D [doc 6] [x]", ["1", "2", "3", "6"]
    )
    assert cited == ["1", "2", "3", "6"]
    assert invalid == ["4", "5", "x"]
    assert parse_citations("no brackets here", ["1"]) == ([], [])


def test_not_enough_evidence_flag(engine):
    text = "Not enough evidence. The passages do not discuss coffee and sleep."
    result = answer("coffee and sleep?", engine=engine, client=FakeClient(text=text))
    assert result.insufficient_evidence and result.citations == []


def test_refusal_is_reported_not_parsed(engine):
    client = FakeClient(stop_reason="refusal", stop_details=SimpleNamespace(category="bio"))
    result = answer("anything", engine=engine, client=client)
    assert result.stop_reason == "refusal" and "bio" in result.text and result.citations == []
