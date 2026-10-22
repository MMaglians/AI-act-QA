from types import SimpleNamespace

import pytest
from google.genai import errors

from ai_act_qa.corpus import Chunk
from ai_act_qa.generate import (
    NOT_FOUND,
    answer_question,
    build_prompt,
    extract_citations,
)


def make_chunk(ref: str, title: str, text: str) -> Chunk:
    return Chunk(id=ref.lower().replace(" ", "-") + "-1", kind="article", ref=ref, title=title,
                 chapter="", text=text, part=1, parts=1)


SOURCES = [
    make_chunk("Article 5", "Prohibited AI practices", "Manipulative techniques are prohibited."),
    make_chunk("Article 6", "Classification rules", "Systems under Annex III are high-risk."),
]


class FakeModels:
    def __init__(self, text="", fail_times=0, fail_code=429):
        self.text, self.calls = text, 0
        self.fail_times, self.fail_code = fail_times, fail_code
        self.last = None

    def generate_content(self, model, contents, config):
        self.calls += 1
        self.last = SimpleNamespace(model=model, contents=contents, config=config)
        if self.calls <= self.fail_times:
            raise errors.APIError(self.fail_code, {"error": {"message": "finto"}})
        usage = SimpleNamespace(prompt_token_count=120, candidates_token_count=15)
        return SimpleNamespace(text=self.text, usage_metadata=usage)


class FakeClient:
    def __init__(self, **kwargs):
        self.models = FakeModels(**kwargs)


def test_extract_citations_variants():
    assert extract_citations("A [S1]. B [S2][S3]. C [S1, S4].") == [1, 2, 3, 4]


def test_extract_citations_ignores_other_brackets():
    assert extract_citations("See [note] and (S2) and [Source 3].") == []


def test_build_prompt_numbers_sources_and_keeps_question():
    prompt = build_prompt("What is banned?", SOURCES)
    assert "[S1] Article 5 (Prohibited AI practices)" in prompt
    assert "[S2] Article 6 (Classification rules)" in prompt
    assert prompt.endswith("Question: What is banned?")


def test_answer_with_valid_citation():
    client = FakeClient(text="Manipulative techniques are banned [S1].")
    a = answer_question("What is banned?", SOURCES, client)
    assert a.found and a.cited == [1] and a.invalid == [] and not a.uncited
    assert (a.prompt_tokens, a.output_tokens) == (120, 15)
    assert a.sources[0].ref == "Article 5"


def test_invented_citation_is_flagged():
    a = answer_question("q", SOURCES, FakeClient(text="Yes [S1][S7]."))
    assert a.cited == [1] and a.invalid == [7]


def test_answer_without_citations_is_flagged():
    a = answer_question("q", SOURCES, FakeClient(text="Yes, it is banned."))
    assert a.found and a.uncited and a.cited == []


def test_not_found_marker():
    a = answer_question("q", SOURCES, FakeClient(text=NOT_FOUND))
    assert not a.found and not a.uncited and a.cited == []


def test_empty_model_answer_counts_as_not_found():
    a = answer_question("q", SOURCES, FakeClient(text=None))
    assert not a.found


def test_no_sources_means_no_api_call():
    client = FakeClient(text="x")
    a = answer_question("q", [], client)
    assert not a.found and client.models.calls == 0


def test_request_uses_zero_temperature_and_system_prompt():
    client = FakeClient(text="ok [S1]")
    answer_question("q", SOURCES, client, model="some-model")
    last = client.models.last
    assert last.model == "some-model"
    assert last.config.temperature == 0.0
    assert "ONLY the numbered excerpts" in last.config.system_instruction


def test_retry_then_success():
    client = FakeClient(text="ok [S1]", fail_times=2, fail_code=503)
    a = answer_question("q", SOURCES, client, sleep=lambda s: None)
    assert a.found and client.models.calls == 3


def test_non_retryable_error_is_raised():
    client = FakeClient(text="ok", fail_times=3, fail_code=400)
    with pytest.raises(errors.APIError):
        answer_question("q", SOURCES, client, sleep=lambda s: None)
    assert client.models.calls == 1