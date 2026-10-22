import hashlib

import numpy as np
import pytest
from google.genai import errors, types

from ai_act_qa.embed_texts import (
    GeminiEmbedder,
    embed_cached,
    format_document,
    format_query,
)

DIM = 8


class FakeEmbedding:
    def __init__(self, values):
        self.values = values


class FakeResult:
    def __init__(self, embeddings):
        self.embeddings = embeddings


def fake_vector(text: str) -> list[float]:
    """Vettore deterministico derivato dal testo (nessuna rete)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [b / 255 + 0.01 for b in digest[:DIM]]


class FakeModels:
    def __init__(self, fail_times: int = 0, fail_code: int = 429):
        self.calls = 0
        self.fail_times = fail_times
        self.fail_code = fail_code

    def embed_content(self, model, contents, config):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise errors.APIError(self.fail_code, {"error": {"message": "finto"}})
        # Il bug da evitare: stringhe semplici = un solo vettore aggregato.
        assert all(isinstance(c, types.Content) for c in contents)
        assert config.output_dimensionality == DIM
        texts = [c.parts[0].text for c in contents]
        return FakeResult([FakeEmbedding(fake_vector(t)) for t in texts])


class FakeClient:
    def __init__(self, **kwargs):
        self.models = FakeModels(**kwargs)


def make_embedder(client, **kwargs):
    return GeminiEmbedder(client, dim=DIM, sleep=lambda s: None, **kwargs)


def test_formats():
    assert format_document("ciao", "Titolo") == "title: Titolo | text: ciao"
    assert format_document("ciao") == "title: none | text: ciao"
    assert format_query("e' vero?") == "task: question answering | query: e' vero?"


def test_embed_returns_one_row_per_text():
    emb = make_embedder(FakeClient())
    vectors = emb.embed(["a", "b", "c"])
    assert vectors.shape == (3, DIM)


def test_embed_splits_into_batches():
    client = FakeClient()
    emb = make_embedder(client, batch_size=2)
    emb.embed(["a", "b", "c", "d", "e"])
    assert client.models.calls == 3  # 2 + 2 + 1


def test_embed_empty_list_makes_no_call():
    client = FakeClient()
    assert make_embedder(client).embed([]).shape == (0, DIM)
    assert client.models.calls == 0


def test_retry_after_429_then_success():
    client = FakeClient(fail_times=2, fail_code=429)
    vectors = make_embedder(client).embed(["a"])
    assert vectors.shape == (1, DIM)
    assert client.models.calls == 3


def test_non_retryable_error_is_raised_immediately():
    client = FakeClient(fail_times=5, fail_code=400)
    with pytest.raises(errors.APIError):
        make_embedder(client).embed(["a"])
    assert client.models.calls == 1


def test_gives_up_after_max_retries():
    client = FakeClient(fail_times=99, fail_code=503)
    with pytest.raises(errors.APIError):
        make_embedder(client, max_retries=3).embed(["a"])
    assert client.models.calls == 3


def test_cache_avoids_second_call(tmp_path):
    client = FakeClient()
    emb = make_embedder(client)
    cache = tmp_path / "emb.json"
    first = embed_cached(["a", "b"], emb, cache)
    calls_after_first = client.models.calls
    second = embed_cached(["a", "b"], emb, cache)
    assert client.models.calls == calls_after_first
    assert np.allclose(first, second)


def test_cache_embeds_only_missing_and_dedups(tmp_path):
    client = FakeClient()
    emb = make_embedder(client, batch_size=100)
    cache = tmp_path / "emb.json"
    embed_cached(["a"], emb, cache)
    client.models.calls = 0
    vectors = embed_cached(["a", "b", "b"], emb, cache)
    assert client.models.calls == 1
    assert vectors.shape == (3, DIM)
    assert np.allclose(vectors[1], vectors[2])