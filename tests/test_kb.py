"""Test della knowledge base con un client simulato: nessuna rete."""

from types import SimpleNamespace

import numpy as np
import pytest
from google.genai import errors

from ai_act_qa.bm25 import tokenize
from ai_act_qa.corpus import build_chunks
from ai_act_qa.embed_texts import GeminiEmbedder
from ai_act_qa.generate import answer_question
from ai_act_qa.kb import embed_chunks, make_query_embedder
from ai_act_qa.retrieval import HybridRetriever
from tests.sample import SAMPLE

CHUNKS = build_chunks(SAMPLE)
VOCAB = sorted({t for c in CHUNKS for t in tokenize(c.index_text())})
DIM = len(VOCAB)


class BagOfWordsModels:
    """Finto 'embedding': conta le parole del vocabolario. Simula la ricerca semantica."""

    def __init__(self, fail_on_call=None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def embed_content(self, model, contents, config):
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise errors.APIError(400, {"error": {"message": "finto"}})
        rows = []
        for content in contents:
            counts = np.zeros(DIM)
            for token in tokenize(content.parts[0].text):
                if token in VOCAB:
                    counts[VOCAB.index(token)] += 1
            rows.append(SimpleNamespace(values=(counts + 1e-3).tolist()))
        return SimpleNamespace(embeddings=rows)


def make_embedder(models):
    return GeminiEmbedder(SimpleNamespace(models=models), dim=DIM, sleep=lambda s: None)


def test_end_to_end_retrieval(tmp_path):
    embedder = make_embedder(BagOfWordsModels())
    vectors = embed_chunks(CHUNKS, embedder, tmp_path / "docs.json", group=2, log=lambda m: None)
    assert vectors.shape == (len(CHUNKS), DIM)

    retriever = HybridRetriever(CHUNKS, vectors, make_query_embedder(embedder, tmp_path / "q.json"))
    hits = retriever.search("manipulative techniques prohibited", k=3, mode="hybrid")
    assert CHUNKS[hits[0].index].ref == "Article 5"
    
    
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: SimpleNamespace(
        text="They are banned [S1].", usage_metadata=None)))
    answer = answer_question("What is banned?", [CHUNKS[h.index] for h in hits], client)
    assert answer.found and answer.cited == [1]
    assert answer.sources[0].ref == "Article 5"


def test_embedding_resumes_after_a_failure(tmp_path):
    cache = tmp_path / "docs.json"
    models = BagOfWordsModels(fail_on_call=2)  # il secondo gruppo fallisce
    with pytest.raises(errors.APIError):
        embed_chunks(CHUNKS, make_embedder(models), cache, group=2, log=lambda m: None)
    done_before = models.calls

    models2 = BagOfWordsModels()
    embed_chunks(CHUNKS, make_embedder(models2), cache, group=2, log=lambda m: None)
    total_groups = -(-len(CHUNKS) // 2)
    assert done_before == 2
    assert models2.calls == total_groups - 1  # il primo gruppo era gia' in cache


def test_pause_only_after_groups_that_called_the_api(tmp_path):
    cache = tmp_path / "docs.json"
    pauses = []
    embed_chunks(CHUNKS, make_embedder(BagOfWordsModels()), cache, group=2, pause=5,
                 log=lambda m: None, sleep=pauses.append)
    groups = -(-len(CHUNKS) // 2)
    assert pauses == [5] * (groups - 1)  # niente pausa dopo l'ultimo gruppo

    pauses.clear()
    embed_chunks(CHUNKS, make_embedder(BagOfWordsModels()), cache, group=2, pause=5,
                 log=lambda m: None, sleep=pauses.append)
    assert pauses == []  # tutto in cache: nessuna attesa