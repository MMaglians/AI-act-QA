import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.llms import MockLLM  # noqa: E402

from ai_act_qa.llama_version import build_index, make_query_engine, retrieve  # noqa: E402
from ai_act_qa.retrieval import HybridRetriever  # noqa: E402
from tests.test_kb import CHUNKS, DIM, VOCAB  # noqa: E402
from ai_act_qa.bm25 import tokenize  # noqa: E402

import numpy as np  # noqa: E402


def bag_of_words(text: str) -> np.ndarray:
    counts = np.zeros(DIM)
    for token in tokenize(text):
        if token in VOCAB:
            counts[VOCAB.index(token)] += 1
    return counts + 1e-3


VECTORS = np.array([bag_of_words(c.index_text()) for c in CHUNKS])


def test_framework_retrieval_matches_our_dense_retrieval():
    ours = HybridRetriever(CHUNKS, VECTORS, bag_of_words)
    index = build_index(CHUNKS, VECTORS, bag_of_words)
    question = "manipulative techniques prohibited"
    expected = [(CHUNKS[h.index].id, h.score) for h in ours.search(question, k=3, mode="dense")]
    got = retrieve(index, question, k=3)
    assert [i for i, _ in got] == [i for i, _ in expected]
    assert [s for _, s in got] == pytest.approx([s for _, s in expected], abs=1e-4)


def test_query_engine_returns_source_nodes():
    index = build_index(CHUNKS, VECTORS, bag_of_words)
    engine = make_query_engine(index, MockLLM(), k=2)
    response = engine.query("manipulative techniques prohibited")
    assert len(response.source_nodes) == 2
    assert response.source_nodes[0].node.metadata["label"].startswith("Article 5")