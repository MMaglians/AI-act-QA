import numpy as np
import pytest

from ai_act_qa.corpus import Chunk
from ai_act_qa.retrieval import HybridRetriever, rrf


def make_chunk(i: int, text: str) -> Chunk:
    return Chunk(id=f"c{i}", kind="article", ref=f"Article {i}", title="", chapter="",
                 text=text, part=1, parts=1)


def test_rrf_sums_reciprocal_ranks():
    fused = dict(rrf([[10, 20], [20, 30]], k=60))
    assert fused[20] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[10] == pytest.approx(1 / 61)
    assert fused[30] == pytest.approx(1 / 62)


def test_rrf_puts_documents_found_by_both_first():
    order = [i for i, _ in rrf([[1, 2, 3], [4, 2, 5]])]
    assert order[0] == 2


def test_rrf_empty():
    assert rrf([[], []]) == []


CHUNKS = [
    make_chunk(0, "alpha"),
    make_chunk(1, "beta"),
    make_chunk(2, "gamma"),
    make_chunk(3, "delta"),
]
# vettori: il chunk 0 e' vicino a [1, 0], il chunk 3 a [0, 1]
VECTORS = np.array([[1.0, 0.1], [0.5, 0.5], [0.4, 0.6], [0.1, 1.0]])


def fake_embed(query: str) -> np.ndarray:
    return np.array([1.0, 0.0]) if "first" in query else np.array([0.0, 1.0])


def test_dense_search_orders_by_cosine():
    r = HybridRetriever(CHUNKS, VECTORS, fake_embed)
    assert [h.index for h in r.search("first", k=2, mode="dense")] == [0, 1]
    assert [h.rank for h in r.search("first", k=2, mode="dense")] == [1, 2]


def test_bm25_mode_uses_words():
    r = HybridRetriever(CHUNKS, VECTORS, fake_embed)
    assert [h.index for h in r.search("gamma", k=3, mode="bm25")] == [2]


def test_hybrid_recovers_a_chunk_that_only_bm25_finds():
    # La domanda "delta first" e' vicina ai vettori [1,0] (chunk 0 e 1)
    # ma la parola "delta" compare solo nel chunk 3.
    r = HybridRetriever(CHUNKS, VECTORS, fake_embed)
    dense_top = [h.index for h in r.search("delta first", k=2, mode="dense")]
    hybrid_top = [h.index for h in r.search("delta first", k=3, mode="hybrid")]
    assert 3 not in dense_top
    assert 3 in hybrid_top


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        HybridRetriever(CHUNKS, VECTORS, fake_embed).search("x", mode="magic")


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        HybridRetriever(CHUNKS, VECTORS[:2], fake_embed)


def test_k_larger_than_corpus_is_capped():
    r = HybridRetriever(CHUNKS, VECTORS, fake_embed)
    assert len(r.search("first", k=50, mode="dense")) == 4