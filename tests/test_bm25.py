import math

import pytest

from ai_act_qa.bm25 import BM25, tokenize

DOCS = [
    "providers of high-risk AI systems shall ensure transparency",
    "prohibited AI practices include manipulative techniques",
    "the Commission shall adopt delegated acts",
    "transparency obligations for providers and deployers of AI systems",
]


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Providers SHALL ensure it!") == ["providers", "ensure"]


def test_relevant_document_ranks_first():
    bm = BM25(DOCS)
    assert bm.top_k("prohibited manipulative practices", k=1)[0][0] == 1


def test_unknown_terms_give_no_results():
    assert BM25(DOCS).top_k("zebra giraffe", k=3) == []


def test_zero_score_documents_are_not_returned():
    hits = BM25(DOCS).top_k("delegated", k=4)
    assert [i for i, _ in hits] == [2]


def test_rare_term_weighs_more_than_common_term():
    bm = BM25(DOCS)
    assert bm.idf["delegated"] > bm.idf["providers"]


def test_idf_formula():
    bm = BM25(DOCS)
    n, nt = 4, 1
    assert bm.idf["delegated"] == pytest.approx(math.log((n - nt + 0.5) / (nt + 0.5) + 1))


def test_shorter_document_wins_with_same_term_frequency():
    docs = ["alpha beta", "alpha " + " ".join(["filler"] * 30), "gamma"]
    hits = BM25(docs).top_k("alpha", k=2)
    assert [i for i, _ in hits] == [0, 1]


def test_repeated_query_terms_count_once():
    bm = BM25(DOCS)
    assert bm.scores("prohibited")[1] == pytest.approx(bm.scores("prohibited prohibited")[1])


def test_empty_corpus_is_safe():
    assert BM25([]).top_k("anything") == []