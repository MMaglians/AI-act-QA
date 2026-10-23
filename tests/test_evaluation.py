import json
from types import SimpleNamespace

import pytest

from ai_act_qa.corpus import Chunk
from ai_act_qa.evaluation import (
    Question,
    compare,
    first_gold_rank,
    load_questions,
    normalize_text,
    run_retrieval,
    summarize,
    subset,
    validate_questions,
    wilson,
)


def make_chunk(ref: str, text: str) -> Chunk:
    return Chunk(id=ref.lower().replace(" ", "-") + "-1", kind="article", ref=ref, title="",
                 chapter="", text=text, part=1, parts=1)


CHUNKS = [
    make_chunk("Article 1", "The purpose is to improve the internal market."),
    make_chunk("Article 2", "This Regulation applies to providers."),
    make_chunk("Article 3", "'Deep fake' means manipulated content."),
]


def make_question(qid="q1", gold=("Article 2",), evidence=("applies to providers",), **kw) -> Question:
    base = dict(id=qid, lang="en", difficulty="easy", question="who?", gold=list(gold),
                answer="providers", evidence=list(evidence))
    base.update(kw)
    return Question(**base)


class FakeRetriever:
    """Restituisce classifiche decise dal test, una per modalita'."""

    def __init__(self, orders):
        self.chunks = CHUNKS
        self.orders = orders  # mode -> lista di indici di chunk

    def search(self, query, k=5, mode="hybrid"):
        return [SimpleNamespace(index=i, score=1.0, rank=r)
                for r, i in enumerate(self.orders[mode][:k], start=1)]


def test_wilson_known_value():
    lo, hi = wilson(27, 30)
    assert lo == pytest.approx(0.7438, abs=1e-3)
    assert hi == pytest.approx(0.9654, abs=1e-3)


def test_wilson_edges():
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(30, 30)
    assert hi == pytest.approx(1.0) and lo < 1.0


def test_normalize_text_handles_quotes_case_and_spaces():
    assert normalize_text("\u2018Deep  FAKE\u2019\xa0means") == "'deep fake' means"


def test_valid_questions_have_no_problems():
    assert validate_questions([make_question()], CHUNKS) == []


def test_validation_catches_missing_reference_and_evidence():
    problems = validate_questions(
        [make_question("a", gold=("Article 9",)), make_question("b", evidence=("not in the text",))],
        CHUNKS,
    )
    assert any("a:" in p and "non esiste" in p for p in problems)
    assert any("b:" in p and "evidenza non trovata" in p for p in problems)


def test_validation_catches_duplicate_ids_and_bad_fields():
    problems = validate_questions(
        [make_question("x"), make_question("x", lang="fr", difficulty="tough", answer="")], CHUNKS)
    assert any("duplicato" in p for p in problems)
    assert any("lingua" in p for p in problems)
    assert any("difficolta" in p for p in problems)
    assert any("vuota" in p for p in problems)


def test_evidence_matches_despite_curly_quotes():
    q = make_question("c", gold=("Article 3",), evidence=("'deep fake' means manipulated",))
    chunks = CHUNKS[:2] + [make_chunk("Article 3", "\u2018Deep fake\u2019 means manipulated content.")]
    assert validate_questions([q], chunks) == []


def test_summarize_hit_and_recall():
    questions = [make_question("q1", gold=("Article 2",)),
                 make_question("q2", gold=("Article 1", "Article 3"))]
    retriever = FakeRetriever({"dense": [1, 0, 2], "bm25": [0, 1, 2]})
    runs = run_retrieval(retriever, questions, modes=("dense", "bm25"), max_k=3)
    s = summarize(runs, questions, ks=(1, 3))
    assert s["dense"][1]["hits"] == 1  # q1 trova Article 2 al primo posto
    assert s["dense"][3]["hits"] == 2
    assert s["dense"][1]["recall"] == pytest.approx((1.0 + 0.0) / 2)
    assert s["dense"][3]["recall"] == pytest.approx((1.0 + 1.0) / 2)  # q2 trova entrambi i riferimenti
    assert s["bm25"][1]["hits"] == 1  # q2 trova Article 1 al primo posto, q1 no
    lo, hi = s["dense"][3]["ci"]
    assert lo < s["dense"][3]["hit"] <= hi


def test_first_gold_rank():
    assert first_gold_rank(["Article 1", "Article 2"], ["Article 2"]) == 2
    assert first_gold_rank(["Article 1"], ["Article 3"]) is None


def test_compare_lists_questions_found_by_only_one_mode():
    questions = [make_question("q1", gold=("Article 2",)), make_question("q2", gold=("Article 1",))]
    runs = {"dense": {"q1": ["Article 2"], "q2": ["Article 3"]},
            "bm25": {"q1": ["Article 3"], "q2": ["Article 1"]}}
    assert compare(runs, questions, "dense", "bm25", 1) == (["q1"], ["q2"])


def test_subset_filters_by_field():
    questions = [make_question("q1", lang="en"), make_question("q2", lang="it")]
    assert [q.id for q in subset(questions, lang="it")] == ["q2"]


def test_load_questions_roundtrip(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([make_question().__dict__]), encoding="utf-8")
    assert load_questions(path)[0].gold == ["Article 2"]
    
