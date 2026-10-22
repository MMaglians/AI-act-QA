import pytest

from ai_act_qa.retrieval_metrics import hit_at_k, mean, recall_at_k


def test_hit_at_k_found_inside_k():
    assert hit_at_k(["a", "b", "c"], {"b"}, k=2) == 1.0


def test_hit_at_k_found_outside_k():
    assert hit_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0


def test_recall_at_k_partial():
    assert recall_at_k(["a", "b", "c"], {"a", "c", "z"}, k=3) == pytest.approx(2 / 3)


def test_recall_requires_relevant_docs():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), k=1)


def test_mean():
    assert mean([1.0, 0.0, 1.0, 1.0]) == 0.75