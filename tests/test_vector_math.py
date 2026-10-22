import numpy as np
import pytest

from ai_act_qa.vector_math import cosine_similarity, normalize, top_k


def test_cosine_identical_vectors_is_one():
    assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_opposite_is_minus_one():
    assert cosine_similarity([1, 2], [-1, -2]) == pytest.approx(-1.0)


def test_cosine_ignores_length():
    assert cosine_similarity([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)


def test_cosine_zero_vector_raises():
    with pytest.raises(ValueError):
        cosine_similarity([0, 0], [1, 1])


def test_normalize_rows_have_unit_norm():
    m = normalize(np.array([[3.0, 4.0], [1.0, 1.0]]))
    assert np.linalg.norm(m, axis=1) == pytest.approx([1.0, 1.0])


def test_normalize_zero_vector_raises():
    with pytest.raises(ValueError):
        normalize(np.array([[0.0, 0.0]]))


def test_dot_of_normalized_equals_cosine():
    a, b = np.array([1.0, 2.0, 3.0]), np.array([3.0, 1.0, 0.5])
    dot = float(normalize(a) @ normalize(b))
    assert dot == pytest.approx(cosine_similarity(a, b), abs=1e-6)


def test_top_k_orders_by_similarity():
    matrix = np.array([[0.0, 1.0], [1.0, 0.1], [1.0, 1.0]])
    idx, scores = top_k(np.array([1.0, 0.0]), matrix, k=3)
    assert list(idx) == [1, 2, 0]
    assert scores[0] > scores[1] > scores[2]


def test_top_k_larger_than_corpus_is_capped():
    idx, _ = top_k(np.array([1.0, 0.0]), np.array([[1.0, 0.0], [0.0, 1.0]]), k=10)
    assert len(idx) == 2