import pytest

from ai_act_qa.chunking import chunk_words


def make_text(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_short_text_is_one_chunk():
    assert chunk_words(make_text(5), size=10, overlap=2) == [make_text(5)]


def test_empty_text_gives_no_chunks():
    assert chunk_words("   ", size=10, overlap=2) == []


def test_no_chunk_exceeds_size():
    chunks = chunk_words(make_text(95), size=20, overlap=5)
    assert all(len(c.split()) <= 20 for c in chunks)


def test_overlap_is_shared_between_consecutive_chunks():
    chunks = chunk_words(make_text(50), size=20, overlap=5)
    first, second = chunks[0].split(), chunks[1].split()
    assert first[-5:] == second[:5]


def test_every_word_is_covered_and_order_kept():
    text = make_text(95)
    chunks = chunk_words(text, size=20, overlap=5)
    covered = []
    for i, c in enumerate(chunks):
        words = c.split()
        covered.extend(words if i == 0 else words[5:])
    assert " ".join(covered) == text


def test_exact_multiple_has_no_tiny_tail_chunk():
    chunks = chunk_words(make_text(35), size=20, overlap=5)
    assert len(chunks) == 2
    assert chunks[-1].split()[-1] == "w34"


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_parameters_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_words("a b c", size=size, overlap=overlap)