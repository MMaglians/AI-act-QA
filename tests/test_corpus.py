from tests.sample import SAMPLE
from ai_act_qa.corpus import (
    build_chunks,
    clean_lines,
    corpus_stats,
    load_chunks,
    parse_sections,
    save_chunks,
    split_lines,
)


def test_clean_lines_removes_markers_and_nbsp():
    lines = clean_lines("[\u25bcM1]\nrules\u00a0for   systems\n\u25baM1 text \u25c4\n")
    assert lines == ["rules for systems", "text"]


def test_parse_sections_finds_articles_and_annexes():
    refs = [s["ref"] for s in parse_sections(SAMPLE)]
    assert refs == ["Article 1", "Article 4a", "Article 5", "Article 6", "Annex I", "Annex III"]


def test_duplicate_article_keeps_the_longer_copy():
    sections = {s["ref"]: s for s in parse_sections(SAMPLE)}
    assert "functioning of the internal market" in " ".join(sections["Article 1"]["lines"])


def test_title_and_chapter_are_attached():
    sections = {s["ref"]: s for s in parse_sections(SAMPLE)}
    assert sections["Article 5"]["title"] == "Prohibited AI practices"
    assert sections["Article 5"]["chapter"] == "Chapter II - PROHIBITED PRACTICES"
    assert sections["Article 1"]["chapter"] == "Chapter I - GENERAL PROVISIONS"


def test_section_heading_does_not_leak_into_text():
    sections = {s["ref"]: s for s in parse_sections(SAMPLE)}
    assert not any("SECTION" in line or "Requirements" in line for line in sections["Article 5"]["lines"])
    assert sections["Article 6"]["chapter"] == "Chapter II - PROHIBITED PRACTICES"


def test_preamble_is_ignored():
    text = " ".join(" ".join(s["lines"]) for s in parse_sections(SAMPLE))
    assert "Whereas" not in text
    assert "Document 0" not in text


def test_markers_are_gone_from_chunk_text():
    chunks = build_chunks(SAMPLE)
    assert not any("\u25bc" in c.text or "\u25ba" in c.text or "\u25c4" in c.text for c in chunks)


def test_split_lines_packs_and_respects_max_words():
    lines = [" ".join(["w"] * 40) for _ in range(6)]
    pieces = split_lines(lines, max_words=100, overlap=10)
    assert len(pieces) == 3
    assert all(len(p.split()) <= 100 for p in pieces)


def test_split_lines_cuts_a_very_long_line():
    pieces = split_lines([" ".join(f"w{i}" for i in range(500))], max_words=100, overlap=10)
    assert len(pieces) > 1
    assert all(len(p.split()) <= 100 for p in pieces)


def test_long_article_gets_parts_and_labels():
    body = "\n".join(" ".join(["alpha"] * 60) for _ in range(10))
    raw = f"Article 3\nDefinitions\n{body}\n"
    chunks = build_chunks(raw, max_words=130, overlap=10)
    assert len(chunks) > 1
    assert chunks[0].id == "article-3-1"
    assert chunks[0].parts == len(chunks)
    assert chunks[0].label() == f"Article 3 (Definitions), part 1/{len(chunks)}"


def test_short_article_label_has_no_part():
    chunks = {c.ref: c for c in build_chunks(SAMPLE)}
    assert chunks["Article 5"].label() == "Article 5 (Prohibited AI practices)"


def test_stats_counts():
    stats = corpus_stats(build_chunks(SAMPLE))
    assert stats["articles"] == 4
    assert stats["annexes"] == 2
    assert stats["chunks"] == 6


def test_save_and_load_roundtrip(tmp_path):
    chunks = build_chunks(SAMPLE)
    path = tmp_path / "sub" / "chunks.json"
    save_chunks(chunks, path)
    assert load_chunks(path) == chunks