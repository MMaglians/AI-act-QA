# AI-act-QA

Question answering on the EU AI Act (Regulation (EU) 2024/1689) with cited sources. It uses Gemini embeddings, FAISS and BM25 with rank fusion, and a measured retrieval evaluation.

## What it does

You ask a question in English or Italian. The tool retrieves the most relevant passages of the regulation, asks a Gemini model to answer **only** from those passages, and checks that every citation (`[S1]`, `[S2]`...) points to a passage that was actually provided. If the passages do not contain the answer, the model must reply `NOT_FOUND`.

This is a learning project. It was not thinked as a legal advice.

## Quickstart

```powershell
uv sync
copy .env.example .env        # then write your GEMINI_API_KEY in .env
```

1. Save the consolidated English HTML of the regulation as `data/ai_act_en.html`
   (EUR-Lex, consolidated version of 27 July 2026, CELEX 02024R1689-20260727).
2. Build the passages and the vectors:

```powershell
uv run python -m ai_act_qa.cli chunks
uv run python -m ai_act_qa.cli embed
```

3. Ask a question:

```powershell
uv run python -m ai_act_qa.cli ask "Which AI practices are prohibited?" -k 5
uv run python -m ai_act_qa.cli search "Which AI practices are prohibited?"   # dense vs bm25 vs hybrid
uv run python -m ai_act_qa.cli eval                                           # retrieval evaluation
```

## How it works

1. **Chunking**: structure-aware, one passage per article or annex, split in parts of at most 220 words (361 chunks, 119 articles, 14 annexes).
2. **Dense retrieval**: `gemini-embedding-2` (768 dimensions), with different formats for documents and queries, exact search with FAISS `IndexFlatIP`.
3. **Lexical retrieval**: BM25 (Okapi, k1=1.5, b=0.75) written by hand, no stemming, English stop words only.
4. **Hybrid**: Reciprocal Rank Fusion (k=60) of the first 30 results of each list.
5. **Generation**: `gemini-3.5-flash-lite`, temperature 0, answers only from the retrieved passages, mechanical check of the citations.
6. **LlamaIndex version**: the same chunks and vectors, to compare a framework against the hand-written pipeline.

## Evaluation

30 hand-written questions (15 English, 15 Italian), each with the expected article or annex and a reference answer. Every expected article was checked against the text. Metric: **hit@k**, the fraction of questions for which at least one of the first *k* retrieved chunks belongs to the expected article. *k* counts chunks, not distinct articles. Intervals are Wilson 95%.

| mode | hit@1 | hit@3 | hit@5 | hit@10 |
|---|---|---|---|---|
| dense | 0.93 (0.79-0.98) | 1.00 (0.89-1.00) | 1.00 (0.89-1.00) | 1.00 (0.89-1.00) |
| bm25 | 0.40 (0.25-0.58) | 0.50 (0.33-0.67) | 0.57 (0.39-0.73) | 0.57 (0.39-0.73) |
| hybrid | 0.73 (0.56-0.86) | 0.90 (0.74-0.97) | 0.97 (0.83-0.99) | 1.00 (0.89-1.00) |

What the numbers say:

- **Dense retrieval is the best mode on this set; the hybrid does not beat it.** At k=1 dense finds 6 questions that hybrid misses and hybrid finds none (exact sign test, p about 0.03; multiple comparisons not corrected).
- **BM25 fails on Italian questions** (hit@1 0.07, n=15): the corpus is English and the tokenizer has no Italian handling, so BM25 matches only accidental words such as `da`, `i`, `e`, `per`. In hybrid, such a chunk can outrank the right one.
- The hybrid loses a few places for a second reason: a chunk present in both lists gets a bonus even if mediocre, and ties at rank 1 are broken by the chunk position in the text.
- Only one question (q27, Article 113) is not found by hybrid within k=5.

Full per-language and per-difficulty tables are produced by `eval` and saved in `eval/results.json`.

## Limitations

- **Small, easy set.** n=30, written by the author with the article titles in mind, and the document embedding includes the title. Dense reaches the ceiling (30/30 at k=3), so the set cannot separate good systems from very good ones.
- **Coarse gold.** The expected answer is an article or annex, not a passage. One question (q10, deployers' logs) is ambiguous: Article 19 (providers) is a legitimate competitor of the expected Article 26.
- **Not covered**: questions with identifiers or numbers, out-of-scope questions, other documents or versions, answer correctness. The citation check verifies that a cited source exists, not that it supports the sentence.
- **Generation is not evaluated**, only retrieval.
- **Not implemented**: PDF/OCR and tables, query rewriting, a cross-encoder reranker.
- **Possible fixes, to be tested on a separate question set** (not on these 30): break ties by dense rank, lower the BM25 weight, use BM25 only for English queries.

## Costs

One question with `-k 5` used `1378` input tokens and `493` output tokens. Prices change: see the [Gemini API pricing page](https://ai.google.dev/gemini-api/docs/pricing). Embeddings are computed once and cached in `.cache/`.

## Tests

```powershell
uv run pytest -q
```

97 tests, none of them calls the network.

## License

MIT, see `LICENSE`.

## Source and attribution

Contains text of Regulation (EU) 2024/1689, consolidated version of 27 July 2026, © European Union, 1998-2026, source EUR-Lex (eur-lex.europa.eu), licence CC BY 4.0. The text was processed: split into passages by article and cleaned of amendment markers. Only the version published in the Official Journal is authentic. This project is not legal advice.
